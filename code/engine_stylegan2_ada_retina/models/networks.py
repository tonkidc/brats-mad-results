"""StyleGAN2 Generator + Discriminator in pure PyTorch (no custom ops).

Faithful to StyleGAN2 in the parts that matter for quality:
  * equalized learning rate on every learned layer
  * mapping network with pixel-norm input + w-average truncation
  * modulated / demodulated convolutions (weight demodulation)
  * per-pixel noise injection
  * skip-connection ToRGB generator, residual discriminator
  * minibatch-stddev before the discriminator epilogue

Deliberate simplifications for portability (DirectML/MPS/CPU) and brevity:
  * bilinear up/down-sampling instead of FIR (upfirdn2d) filtering
  * a single w per image (no per-layer style mixing)
  * no path-length regularization
These cost a little fidelity but keep everything to stock PyTorch ops.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _act(x):
    # leaky-relu with the StyleGAN2 gain so activation variance is preserved
    return F.leaky_relu(x, 0.2) * np.sqrt(2.0)


class EqLinear(nn.Module):
    """Fully-connected layer with equalized learning rate."""

    def __init__(self, in_f, out_f, bias=True, lr_mul=1.0, activation=False, bias_init=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_f,), float(bias_init))) if bias else None
        self.scale = (1.0 / np.sqrt(in_f)) * lr_mul
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        b = self.bias * self.lr_mul if self.bias is not None else None
        x = F.linear(x, self.weight * self.scale, b)
        return _act(x) if self.activation else x


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=2, w_avg_beta=0.995):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.w_avg_beta = w_avg_beta
        layers = []
        f = z_dim
        for _ in range(num_layers):
            layers.append(EqLinear(f, w_dim, lr_mul=0.01, activation=True))
            f = w_dim
        self.net = nn.Sequential(*layers)
        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(self, z, truncation_psi=1.0, update_emas=False):
        x = z * torch.rsqrt(z.pow(2).mean(dim=1, keepdim=True) + 1e-8)  # pixel norm
        w = self.net(x)
        if update_emas:
            # keep the running mean in fp32 even under autocast (w may be fp16)
            self.w_avg.copy_(w.detach().float().mean(0).lerp(self.w_avg, self.w_avg_beta))
        if truncation_psi != 1.0:
            w = self.w_avg.to(w.dtype).lerp(w, truncation_psi)
        return w


class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, kernel=3, up=False, demodulate=True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.padding = kernel // 2
        self.up = up
        self.demodulate = demodulate
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.scale = 1.0 / np.sqrt(in_ch * kernel * kernel)
        self.affine = EqLinear(w_dim, in_ch, bias=True, bias_init=1.0)  # style, init to 1

    def forward(self, x, w):
        B, C, H, Wd = x.shape
        style = self.affine(w)  # (B, in_ch)
        weight = self.scale * self.weight.unsqueeze(0)          # (1, out, in, k, k)
        weight = weight * style.view(B, 1, C, 1, 1)            # (B, out, in, k, k)
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8)  # (B, out)
            weight = weight * demod.view(B, self.out_ch, 1, 1, 1)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            H, Wd = x.shape[2], x.shape[3]
        x = x.reshape(1, B * C, H, Wd)
        weight = weight.view(B * self.out_ch, C, self.kernel, self.kernel)
        out = F.conv2d(x, weight, padding=self.padding, groups=B)
        return out.view(B, self.out_ch, out.shape[2], out.shape[3])


class SynthesisLayer(nn.Module):
    def __init__(self, in_ch, out_ch, w_dim, up=False):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, w_dim, kernel=3, up=up)
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, w, noise_mode="random"):
        x = self.conv(x, w)
        if noise_mode != "none":
            noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_strength
        x = x + self.bias.view(1, -1, 1, 1)
        # conv_clamp=256 as in official StyleGAN2-ADA: prevents fp16 activation
        # overflow (-> NaN) when training under autocast; a no-op at fp32 scales.
        return _act(x).clamp(-256, 256)


class ToRGB(nn.Module):
    def __init__(self, in_ch, w_dim, img_channels):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, img_channels, w_dim, kernel=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(img_channels))

    def forward(self, x, w):
        return (self.conv(x, w) + self.bias.view(1, -1, 1, 1)).clamp(-256, 256)


def _nf(res, channel_base, channel_max):
    return int(min(channel_base // res, channel_max))


class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, img_channels=1, channel_base=16384, channel_max=512):
        super().__init__()
        res_log2 = int(np.log2(img_resolution))
        assert 2 ** res_log2 == img_resolution and img_resolution >= 4, "resolution must be a power of two >= 4"
        self.w_dim = w_dim
        self.img_channels = img_channels
        resolutions = [2 ** i for i in range(2, res_log2 + 1)]  # 4, 8, ..., res

        c4 = _nf(4, channel_base, channel_max)
        self.const = nn.Parameter(torch.randn(1, c4, 4, 4))
        self.first_layer = SynthesisLayer(c4, c4, w_dim, up=False)
        self.first_torgb = ToRGB(c4, w_dim, img_channels)

        self.blocks = nn.ModuleList()
        self.torgbs = nn.ModuleList()
        in_ch = c4
        for res in resolutions[1:]:  # 8 .. res
            out_ch = _nf(res, channel_base, channel_max)
            self.blocks.append(nn.ModuleList([
                SynthesisLayer(in_ch, out_ch, w_dim, up=True),
                SynthesisLayer(out_ch, out_ch, w_dim, up=False),
            ]))
            self.torgbs.append(ToRGB(out_ch, w_dim, img_channels))
            in_ch = out_ch

    def forward(self, w, noise_mode="random"):
        B = w.shape[0]
        x = self.const.repeat(B, 1, 1, 1)
        x = self.first_layer(x, w, noise_mode)
        img = self.first_torgb(x, w)
        for (l0, l1), torgb in zip(self.blocks, self.torgbs):
            x = l0(x, w, noise_mode)
            x = l1(x, w, noise_mode)
            img = F.interpolate(img, scale_factor=2, mode="bilinear", align_corners=False)
            img = img + torgb(x, w)
        return img


class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, img_resolution=128, img_channels=1,
                 mapping_layers=2, channel_base=16384, channel_max=512):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        self.synthesis = SynthesisNetwork(w_dim, img_resolution, img_channels, channel_base, channel_max)

    def forward(self, z, truncation_psi=1.0, noise_mode="random", update_emas=False):
        w = self.mapping(z, truncation_psi=truncation_psi, update_emas=update_emas)
        return self.synthesis(w, noise_mode=noise_mode)


# --------------------------------------------------------------------------------------
# Discriminator
# --------------------------------------------------------------------------------------

class EqConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale = 1.0 / np.sqrt(in_ch * kernel * kernel)
        self.padding = kernel // 2

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class DBlock(nn.Module):
    """Residual discriminator block: two convs + 2x downsample, with a 1x1 skip."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv0 = EqConv2d(in_ch, in_ch, 3)
        self.conv1 = EqConv2d(in_ch, out_ch, 3)
        self.skip = EqConv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        y = _act(self.conv0(x))
        y = _act(self.conv1(y))
        y = F.avg_pool2d(y, 2)
        s = F.avg_pool2d(x, 2)
        s = self.skip(s)
        return (y + s) * (1.0 / np.sqrt(2.0))


class MinibatchStd(nn.Module):
    def __init__(self, group_size=4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.group_size
        while B % g != 0:
            g -= 1
        y = x.view(g, B // g, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.pow(2).mean(dim=0) + 1e-8).sqrt()      # (B//g, C, H, W)
        y = y.mean(dim=[1, 2, 3], keepdim=True)        # (B//g, 1, 1, 1)
        y = y.repeat(g, 1, H, W)                        # (B, 1, H, W)
        return torch.cat([x, y], dim=1)


class Discriminator(nn.Module):
    def __init__(self, img_resolution=128, img_channels=1, channel_base=16384, channel_max=512):
        super().__init__()
        res_log2 = int(np.log2(img_resolution))
        self.from_rgb = EqConv2d(img_channels, _nf(img_resolution, channel_base, channel_max), 1)
        blocks = []
        in_ch = _nf(img_resolution, channel_base, channel_max)
        for res in [2 ** i for i in range(res_log2, 2, -1)]:  # res .. 8
            out_ch = _nf(res // 2, channel_base, channel_max)
            blocks.append(DBlock(in_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.mbstd = MinibatchStd(4)
        c4 = _nf(4, channel_base, channel_max)
        self.conv = EqConv2d(in_ch + 1, c4, 3)
        self.fc0 = EqLinear(c4 * 4 * 4, c4, activation=True)
        self.fc1 = EqLinear(c4, 1)

    def forward(self, x):
        x = _act(self.from_rgb(x))
        x = self.blocks(x)
        x = self.mbstd(x)
        x = _act(self.conv(x))
        x = x.flatten(1)
        x = self.fc0(x)
        return self.fc1(x)
