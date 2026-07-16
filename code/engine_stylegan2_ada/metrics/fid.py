"""Frechet Inception Distance (pragmatic, InceptionV3 pool3 features).

Downloads torchvision's InceptionV3 weights once. Grayscale slices are replicated to
3 channels and resized to 299. This tracks relative quality well; absolute values are
comparable across runs of *this* code, not against the official TF FID.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class _Inception(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models import inception_v3
        try:
            from torchvision.models import Inception_V3_Weights
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        except Exception:
            net = inception_v3(pretrained=True, aux_logits=True)
        net.fc = nn.Identity()   # -> forward returns 2048-d pooled features in eval mode
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    @torch.no_grad()
    def features(self, x01):
        # x01: (B,1 or 3,H,W) in [0,1]
        if x01.shape[1] == 1:
            x01 = x01.repeat(1, 3, 1, 1)
        x = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        out = self.net(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def _stats(feats):
    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def _frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    diff = mu1 - mu2
    # Newer SciPy dropped the `disp`/`blocksize` kwargs and returns just the matrix.
    def _sqrtm(a):
        r = linalg.sqrtm(a)
        return r[0] if isinstance(r, tuple) else r
    covmean = _sqrtm(s1.dot(s2))
    # Covariance from few samples is near-singular -> sqrtm can be non-finite; nudge it.
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * 1e-6
        covmean = _sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


@torch.no_grad()
def compute_fid(G, ds, cfg, device, num=2000, batch=32):
    inc = _Inception(device)
    z_dim = _cfg_get(cfg, "z_dim", 512)
    num = min(num, len(ds))

    # real features
    real_feats = []
    idxs = torch.randperm(len(ds))[:num].tolist()
    for i in range(0, num, batch):
        chunk = idxs[i:i + batch]
        imgs = torch.stack([ds[j][0] for j in chunk]).to(device)  # (b,1,H,W) in [-1,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        real_feats.append(inc.features(imgs).cpu().numpy())
    real_feats = np.concatenate(real_feats, axis=0)

    # fake features
    fake_feats = []
    done = 0
    G.eval()
    while done < num:
        b = min(batch, num - done)
        z = torch.randn(b, z_dim, device=device)
        imgs = G(z, truncation_psi=1.0, noise_mode="const")
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        fake_feats.append(inc.features(imgs).cpu().numpy())
        done += b
    fake_feats = np.concatenate(fake_feats, axis=0)

    mu_r, s_r = _stats(real_feats)
    mu_f, s_f = _stats(fake_feats)
    return _frechet(mu_r, s_r, mu_f, s_f)
