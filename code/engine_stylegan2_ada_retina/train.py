"""Training loop for pure-PyTorch StyleGAN2-ADA.

Non-saturating logistic GAN loss, lazy R1 regularization (gated by device caps),
EMA generator, ADA p-controller, periodic image samples + checkpoints, and optional
Weights & Biases logging including FID.

Usage (from the notebook):
    from stylegan2_ada.train import Config, train
    cfg = Config(data='data/brats_slices', out='runs/brats', resolution=128,
                 batch=16, kimg=200)
    train(cfg)
"""

import copy
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import AugmentPipe
from .data.dataset import SliceDataset, RetinaPatchDataset
from .device import pick_device, probe, describe
from .models.networks import Generator, Discriminator


@dataclass
class Config:
    data: str                      # folder of PNG slices (brats) or .npy patches (retina)
    out: str = "runs/brats"        # output dir
    dataset: str = "brats"         # 'brats' -> grayscale SliceDataset; 'retina' -> 7ch RetinaPatchDataset
    color_channels: int = None     # image channels that take photometric aug (retina: 3); None -> all
    resolution: int = 128
    batch: int = 16
    kimg: int = 200                # training length in thousands of images
    z_dim: int = 512
    w_dim: int = 512
    channel_base: int = 16384      # ~fmaps=0.5; raise to 32768 for a bigger model
    channel_max: int = 512
    mapping_layers: int = 2
    lr: float = 0.0025
    r1_gamma: float = None         # None -> heuristic 0.0002 * res^2 / batch
    r1_interval: int = 16          # lazy R1 every N steps
    r1_cudnn: bool = None          # cuDNN/MIOpen for R1's conv double-backward.
                                   # None -> auto: off on ROCm (MIOpen's im2col kernel
                                   # fails to compile for gfx1101 at >=256px, so route
                                   # the double-backward through native ATen conv), on
                                   # elsewhere. Set True/False to override.
    ada_target: float = 0.6
    ada_interval: int = 4          # update p every N steps
    ada_kimg: int = 500            # images (in k) for p to traverse the full [0,1] range
    p_init: float = 0.0            # initial ADA strength (start >0 to fight instant D-overfit on tiny data)
    ema_kimg: float = 10.0         # G_ema half-life in kimg
    sample_every_kimg: int = 20
    ckpt_every_kimg: int = 100
    fid_every_kimg: int = 0        # 0 disables periodic FID (still logged at end if wandb)
    fid_num: int = 2000
    workers: int = 0
    xflip: bool = True
    amp: bool = True               # requested; only used if the device supports it
    device: str = None             # None -> auto; else 'cuda'|'dml'|'mps'|'cpu'
    seed: int = 0
    wandb_project: str = "stylegan2-ada-brats"


def _requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad_(flag)


def _ema_update(g_ema, g, decay):
    with torch.no_grad():
        for pe, p in zip(g_ema.parameters(), g.parameters()):
            pe.copy_(p.lerp(pe, decay))
        for be, b in zip(g_ema.buffers(), g.buffers()):
            be.copy_(b)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    caps = probe(device)
    print(describe(caps))

    out_dir = Path(cfg.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- data ---
    if cfg.dataset == "retina":
        ds = RetinaPatchDataset(cfg.data, resolution=cfg.resolution, xflip=cfg.xflip,
                                color_channels=cfg.color_channels or 3)
    else:
        ds = SliceDataset(cfg.data, resolution=cfg.resolution, xflip=cfg.xflip)
    print(f"Dataset: {len(ds)} images (xflip={cfg.xflip}) @ {cfg.resolution}px, {ds.num_channels}ch")
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=True, drop_last=True,
                        num_workers=cfg.workers, pin_memory=(caps.kind == "cuda"))

    def infinite(dl):
        while True:
            for b in dl:
                yield b

    data_iter = infinite(loader)

    # --- models ---
    common = dict(img_resolution=cfg.resolution, img_channels=ds.num_channels,
                  channel_base=cfg.channel_base, channel_max=cfg.channel_max)
    G = Generator(z_dim=cfg.z_dim, w_dim=cfg.w_dim, mapping_layers=cfg.mapping_layers, **common).to(device)
    D = Discriminator(**common).to(device)
    G_ema = copy.deepcopy(G).eval()
    _requires_grad(G_ema, False)

    g_opt = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.0, 0.99), eps=1e-8)
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.0, 0.99), eps=1e-8)

    # --- augmentation / ADA ---
    aug = AugmentPipe(p=cfg.p_init, allow_geometric=caps.geometric_aug,
                      color_channels=cfg.color_channels)

    # --- regularization strengths ---
    r1_gamma = cfg.r1_gamma if cfg.r1_gamma is not None else 0.0002 * (cfg.resolution ** 2) / cfg.batch
    do_r1 = caps.r1 and r1_gamma > 0
    if caps.r1 is False and r1_gamma > 0:
        print("R1 regularization disabled on this device (no double-backward).")
    # cuDNN/MIOpen for the R1 double-backward. On ROCm+FAST find mode MIOpen picks the
    # broken im2col solver and crashes at >=256px, so route R1 through native conv. But
    # with MIOPEN_FIND_MODE=NORMAL, MIOpen searches and finds a *working* fast solver for
    # the double-backward (and normal steps get ~10x faster too), so keep R1 on MIOpen.
    _find_normal = os.environ.get("MIOPEN_FIND_MODE", "").upper() == "NORMAL"
    r1_cudnn = cfg.r1_cudnn if cfg.r1_cudnn is not None else (torch.version.hip is None or _find_normal)
    if do_r1 and not r1_cudnn:
        print("R1 double-backward routed through native conv (cuDNN/MIOpen off for R1 only).")

    use_amp = cfg.amp and caps.amp
    amp_device_type = "cuda" if caps.kind == "cuda" else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"AMP (this run): {'on' if use_amp else 'off (fp32)'}")

    # --- optional wandb ---
    run = None
    try:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", cfg.wandb_project),
                         name=Path(cfg.out).name, dir=cfg.out, resume="allow",
                         config=asdict(cfg))
        print("Weights & Biases logging enabled.")
    except ImportError:
        print("wandb not installed -> skipping W&B logging.")
    except Exception as err:  # not logged in, offline, etc.
        print("wandb init failed -> skipping W&B logging:", err)

    # fixed noise for consistent sample grids
    grid_n = 16
    grid_z = torch.randn(grid_n, cfg.z_dim, device=device)

    total_imgs = cfg.kimg * 1000
    ema_beta = 0.5 ** (cfg.batch / max(1.0, cfg.ema_kimg * 1000))
    cur_img = 0
    step = 0
    ada_stat = 0.0  # running mean of sign(D(real))
    t0 = time.time()
    last_t = t0
    last_img = 0

    while cur_img < total_imgs:
        real, _ = next(data_iter)
        real = real.to(device)

        # =========================== D step ===========================
        _requires_grad(G, False)
        _requires_grad(D, True)
        d_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device_type, enabled=use_amp):
            z = torch.randn(cfg.batch, cfg.z_dim, device=device)
            fake = G(z)
            d_fake = D(aug(fake.detach()))
            real_aug = aug(real)
            d_real = D(real_aug)
            loss_d = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
        scaler.scale(loss_d).backward()

        # lazy R1 on real images. Scaled with the same GradScaler as loss_d so both
        # backward passes accumulate consistently-scaled grads before scaler.step().
        loss_r1_val = 0.0
        if do_r1 and (step % cfg.r1_interval == 0):
            # The whole block (forward, grad, and the second-order backward) runs under
            # the cuDNN flag: on ROCm r1_cudnn=False forces native ATen conv so the conv
            # double-backward avoids MIOpen's uncompilable im2col kernel on gfx1101.
            with torch.backends.cudnn.flags(enabled=r1_cudnn):
                real_r1 = real.detach().requires_grad_(True)
                d_real_r1 = D(aug(real_r1))
                grad = torch.autograd.grad(outputs=d_real_r1.sum(), inputs=real_r1,
                                           create_graph=True, only_inputs=True)[0]
                r1_pen = grad.pow(2).sum(dim=[1, 2, 3]).mean()
                scaler.scale(r1_gamma * 0.5 * r1_pen * cfg.r1_interval).backward()
            loss_r1_val = float(r1_pen.detach())

        scaler.step(d_opt)
        scaler.update()

        # ADA controller: nudge p toward keeping sign(D(real)) at ada_target
        with torch.no_grad():
            ada_stat = float(torch.sign(d_real).mean())
        if step % cfg.ada_interval == 0:
            adjust = np.sign(ada_stat - cfg.ada_target) * (cfg.batch * cfg.ada_interval) / (cfg.ada_kimg * 1000)
            aug.p = float(np.clip(aug.p + adjust, 0.0, 1.0))

        # =========================== G step ===========================
        _requires_grad(G, True)
        _requires_grad(D, False)
        g_opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device_type, enabled=use_amp):
            z = torch.randn(cfg.batch, cfg.z_dim, device=device)
            fake = G(z, update_emas=True)
            g_logits = D(aug(fake))
            loss_g = F.softplus(-g_logits).mean()
        scaler.scale(loss_g).backward()
        scaler.step(g_opt)
        scaler.update()

        # EMA
        _ema_update(G_ema, G, ema_beta)

        cur_img += cfg.batch
        step += 1

        # =========================== logging ===========================
        if step % 10 == 0:
            now = time.time()
            sec_per_kimg = (now - last_t) / max(1e-6, (cur_img - last_img) / 1000.0)
            last_t, last_img = now, cur_img
            ld, lg = float(loss_d.detach()), float(loss_g.detach())
            msg = (f"kimg {cur_img/1000:7.1f}/{cfg.kimg}  "
                   f"loss_D {ld:.3f}  loss_G {lg:.3f}  "
                   f"p {aug.p:.3f}  r1 {loss_r1_val:.3f}  {sec_per_kimg:.1f} s/kimg")
            print(msg)
            if run is not None:
                run.log({
                    "loss/D": ld, "loss/G": lg,
                    "aug/p": aug.p, "reg/r1": loss_r1_val,
                    "perf/sec_per_kimg": sec_per_kimg, "ada/sign_real": ada_stat,
                }, step=int(cur_img / 1000))

        # image samples
        if (cur_img // 1000) > 0 and (cur_img % (cfg.sample_every_kimg * 1000)) < cfg.batch:
            _save_samples(G_ema, grid_z, out_dir, cur_img, run)

        # FID
        if cfg.fid_every_kimg and (cur_img % (cfg.fid_every_kimg * 1000)) < cfg.batch:
            _maybe_fid(G_ema, ds, cfg, device, cur_img, run)

        # checkpoints
        if (cur_img // 1000) > 0 and (cur_img % (cfg.ckpt_every_kimg * 1000)) < cfg.batch:
            _save_ckpt(G_ema, cfg, out_dir / "checkpoints" / f"kimg{cur_img//1000:05d}.pt")

    # --- final artifacts ---
    _save_samples(G_ema, grid_z, out_dir, cur_img, run)
    _save_ckpt(G_ema, cfg, out_dir / "checkpoints" / "final.pt")
    _maybe_fid(G_ema, ds, cfg, device, cur_img, run, force=True)
    if run is not None:
        run.finish()
    print(f"Training done in {(time.time()-t0)/60:.1f} min. "
          f"Checkpoints in {out_dir/'checkpoints'}")
    return G_ema


def _save_ckpt(G_ema, cfg, path):
    torch.save({"G_ema": G_ema.state_dict(), "config": asdict(cfg),
                "img_channels": G_ema.img_channels}, path)


@torch.no_grad()
def _save_samples(G_ema, grid_z, out_dir, cur_img, run):
    from torchvision.utils import save_image
    G_ema.eval()
    imgs = G_ema(grid_z, truncation_psi=0.7, noise_mode="const")  # (N,C,H,W) in ~[-1,1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    imgs = imgs[:, :3]  # show only the image channels (RGB for retina, the 1ch for brats)
    path = out_dir / "samples" / f"fakes_kimg{cur_img//1000:05d}.png"
    save_image(imgs, str(path), nrow=4)
    if run is not None:
        import wandb
        run.log({"samples": wandb.Image(str(path))}, step=int(cur_img / 1000))


def _maybe_fid(G_ema, ds, cfg, device, cur_img, run, force=False):
    try:
        from .metrics.fid import compute_fid
        fid = compute_fid(G_ema, ds, cfg, device, num=cfg.fid_num)
        print(f"  FID @ {cur_img//1000} kimg: {fid:.2f}")
        if run is not None:
            run.log({"Metrics/fid": fid}, step=int(cur_img / 1000))
        return fid
    except Exception as err:
        if force:
            print("FID skipped:", err)
        return None
