"""Load a trained G_ema checkpoint and sample / interpolate from it."""

import torch

from .models.networks import Generator


def load_generator(ckpt_path, device):
    """Return (G_ema in eval mode on `device`, config dict) from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    G = Generator(
        z_dim=cfg["z_dim"], w_dim=cfg["w_dim"],
        img_resolution=cfg["resolution"], img_channels=1,
        mapping_layers=cfg["mapping_layers"],
        channel_base=cfg["channel_base"], channel_max=cfg["channel_max"],
    )
    G.load_state_dict(ckpt["G_ema"])
    G = G.to(device).eval()
    for p in G.parameters():
        p.requires_grad_(False)
    return G, cfg


@torch.no_grad()
def sample(G, cfg, device, num=16, psi=0.7, seed=None):
    """Return (num, 1, H, W) images in [0, 1]."""
    if seed is not None:
        torch.manual_seed(seed)
    z = torch.randn(num, cfg["z_dim"], device=device)
    imgs = G(z, truncation_psi=psi, noise_mode="const")
    return (imgs.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def interpolate(G, cfg, device, steps=8, psi=0.7, seed=None):
    """Linear interpolation in W-space between two random points. (steps, 1, H, W) in [0,1]."""
    if seed is not None:
        torch.manual_seed(seed)
    z0 = torch.randn(1, cfg["z_dim"], device=device)
    z1 = torch.randn(1, cfg["z_dim"], device=device)
    w0 = G.mapping(z0, truncation_psi=psi)
    w1 = G.mapping(z1, truncation_psi=psi)
    ts = torch.linspace(0, 1, steps, device=device).view(steps, 1)
    ws = w0 * (1 - ts) + w1 * ts               # (steps, w_dim)
    imgs = G.synthesis(ws, noise_mode="const")
    return (imgs.clamp(-1, 1) + 1) / 2
