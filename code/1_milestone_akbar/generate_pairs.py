"""Generate N synthetic (modalities..., mask) pairs from a paired GAN's final.pt.
Channel count is read from the checkpoint, so this handles BOTH the 2-channel
(T1Gd+mask) and the 5-channel (t1,t1ce,t2,flair+mask) models. Saves each as a
(C,H,W) .npy: first C-1 channels = image modalities [0,1], LAST channel = binary
tumor mask {0,1} -- the exact contract the U-Net probe (tstr/unet.py) reads
(in_ch=C-1, out_ch=1)."""
import os, sys, argparse, numpy as np, torch
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
from stylegan2_ada.models.networks import Generator


def _img_channels(ck):
    """Infer generator output channels from the checkpoint (top-level key, config, or weights)."""
    if "img_channels" in ck:
        return int(ck["img_channels"])
    c = ck["config"]
    if "img_channels" in c:
        return int(c["img_channels"])
    # fall back to the last torgb conv's out_channels in the EMA weights
    for k, v in reversed(list(ck["G_ema"].items())):
        if k.endswith(".weight") and v.dim() == 4:
            return int(v.shape[0])
    raise ValueError("could not infer img_channels from checkpoint")


def generate(ckpt_path, out_dir, n=5000, batch=16, seed=0):
    torch.manual_seed(seed)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    c = ck["config"]
    ch = _img_channels(ck)
    G = Generator(z_dim=c["z_dim"], w_dim=c["w_dim"], img_resolution=c["resolution"],
                  img_channels=ch, mapping_layers=c["mapping_layers"],
                  channel_base=c["channel_base"], channel_max=c["channel_max"])
    G.load_state_dict(ck["G_ema"]); G.eval().cuda()
    os.makedirs(out_dir, exist_ok=True)
    print(f"generating {n} pairs @ {c['resolution']}px, {ch} channels ({ch-1} img + mask) -> {out_dir}")
    i = 0
    with torch.no_grad():
        while i < n:
            z = torch.randn(min(batch, n - i), c["z_dim"], device="cuda")
            out = G(z, truncation_psi=1.0, noise_mode="random")            # (B,C,H,W) ~[-1,1]
            img = ((out[:, :-1].clamp(-1, 1) + 1) / 2).cpu().numpy()        # modalities [0,1]
            msk = (out[:, -1:] > 0.0).float().cpu().numpy()                 # mask {0,1}
            for k in range(img.shape[0]):
                np.save(os.path.join(out_dir, f"{i:05d}.npy"),
                        np.concatenate([img[k], msk[k]], 0).astype(np.float32))  # (C,H,W)
                i += 1
            if i % 1000 == 0:
                print("  ", i)
    # quick sanity: fraction of fakes that contain a tumor (mask = last channel)
    tum = sum(int(np.load(os.path.join(out_dir, f))[-1].sum() > 0)
              for f in os.listdir(out_dir)[:500] if f.endswith(".npy"))
    print(f"done: {i} pairs. tumor-present in ~{tum/5:.0f}% of first 500")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\Users\Tonkid\Downloads\runs\paired256_full\checkpoints\final.pt")
    ap.add_argument("--out", default=r"C:\Users\Tonkid\Downloads\tstr\data\gen1_fakes")
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    generate(a.ckpt, a.out, a.n)
