"""BraTS 2020 MAD study — Run A (synthetic-only recursion), multi-class (WT/TC/ET).

Reuses the VALIDATED milestone pipeline (channel-generic paired StyleGAN2-ADA + the configurable
U-Net probe from tstr/unet.py). Each generation:
  1. train a 7-channel GAN (4 modalities + WT/TC/ET masks)   [gen1 = REAL, genN = gen(N-1) fakes]
  2. generate N_FAKES synthetic (4-mod + 3-mask) volumes
  3. train a FRESH U-Net on those fakes -> score on the LOCKED real test set
  4. record per-class Dice (WT/TC/ET) + ET recall (rare) + FID
Writes one row/gen to results/brats_mad/metrics.csv. ET (enhancing tumor, ~17% of WT, hardest region)
is the rare class we expect to collapse first.

PREREQUISITE: run tstr/preprocess_brats_mad.py first (builds data_brats_mad/{train,test}).
Run in a terminal or via a background launcher notebook (long GAN trainings).
"""
import os, sys, glob, csv, time, argparse
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
import numpy as np, torch
from stylegan2_ada.train import Config, train
from stylegan2_ada.models.networks import Generator
from tstr.unet import train_probe

DATA       = r"C:\Users\Tonkid\Downloads\tstr\data_brats_mad"
REAL_TRAIN = os.path.join(DATA, "train")
TEST       = os.path.join(DATA, "test")
RUNS       = r"C:\Users\Tonkid\Downloads\runs\brats_mad"
RESULTS    = r"C:\Users\Tonkid\Downloads\results\brats_mad"
CSV        = os.path.join(RESULTS, "metrics.csv")
os.makedirs(RUNS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

CLASSES = ["WT", "TC", "ET"]            # 3 mask channels (out_ch=3); ET = rare
IN_CH   = 4                            # 4 modalities in
DEVICE  = "cuda"

# ---- knobs (overridable via argparse) ----
KIMG    = 600
N_FAKES = 5000
BATCH   = 16
N_GEN   = 6
PROBE_EPOCHS = 30


def train_generator(data_dirs, out_dir, kimg):
    latest = os.path.join(out_dir, "checkpoints", "latest.pt")     # crash-resume point
    resume = latest if os.path.exists(latest) else None
    if resume:
        print(f"  found {latest} -> RESUMING this generation", flush=True)
    cfg = Config(
        data=data_dirs, out=out_dir, paired=True, resolution=256, batch=BATCH, kimg=kimg,
        lr=0.0015, r1_gamma=10.0, p_init=0.5, ada_kimg=100, ada_target=0.6, ema_kimg=10.0,
        sample_every_kimg=20, ckpt_every_kimg=100, fid_every_kimg=0,
        amp=True, r1_cudnn=None, workers=0, seed=0, resume=resume,
    )
    train(cfg)
    return os.path.join(out_dir, "checkpoints", "final.pt")


@torch.no_grad()
def generate_fakes(ckpt_path, out_dir, n=N_FAKES, batch=BATCH, seed=0):
    """(4 modalities [0,1]) + (3 masks {0,1}) saved as (7,H,W) -- the U-Net probe's contract."""
    torch.manual_seed(seed)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); c = ck["config"]
    ch = int(ck.get("img_channels", 7))
    G = Generator(z_dim=c["z_dim"], w_dim=c["w_dim"], img_resolution=c["resolution"],
                  img_channels=ch, mapping_layers=c["mapping_layers"],
                  channel_base=c["channel_base"], channel_max=c["channel_max"])
    G.load_state_dict(ck["G_ema"]); G.eval().cuda()
    os.makedirs(out_dir, exist_ok=True)
    print(f"  generating {n} fakes ({ch}ch = {IN_CH} mod + {ch-IN_CH} masks) -> {out_dir}", flush=True)
    i = 0
    while i < n:
        b = min(batch, n - i)
        z = torch.randn(b, c["z_dim"], device="cuda")
        out = G(z, truncation_psi=1.0, noise_mode="random")             # (b,7,H,W) ~[-1,1]
        img = ((out[:, :IN_CH].clamp(-1, 1) + 1) / 2).cpu().numpy()      # 4 modalities [0,1]
        msk = (out[:, IN_CH:] > 0.0).float().cpu().numpy()              # 3 masks {0,1}
        for k in range(b):
            np.save(os.path.join(out_dir, f"{i:05d}.npy"),
                    np.concatenate([img[k], msk[k]], 0).astype(np.float32))
            i += 1
    print(f"  done: {i} fakes", flush=True)
    return out_dir


def gen_dir(gen):   return os.path.join(RUNS, f"gen{gen}")
def fakes_dir(gen): return os.path.join(gen_dir(gen), "fakes")

FIELDS = ["gen", "kimg", "dice_mean", "dice_WT", "dice_TC", "dice_ET",
          "recall_WT", "recall_TC", "recall_ET"]

def _append(row):
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_gen(gen, kimg):
    t0 = time.time(); out = gen_dir(gen); os.makedirs(out, exist_ok=True)
    data = REAL_TRAIN if gen == 1 else fakes_dir(gen - 1)           # Run A recursion
    ck = os.path.join(out, "checkpoints", "final.pt")
    if not os.path.exists(ck):
        ck = train_generator(data, out, kimg)
    fdir = fakes_dir(gen)
    if len(glob.glob(os.path.join(fdir, "*.npy"))) < N_FAKES:
        generate_fakes(ck, fdir, n=N_FAKES, seed=1000 + gen)
    r = train_probe([fdir], TEST, out_dir=os.path.join(out, "probe"),
                    in_ch=IN_CH, out_ch=len(CLASSES), class_names=CLASSES,
                    epochs=PROBE_EPOCHS, batch=BATCH, res=256, log_every=10)
    row = {"gen": gen, "kimg": kimg, "dice_mean": round(r["dice"], 4)}
    for c in CLASSES:
        row[f"dice_{c}"]   = round(r["per_class_dice"][c], 4)
        row[f"recall_{c}"] = round(r["per_class_recall"][c], 4)
    _append(row)
    print(f"[GEN {gen} DONE in {(time.time()-t0)/60:.0f} min] {row}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimg", type=int, default=KIMG)
    ap.add_argument("--gens", type=int, default=N_GEN)
    ap.add_argument("--start", type=int, default=1)
    a = ap.parse_args()
    print(f"=== BraTS MAD Run A — gens {a.start}..{a.gens}, KIMG={a.kimg}, N_FAKES={N_FAKES} ===", flush=True)
    assert os.path.isdir(REAL_TRAIN), "run preprocess_brats_mad.py first"
    for gen in range(a.start, a.gens + 1):
        run_gen(gen, a.kimg)
    print("=== BraTS MAD Run A DONE ->", CSV, "===", flush=True)


if __name__ == "__main__":
    main()
