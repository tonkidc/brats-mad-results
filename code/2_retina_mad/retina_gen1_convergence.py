"""GEN-1 CONVERGENCE RUN — find the kimg where gen-1 is trustworthy before locking it for the study.

Trains ONE generator on the REAL patches to KIMG_MAX, checkpointing every CKPT_EVERY kimg.
At each checkpoint it measures, vs the real data:
  * FID and KID                        (overall realism)
  * SE-present fraction & mean SE area (does the GAN actually make the RARE soft-exudate masks?)
  * saves an SE-mask montage PNG       (so you can EYEBALL that SE masks are real, not mush)

Why: your finding rests on gen-1 having genuine SE masks. If SE never forms because the GAN is
undertrained, "SE collapsed through recursion" is indistinguishable from "SE never existed" — a
worthless plot. So pick the kimg where the KID curve flattens AND the SE montages look real, then
lock THAT kimg for gens 2..4 (edit KIMG in retina_pilot_runA.py).

Reuses the validated pilot functions (import). Writes to runs_pilot/gen1_convergence/ +
results_pilot/gen1_convergence.csv + se_montage_kXXXX.png. Does NOT touch the main study.
Launched in the background by retina_convergence.ipynb.
"""
import os, sys, glob, csv, shutil, importlib.util
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")

# import the validated pilot module (functions + config), without running its main()
_spec = importlib.util.spec_from_file_location("pilot", r"C:\Users\Tonkid\Downloads\retina_autophagy\retina_pilot_runA.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)
from stylegan2_ada_retina.train import Config, train

KIMG_MAX   = 600
CKPT_EVERY = 100                                  # -> checkpoints at 100,200,...,600
N_EVAL     = 500                                  # fakes generated per checkpoint for KID/SE
SE_CH      = 3 + P.RARE_IDX                        # channel index of the SE mask in a (7,H,W) fake
OUT = os.path.join(P.RUNS, "gen1_convergence")
CSV = os.path.join(P.RESULTS, "gen1_convergence.csv")
os.makedirs(OUT, exist_ok=True); os.makedirs(P.RESULTS, exist_ok=True)


def se_stats_and_montage(fake_dir, kimg):
    """Fraction of fakes with any SE, mean SE area (px), and save an SE montage PNG."""
    files = sorted(glob.glob(os.path.join(fake_dir, "*.npy")))
    present = 0; areas = []; examples = []
    for f in files:
        a = np.load(f); se = a[SE_CH] > 0
        s = int(se.sum())
        if s > 0:
            present += 1; areas.append(s)
            if len(examples) < 8: examples.append((a, s))
    frac = present / max(1, len(files))
    mean_area = float(np.mean(areas)) if areas else 0.0
    # montage: RGB with SE mask overlaid, for up to 8 SE-bearing fakes
    n = max(1, len(examples))
    fig, ax = plt.subplots(2, 8, figsize=(20, 5.2))
    for j in range(8):
        for r in range(2): ax[r, j].axis("off")
        if j < len(examples):
            a, s = examples[j]
            rgb = ((a[:3].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
            ax[0, j].imshow(rgb); ax[0, j].set_title(f"fake {j}", fontsize=9)
            ax[1, j].imshow(rgb); ax[1, j].imshow(a[SE_CH] > 0, alpha=0.5, cmap="autumn")
            ax[1, j].set_title(f"SE ({s}px)", fontsize=9)
    plt.suptitle(f"gen-1 @ {kimg} kimg — SE-bearing fakes: {present}/{len(files)} ({frac*100:.0f}%) | RGB (top) / SE overlay (bottom)")
    plt.tight_layout(); plt.savefig(os.path.join(P.RESULTS, f"se_montage_k{kimg:05d}.png"), dpi=80)
    plt.close(fig)
    return frac, mean_area, present


def main():
    print(f"=== GEN-1 CONVERGENCE — train to {KIMG_MAX} kimg, checkpoint every {CKPT_EVERY} ===", flush=True)
    P.prep()   # sets splits + leakage assert (patches already built -> skips)

    # one long gen-1 training run on REAL patches, dense checkpoints
    cfg = Config(
        data=P.REAL_TRAIN, out=OUT, dataset="retina", color_channels=3,
        resolution=P.RES, batch=P.BATCH, kimg=KIMG_MAX,
        lr=0.0015, r1_gamma=10.0, p_init=0.5, ada_kimg=100, ada_target=0.6,
        ema_kimg=10.0, sample_every_kimg=25, ckpt_every_kimg=CKPT_EVERY,
        fid_every_kimg=0, amp=True, r1_cudnn=None, workers=0, seed=P.SEED,
        wandb_project="retina-autophagy",
    )
    train(cfg)
    print("training done; analysing checkpoints...", flush=True)

    fields = ["kimg", "fid", "kid", "se_present_frac", "se_mean_area_px", "se_present_count"]
    rows = []
    for k in range(CKPT_EVERY, KIMG_MAX + 1, CKPT_EVERY):
        ck = os.path.join(OUT, "checkpoints", f"kimg{k:05d}.pt")
        if not os.path.exists(ck):
            print("  missing checkpoint", ck, flush=True); continue
        tmp = os.path.join(OUT, f"_fakes_k{k}")
        P.generate_fakes(ck, tmp, n=N_EVAL, seed=k)
        fid, kid = P.fid_kid(tmp)
        frac, area, cnt = se_stats_and_montage(tmp, k)
        row = {"kimg": k, "fid": round(fid, 3), "kid": round(kid, 5),
               "se_present_frac": round(frac, 4), "se_mean_area_px": round(area, 1),
               "se_present_count": cnt}
        rows.append(row)
        print(f"  [k={k}] FID {fid:.2f} | KID {kid:.5f} | SE in {frac*100:.0f}% of fakes (mean {area:.0f}px)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)     # free disk

    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); [w.writerow(r) for r in rows]
    print(f"=== CONVERGENCE DONE -> {CSV} + se_montage_k*.png ===", flush=True)
    print("Pick the kimg where KID flattens AND SE montages look real, then set that as KIMG in", flush=True)
    print("retina_pilot_runA.py before running gens 2..4.", flush=True)


if __name__ == "__main__":
    main()
