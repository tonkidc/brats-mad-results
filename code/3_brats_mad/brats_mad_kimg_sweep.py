"""Gen-1 kimg sweep — find the SHORTEST GAN training length that still gives good fake->real Dice.

The gen-1 GAN already saved a checkpoint every 100 kimg. This reuses those (NO GAN retraining) and,
for each checkpoint, does the honest test: generate fakes -> train a fresh U-Net -> score the LOCKED
real test set. If a low-kimg checkpoint (e.g. 300) matches kimg 600, we can run every generation in
the recursion at that shorter length and ~halve the whole 6-gen run.

Reuses generate_fakes + the probe from brats_mad_runA (identical settings -> apples-to-apples with the
0.733 gen-1 gate). Deletes each checkpoint's fakes right after probing (they are ~2 MB each, 9+ GB per
checkpoint). Writes results/brats_mad/kimg_sweep.csv and prints a recommended knee.

Run:  python tstr/brats_mad_kimg_sweep.py                 # 5000 fakes, 30 epochs (matches baseline)
      python tstr/brats_mad_kimg_sweep.py --fakes 3000 --epochs 20   # faster, still fair
"""
import os, sys, glob, csv, time, shutil, argparse
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")

from tstr.brats_mad_runA import (generate_fakes, IN_CH, CLASSES, TEST, RUNS, RESULTS)
from tstr.unet import train_probe

CKPT_DIR = os.path.join(RUNS, "gen1", "checkpoints")
SWEEP    = os.path.join(RUNS, "gen1", "sweep_fakes")     # temp, deleted per-checkpoint
CSV      = os.path.join(RESULTS, "kimg_sweep.csv")
FIELDS   = ["kimg", "dice_mean", "dice_WT", "dice_TC", "dice_ET",
            "recall_WT", "recall_TC", "recall_ET"]


def _kimg_of(path):
    b = os.path.basename(path)                            # kimg00300.pt
    return int(b.replace("kimg", "").replace(".pt", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fakes", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "kimg*.pt")), key=_kimg_of)
    assert ckpts, f"no kimg*.pt checkpoints in {CKPT_DIR} (run gen-1 first)"
    print(f"=== kimg sweep over {len(ckpts)} checkpoints: {[_kimg_of(c) for c in ckpts]} ===")
    print(f"    {a.fakes} fakes, {a.epochs} epochs each | fakes deleted after each probe\n", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    with open(CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    rows = []
    for ck in ckpts:
        k = _kimg_of(ck); t0 = time.time()
        print(f"--- kimg {k} ---", flush=True)
        if os.path.exists(SWEEP): shutil.rmtree(SWEEP, ignore_errors=True)
        generate_fakes(ck, SWEEP, n=a.fakes, seed=7000 + k)
        r = train_probe([SWEEP], TEST, out_dir=os.path.join(RUNS, "gen1", f"probe_k{k:05d}"),
                        in_ch=IN_CH, out_ch=len(CLASSES), class_names=CLASSES,
                        epochs=a.epochs, batch=16, res=256, log_every=10)
        row = {"kimg": k, "dice_mean": round(r["dice"], 4)}
        for c in CLASSES:
            row[f"dice_{c}"]   = round(r["per_class_dice"][c], 4)
            row[f"recall_{c}"] = round(r["per_class_recall"][c], 4)
        with open(CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        rows.append(row)
        print(f"    kimg {k}: mean {row['dice_mean']}  WT {row['dice_WT']}  TC {row['dice_TC']}  "
              f"ET {row['dice_ET']}   ({(time.time()-t0)/60:.0f} min)\n", flush=True)
    shutil.rmtree(SWEEP, ignore_errors=True)

    # ---- summary + recommended knee ----
    best = max(rows, key=lambda r: r["dice_mean"])
    thresh = best["dice_mean"] - 0.02                     # within 0.02 of the best mean Dice
    knee = min((r for r in rows if r["dice_mean"] >= thresh), key=lambda r: r["kimg"])
    print("=== SWEEP DONE ===")
    print("kimg | mean  |  WT   |  TC   |  ET")
    for r in rows:
        mark = "  <- best" if r is best else ("  <- knee (shortest within 0.02 of best)" if r is knee else "")
        print(f"{r['kimg']:>4} | {r['dice_mean']:.3f} | {r['dice_WT']:.3f} | {r['dice_TC']:.3f} | {r['dice_ET']:.3f}{mark}")
    print(f"\nRecommendation: run the recursion at kimg={knee['kimg']} "
          f"(mean Dice {knee['dice_mean']:.3f} vs best {best['dice_mean']:.3f} at kimg {best['kimg']}).")
    print(f"CSV -> {CSV}")


if __name__ == "__main__":
    main()
