"""GEN-1 PROBE GATE — the decision step before committing to the recursion.

Reuses the converged gen-1 (k500 checkpoint). Generates 5,000 fakes, trains a fresh U-Net on them,
and scores on the LOCKED real test set -> per-class Dice (MA/HE/EX/SE) + SE recall. Also trains a
topline U-Net on the REAL patches for reference. Prints a side-by-side table + the Akbar-style gate
(gen1 mean-Dice >= 0.80 x real mean-Dice).

Fast (~1.5 h): no GAN training. Reuses the validated pilot functions. Writes results_pilot/gen1_probe.json.
Launched in the background by retina_gen1_probe.ipynb.
"""
import os, sys, glob, json, importlib.util
import numpy as np, torch
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
_spec = importlib.util.spec_from_file_location("pilot", r"C:\Users\Tonkid\Downloads\retina_autophagy\retina_pilot_runA.py")
P = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P)

CK    = r"C:\Users\Tonkid\Downloads\retina_autophagy\runs_pilot\gen1_convergence\checkpoints\kimg00500.pt"
FAKES = os.path.join(P.RUNS, "gen1_probe_fakes")
OUTJS = os.path.join(P.RESULTS, "gen1_probe.json")


def main():
    print("=== GEN-1 PROBE GATE (reusing k500) ===", flush=True)
    assert os.path.exists(CK), f"missing k500 checkpoint: {CK}"
    P.prep()   # sets splits + leakage assert (patches exist -> skips build)

    # 1) 5,000 fakes from the converged gen-1
    if len(glob.glob(os.path.join(FAKES, "*.npy"))) < P.N_FAKES:
        P.generate_fakes(CK, FAKES, n=P.N_FAKES, seed=500)
    else:
        print("fakes already present, skipping generation", flush=True)

    # 2) topline: U-Net on REAL patches
    print("\n-- topline: U-Net on REAL patches --", flush=True)
    net_real = P.train_unet(P.REAL_TRAIN); m_real = P.eval_on_locked_test(net_real)
    del net_real; torch.cuda.empty_cache()
    print("real :", m_real, flush=True)

    # 3) gen-1: U-Net on the FAKES
    print("\n-- gen-1: U-Net on k500 FAKES --", flush=True)
    net_g1 = P.train_unet(FAKES); m_g1 = P.eval_on_locked_test(net_g1)
    del net_g1; torch.cuda.empty_cache()
    print("gen1 :", m_g1, flush=True)

    rel = m_g1["dice_mean"] / max(1e-6, m_real["dice_mean"])
    result = {"real": m_real, "gen1": m_g1, "rel_dice": rel,
              "gate_pass": bool(rel >= 0.80)}
    json.dump(result, open(OUTJS, "w"), indent=2)

    # 4) readable table
    print("\n==================== GEN-1 PROBE RESULT ====================", flush=True)
    print(f"{'class':>6} | {'real Dice':>10} | {'gen1 Dice':>10}", flush=True)
    for c in P.LESIONS:
        print(f"{c:>6} | {m_real['dice_per_class'][c]:>10.3f} | {m_g1['dice_per_class'][c]:>10.3f}", flush=True)
    print(f"{'mean':>6} | {m_real['dice_mean']:>10.3f} | {m_g1['dice_mean']:>10.3f}", flush=True)
    print(f"\nSE recall   real {m_real['se_recall']:.3f} | gen1 {m_g1['se_recall']:.3f}", flush=True)
    print(f"gen1/real mean-Dice = {rel:.2f}   (Akbar band 0.80-0.90)", flush=True)
    print("GATE:", "PASS -> pipeline sound, proceed to recursion" if rel >= 0.80
          else "LOW -> gen-1 transfer weak; decide before scaling", flush=True)
    print(f"saved -> {OUTJS}", flush=True)


if __name__ == "__main__":
    main()
