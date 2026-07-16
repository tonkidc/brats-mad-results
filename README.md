# StyleGAN2-ADA — Results Portfolio

Three linked studies on paired (image + segmentation-mask) medical-image generation with a custom
pure-PyTorch StyleGAN2-ADA, all evaluated the honest way: **train a U-Net on synthetic data, score it
on a LOCKED real test set** (fake→real Dice, a.k.a. TSTR — Train Synthetic, Test Real).

| # | Study | Question | Headline result |
|---|-------|----------|-----------------|
| 1 | **Milestone — Akbar replication (BraTS)** | Can our paired GAN match a published fake→real Dice? | **0.72** ≈ Akbar et al.'s **0.737** ✅ |
| 2 | **Retina MAD (IDRiD)** | Does recursive self-training collapse rare lesions first? | **Negative result** — dataset (44 imgs) too small; fakes never realistic enough (gate 0.16×) |
| 3 | **BraTS MAD (multi-class)** | Same MAD question, on a validated large dataset | **In progress** — early signal: rare class (ET) is *robust*, not first to collapse |

> **Code** for all three studies is in [`code/`](code/) (engine packages + per-study scripts/notebooks).
> Results/plots/configs live in the three numbered folders below. No patient data or checkpoints are included.

---

## 1 · Milestone — Akbar replication  (`1_milestone_akbar_brats/`)

**Goal:** prove the paired GAN is real by reproducing a known benchmark. Trained a 5-channel
(4 MR modalities T1/T1ce/T2/FLAIR + whole-tumor mask) StyleGAN2-ADA on BraTS 2020, generated synthetic
pairs, trained a U-Net only on fakes, scored on held-out real patients.

- **fake→real Dice ≈ 0.72** vs Akbar et al. 2024's **0.737** (their StyleGAN2 setup). Match.
- Confirms the whole pipeline (paired generation + channel-generic U-Net probe + locked-test scoring)
  is sound. **This is the foundation the two MAD studies below reuse.**
- Files: `RESULTS_notes.md` (parameter→outcome log, 0.19→0.30→0.72), `StyleGAN2ADA_BraTS_report.md`,
  `results_table.html`, `milestone_config.json`.

## 2 · Retina MAD — IDRiD  (`2_retina_mad_negative/`)

**Goal:** first attempt at the Model Autophagy Disorder (MAD) study — recursively train a GAN on its own
output for N generations and watch which lesion class degrades first. Rare class = **Soft Exudates (SE)**.

- **Clean negative result.** IDRiD has only 44 training images (SE in just 23), so the GAN learned a
  blobby caricature of lesions. A U-Net trained on gen-1 fakes scored **mean Dice 0.068 vs 0.43 real
  topline → gate 0.16×, hard fail.** 3 of 4 classes transferred at ~zero; only the big/bright class (EX)
  partly survived.
- **Root cause = data scarcity, not a bug** (real topline works; the pipeline is the validated one from
  study 1). Conclusion: 44 images can't teach realistic fine-lesion morphology → not a viable dataset
  for the recursion study. **This is what motivated the pivot to BraTS.**
- Files: `CONVERGENCE_results.md` (full write-up), `convergence_plot.png`, `all_lesions_montage.png`
  (blobby fakes), `REAL_se_examples.png` (what real SE looks like), `se_montage_k00500.png`.

## 3 · BraTS MAD — multi-class  (`3_brats_mad_multiclass/`)

**Goal:** the MAD study done right — BraTS 2020 (369 subjects, validated pipeline from study 1).
Three tumor sub-regions of increasing difficulty: **WT** (whole tumor, common) · **TC** (tumor core) ·
**ET** (enhancing tumor, ~17% of WT — the rare/hard "canary"). Run A = pure autophagy (each generation
trains only on the previous generation's fakes). All generations trained at 300 kimg for consistency.

- **Gen-1 gate PASSED big:** fake→real Dice **WT .68 / TC .77 / ET .75 (mean .73)** at 600 kimg — ET
  (rare) transfers at **0.75**, vs retina's rare class at ~0.00. Night and day; BraTS was the right call.
- **Recursion in progress** (see `metrics.csv` + `brats_mad_curve.png`). Early trend across the first
  generations: overall Dice drifts down (with the usual non-monotonic wobble), **but the rare class ET
  stays the *strongest* of the three and WT (the "easy" class) stays weakest** — the opposite of the
  "rare-class-collapses-first" hypothesis. Consistent so far; needs the full 6 generations to conclude.
- Files: `metrics.csv` (live, one row/generation), `metrics_gen1_kimg600_backup.csv` (the 600-kimg gate),
  `gen1_probe_kimg600.json`, `brats_mad_curve.png`, and `samples_kimg300/gen{1..4}_kimg300.png`
  (7-channel synthetic samples, one per generation — eyeball quality across the recursion).

---

### How to read the numbers everywhere
All Dice/recall are **fake→real on a locked real test set** — the generator never sees the test data,
and the U-Net never trains on real data (except the study-1 real topline). It's the strict,
leakage-free way to measure whether synthetic data is actually useful.

_Generated 2026-07-16. Studies 1–2 complete; study 3 recursion ongoing._
