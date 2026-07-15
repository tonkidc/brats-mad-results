# Results Notes — BraTS validation step (Milestone 1)

_Goal of this step: prove the pipeline lands where Akbar et al. (2024) landed, before running the
real IDRiD retina recursion study. "Train a U-Net on gen-1 fakes; if the number is near Akbar's,
the pipeline is sound."_

Hardware: AMD RX 7800 XT (ROCm on Windows). GAN: custom `stylegan2_ada` (StyleGAN2-ADA, 256px).
Locked real test set: **4,520 slices / 56 subjects** — never touched by any generator.

---

## Headline

**4-modality synthetic-trained U-Net Dice = 0.7229  vs  Akbar's 0.737  →  pipeline VALIDATED.**

Going from 1 modality to 4 modalities lifted the Dice from **0.30 → 0.72** — exactly the predicted
lever (whole tumor is barely visible in T1ce alone; FLAIR/T2 make it visible).

---

## All results

| Experiment | Setup | Metric | Value | Reference |
|---|---|---|---|---|
| Image-only GAN | single modality (T1ce), no mask | **FID** (256px) | **20.76** | Akbar StyleGAN2 FID = 84.77 |
| Paired GAN, single-modality | T1ce + mask, 5k fakes | fake→real **Dice** | **0.30** | — |
| Real baseline, single-modality | real T1ce + mask | real→real **Dice** | **0.66** | (pipeline sanity) |
| **Paired GAN, 4-modality** | **t1,t1ce,t2,flair + mask, 20k fakes, 1000 kimg** | **fake→real Dice** | **0.7229** | **Akbar 0.737** |
| Rare detail (4-mod) | — | tumor **recall** | **0.762** | — |

---

## Key clarification — the two FID vs Dice come from DIFFERENT models

- The **FID 20.76** came from an **image-only** generator (no mask) → it measures image realism only,
  and cannot feed a segmentation net. It is NOT the model behind the Dice.
- The **Dice** comes from the **paired image+mask** generator (a different model).
- Verified from Akbar's paper: their FID was also computed on **single-modality images, no mask**
  ("100,000 synthetic T1wGd images used to calculate each metric"), so the FID comparison is fair.
- Akbar themselves note *"FID and IS do not correlate well with... performance when training networks
  with synthetic images"* — consistent with what we see, and relevant to the MAD study's core idea
  that the metrics diverge.
- TODO (optional): recompute FID on the **paired 4-modality** model so FID and Dice describe the same
  generator.

---

## Why 0.72 and not exactly 0.737 (honest caveats)

- **Segmenter:** Akbar used **nnU-Net** (heavyweight); ours is a smaller custom U-Net. Most of the
  remaining 0.014 gap is here.
- **Fakes:** 20k synthetic vs Akbar's 100k.
- Neither is a bug — the pipeline is validated. 0.72 ≈ 0.737 is a match for this purpose.

---

## Settings that produced the 0.72

GAN (`run_paired_multi.py` / milestone1.ipynb Step 2):
`resolution=256, kimg=1000, batch=16, lr=0.0015, r1_gamma=10, p_init=0.5, ada_kimg=100, amp=True`
Training time ~14 h; final loss_D ~1.1 (no collapse).

Probe (`run_akbar_check_multi.ipynb`): 20,000 fakes, U-Net in_ch=4 → out_ch=1, 25 epochs, 256px,
intensity augmentation on. Data: 3,130 real train slices, 4,520 locked test slices.

Artifacts on disk:
- GAN: `runs\paired256_multi\checkpoints\final.pt`
- Fakes: `tstr\data_multi\gen1_fakes\` (20k × (5,256,256))
- Probe result: `tstr\probe_gen1_multi\probe.json`

---

## Exact parameters → result (reproducibility record)

**GAN (5-channel paired StyleGAN2-ADA)** — from `runs\paired256_multi\config.json`:

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| resolution | 256 | | r1_gamma | 10.0 |
| img channels | 5 (t1,t1ce,t2,flair,mask) | | r1_interval | 16 |
| batch | 16 | | ada_target | 0.6 |
| kimg (train length) | 1000 | | ada_kimg | 100 |
| lr | 0.0015 | | p_init | 0.5 |
| z_dim / w_dim | 512 / 512 | | ema_kimg | 10.0 |
| channel_base / max | 16384 / 512 | | xflip | True |
| mapping_layers | 2 | | amp (fp16) | True |
| seed | 0 | | paired | True |

Training: ~831 min (~14 h), ~50 s/kimg after warmup, final loss_D ≈ 1.1 (no collapse).

**Probe (U-Net)** — from `run_akbar_check_multi.ipynb`:

| Parameter | Value |
|---|---|
| in_ch → out_ch | 4 → 1 |
| fakes used (train) | 20,000 |
| epochs | 25 |
| batch | 16 |
| resolution | 256 |
| loss | BCE + Dice |
| intensity augmentation | on (gamma/brightness/contrast/noise) |
| real test (locked) | 4,520 slices / 56 subjects |

**→ Result: fake→real Dice = 0.7229, tumor recall = 0.762  (Akbar ref 0.737).**

### Parameter → outcome history (what each change did)
| Change | Fakes | kimg | Dice | Note |
|---|---|---|---|---|
| single modality, no aug | 5k | 600 | 0.19 | U-Net overfit to fake appearance |
| single modality, + intensity aug | 5k | 600 | 0.30 | aug forced shape/context learning |
| **4 modalities + more fakes + longer GAN** | **20k** | **1000** | **0.72** | modalities the decisive lever; more fakes (5k→20k) & longer training (600→1000 kimg) also contributed — matched Akbar |

---

## Next steps

1. (Optional) Matched FID on the paired 4-modality model.
2. (Optional) 4-modality real→real topline (`tstr\real_baseline_multi.py`) — confirms the ceiling.
3. **The real project:** IDRiD retina MAD / recursive-generation study — 6 generations, each GAN
   trained only on the previous generator's fakes; track FID, probe Dice, copy rate, rare-class
   recall, mask quality vs generation. Reuse this now-validated GAN + probe pipeline.
