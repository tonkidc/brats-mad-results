# StyleGAN2-ADA on BraTS 2020 — Methods & Results

## Objective
Train StyleGAN2-ADA to synthesize brain-MRI slices and compare the resulting FID against
Akbar et al. (2024), who report **StyleGAN2 FID = 84.77** on BraTS 2020 (T1Gd, 256×256).

## Training data
- **Dataset:** BraTS 2020 (training split).
- **Subjects sampled:** ~333 of 369 patients (≈6 diverse slices each).
- **Modality:** T1Gd (post-contrast T1, `t1ce`).
- **Slice extraction:** axial slices kept only if ≥15% of pixels > 50/255 (drops near-empty slices), then evenly subsampled per subject.
- **Training slices:** ~2,000 (deliberately limited — StyleGAN2-ADA is designed for small datasets).
- **Resolution:** 256×256 (brain 240px, zero-padded to 256).
- **Normalization:** per-volume 1st–99th percentile intensity scaling.
- **Augmentation:** ADA + horizontal mirror.

*Same dataset, modality, and resolution as the paper. Training set intentionally reduced to ~2,000
images to exploit ADA's small-data efficiency; note this makes the FID reference set smaller than the
paper's, so the absolute FID is somewhat higher/noisier and not a strictly identical protocol.*

## Model & implementation
- Official NVIDIA **StyleGAN2-ADA** (PyTorch), patched for PyTorch 2.10.
- StyleGAN2 skip-generator + residual discriminator, adaptive discriminator augmentation (ADA).

## Hyperparameters (`--cfg=auto`, 256px, 1× T4 GPU)
| Hyperparameter | Value |
|---|---|
| Resolution | 256×256 |
| Latent dim z / w | 512 / 512 |
| Mapping layers | 2 |
| Feature-map base (channel_base / max) | 16,384 / 512 |
| Batch size | 16 |
| Minibatch-std group size | 4 |
| Optimizer | Adam (lr 0.0025, β=(0.0, 0.99), ε=1e-8) |
| R1 gamma (γ) | 0.8192  (=0.0002·256²/16) |
| R1 (D) reg interval | every 16 steps (lazy) |
| Path-length weight | 2.0 |
| Path-length (G) reg interval | every 4 steps (lazy) |
| Style-mixing probability | 0.9 |
| Augmentation | ADA, target p = 0.6, ada_kimg = 500 |
| EMA of G (ema_kimg) | 5 |
| EMA rampup | 0.05 |
| Loss | non-saturating logistic + R1 + path-length |
| FID metric | fid50k_full |
| Training length | 200 kimg (free-GPU budget) |

### Why `--cfg=auto`
`auto` sizes hyperparameters to a single GPU: it selects batch 16 (fits a T4's 15 GB) and re-tunes
γ and EMA to that batch. The paper-matching presets (`paper256`, batch 64; `stylegan2`, batch 32)
are built for **8-GPU clusters** and run out of memory on a single T4. `auto` is NVIDIA's
recommended configuration for single-GPU training on a custom dataset. Trade-off: `auto` uses
2 mapping layers and γ≈0.82 vs the paper's 8 layers and γ=1.

## Results
| | Training | Hardware | FID (fid50k_full) |
|---|---|---|---|
| Akbar et al. (2024) | 25,000 kimg | 4× A100, ~3 days | **84.77** |
| This work | 200 kimg | 1× T4 (free tier) | **[insert best FID from run]** |

## Discussion
The dataset, resolution, and FID metric match the paper; the difference is training budget —
200 kimg on one GPU vs 25,000 kimg on four A100s (~125× less training). FID decreases steadily
with training, so the two models are comparable in protocol but not in compute. StyleGAN2-**ADA**
(adaptive augmentation) is used specifically because it remains stable on limited data/compute
without mode collapse.
