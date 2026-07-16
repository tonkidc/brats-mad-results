# Code

Source for all three studies. A custom, pure-PyTorch StyleGAN2-ADA (no NVIDIA CUDA-extension deps —
runs on AMD ROCm) plus channel-generic preprocessing and a U-Net probe for fake→real evaluation.

```
code/
├── engine_stylegan2_ada/          Core GAN package — milestone + BraTS MAD (channel-generic paired G/D,
│                                   ADA augment, train loop w/ crash-safe resume, FID)
├── engine_stylegan2_ada_retina/   Same engine, 7-channel retina variant (RGB + 4 lesion masks)
│
├── 1_milestone_akbar/             BraTS 4-modality paired GAN → fake→real Dice 0.72 ≈ Akbar
│   ├── preprocess_paired_multi.py    build 5-ch (4 MR mods + WT mask) slices
│   ├── run_paired_multi.py           train the GAN
│   ├── generate_pairs.py             sample synthetic image+mask pairs
│   ├── real_baseline_multi.py        real-trained U-Net topline
│   └── milestone1.ipynb              launcher/monitor
│
├── 2_retina_mad/                  IDRiD recursive-generation study (negative result)
│   ├── retina_pilot_runA.py          standalone Run-A pilot (all validated fns)
│   ├── retina_gen1_convergence.py    kimg-convergence sweep + SE montages
│   └── retina_gen1_probe.py          gen-1 gate (fakes → U-Net → locked test)
│
└── 3_brats_mad/                   BraTS multi-class MAD (this study)
    ├── preprocess_brats_mad.py       build 7-ch (4 mods + WT/TC/ET masks) slices
    ├── brats_mad_runA.py             the Run-A recursion driver (gen1..6)
    ├── brats_mad_kimg_sweep.py       find shortest adequate training length
    ├── unet.py                       channel-generic U-Net probe (in_ch→out_ch, per-class Dice)
    └── brats_mad.ipynb               launcher/monitor/results
```

## Notes for reading / reuse
- **Paths are absolute** to the machine these ran on (`C:\Users\Tonkid\Downloads\...`). The code documents
  the exact pipeline that produced the results; to run elsewhere, edit the path constants at the top of
  each script (`ROOT`, `DATA`, `OUT`, `RUNS`, `RESULTS`, and the `sys.path.insert`).
- **Not included** (by design): datasets (BraTS/IDRiD — patient data), generated `.npy` fakes, and model
  checkpoints (`.pt`). This is code + results only.
- **Method everywhere:** train a U-Net on *synthetic* data, score it on a *locked real* test set
  (fake→real Dice / TSTR). The generator never sees the test set; the U-Net never trains on real data
  (except the milestone real-baseline topline).
- **Environment:** AMD RX 7800 XT, `torch 2.9.1+rocm7.2.1` on Windows (ROCm reports as CUDA). The trainer
  routes around a couple of MIOpen gfx1101 quirks (im2col fallback; GroupNorm instead of the broken
  batch-norm training kernel in the probe).
