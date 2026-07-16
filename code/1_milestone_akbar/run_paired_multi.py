"""Full 5-channel paired StyleGAN2-ADA run (4 modalities + tumor mask, 256px) -- MILESTONE 1:
match Akbar's setup (t1, t1ce, t2, flair + whole-tumor mask).

Run in a TERMINAL, not the notebook (the ~20-min first-tick MIOpen warmup trips VS Code's
"kernel crashed" watchdog; a terminal process has no such watchdog):

    & "c:\\Users\\Tonkid\\AppData\\Local\\Python\\pythoncore-3.12-64\\python.exe" `
        "C:\\Users\\Tonkid\\Downloads\\run_paired_multi.py" *>&1 | Tee-Object "C:\\Users\\Tonkid\\Downloads\\run_paired_multi.log"

PREREQUISITE: run tstr/preprocess_paired_multi.py first (creates data_multi\train + data_multi\test).
First kimg line is slow (one-time kernel search), then steady. 5 channels is a touch heavier
than 2 but same order of magnitude. Samples in runs\\paired256_multi\\samples show all 5
channels stacked (t1, t1ce, t2, flair, mask). Healthy: loss_D ~0.7-1.5.
"""
import os, sys, argparse
os.environ["WANDB_MODE"] = "disabled"
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")

# Tunable parameters are passed in from the notebook (milestone1.ipynb, Step 2).
# Defaults below match the notebook so the script also works on its own.
ap = argparse.ArgumentParser()
ap.add_argument("--kimg", type=int, default=1000)          # how long the GAN trains (main dial)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=0.0015)
ap.add_argument("--r1_gamma", type=float, default=10.0)
ap.add_argument("--p_init", type=float, default=0.5)
a = ap.parse_args()

from stylegan2_ada.train import Config, train
cfg = Config(
    data=r"C:\Users\Tonkid\Downloads\tstr\data_multi\train",
    out=r"C:\Users\Tonkid\Downloads\runs\paired256_multi",
    resolution=256, paired=True, batch=a.batch,
    kimg=a.kimg,
    lr=a.lr, r1_gamma=a.r1_gamma, p_init=a.p_init, ada_kimg=100, ada_target=0.6, ema_kimg=10.0,
    sample_every_kimg=20, ckpt_every_kimg=100, fid_every_kimg=0,
    amp=True, workers=0, seed=0,
)
print(f"config: kimg={a.kimg} batch={a.batch} lr={a.lr} r1_gamma={a.r1_gamma} p_init={a.p_init}", flush=True)
train(cfg)
print("5-CHANNEL PAIRED GAN DONE ->", cfg.out)
