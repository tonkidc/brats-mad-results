"""5-channel topline: train the U-Net on REAL 4-modality pairs, test on REAL locked test.
This is the ceiling for the fake->real number. With all 4 modalities it should land much
closer to Akbar's 0.737 than the single-modality 0.66."""
import os, sys
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")
from tstr.unet import train_probe

r = train_probe(
    [r"C:\Users\Tonkid\Downloads\tstr\data_multi\train"],   # REAL 4-modality train
    r"C:\Users\Tonkid\Downloads\tstr\data_multi\test",       # REAL locked test
    out_dir=r"C:\Users\Tonkid\Downloads\tstr\probe_real_baseline_multi",
    in_ch=4, out_ch=1, class_names=["tumor"],
    epochs=30, batch=16, res=256, log_every=1,
)
print("\n==============================================")
print(f"  REAL 4-modality U-Net Dice (topline) = {r['dice']:.4f}")
print("  (single-modality real topline was 0.66; Akbar ref 0.737)")
print("==============================================")
