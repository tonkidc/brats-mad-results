"""Convert BRATS2013 `.mha` 3D MRI volumes into 2D grayscale PNG slices.

BRATS layout (after unzipping Challenge + Leaderboard into --raw):
    <raw>/**/VSD.Brain.XX.O.MR_<modality>/*.mha        (T1, T1c, T2, Flair)

For each volume we:
  * read it with SimpleITK             -> array (Z, Y, X)
  * normalize by the 1..99 percentile  -> robust to MRI intensity outliers
  * keep only slices with enough brain  -> drop near-empty top/bottom slices
  * resize to `resolution`x`resolution` -> Lanczos
  * write uint8 PNG                      -> <out>/<subject>_<modality>_z<zzz>.png

Run:
    python -m stylegan2_ada.data.prepare_brats --raw data/brats_raw \
        --out data/brats_slices --resolution 128
"""

import argparse
import os
from pathlib import Path

import numpy as np


def _load_volume(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
    return arr


def _normalize(vol):
    """Percentile-clip to [0,1] using non-zero voxels (background is exactly 0 in BRATS)."""
    nz = vol[vol > 0]
    if nz.size == 0:
        return None
    lo, hi = np.percentile(nz, 1.0), np.percentile(nz, 99.0)
    if hi <= lo:
        return None
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return vol


def _modality_of(path):
    # .../VSD.Brain.XX.O.MR_T1c/VSD.Brain.XX.O.MR_T1c.17570.mha
    name = Path(path).parent.name  # VSD.Brain.XX.O.MR_T1c
    if "_" in name:
        return name.rsplit("_", 1)[-1]  # T1c / T1 / T2 / Flair
    return "MR"


def _subject_of(path, raw_root):
    # subject id is the numeric folder two levels up (e.g. .../HG/0301/<modality>/*.mha)
    parts = Path(path).relative_to(raw_root).parts
    # find the modality folder, subject is the part before it
    for i, p in enumerate(parts):
        if p.startswith("VSD.Brain") and i >= 1:
            return parts[i - 1]
    # fallback: parent-of-parent folder name
    return Path(path).parent.parent.name


def prepare(raw, out, resolution=128, min_brain_frac=0.02, verbose=True):
    from PIL import Image

    raw_root = Path(raw)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mha_files = sorted(raw_root.rglob("*.mha"))
    if not mha_files:
        raise FileNotFoundError(
            f"No .mha files under {raw_root!s}. Unzip the BRATS Challenge/Leaderboard "
            f"archives into this folder first."
        )

    written = 0
    for vi, mha in enumerate(mha_files):
        subject = _subject_of(mha, raw_root)
        modality = _modality_of(mha)
        vol = _load_volume(mha)
        vol = _normalize(vol)
        if vol is None:
            continue
        z_dim = vol.shape[0]
        for z in range(z_dim):
            sl = vol[z]
            if (sl > 0.02).mean() < min_brain_frac:  # skip near-empty slices
                continue
            img = (sl * 255.0).astype(np.uint8)
            pil = Image.fromarray(img, mode="L").resize(
                (resolution, resolution), Image.LANCZOS
            )
            fname = f"{subject}_{modality}_z{z:03d}.png"
            pil.save(out_dir / fname)
            written += 1
        if verbose:
            print(f"[{vi + 1}/{len(mha_files)}] {subject} {modality}: total slices so far = {written}")

    if verbose:
        print(f"Done. Wrote {written} PNG slices to {out_dir}")
    return written


def main(raw=None, out=None, resolution=128):
    """Callable entry point (notebook-friendly, Windows-safe -- no shell quoting)."""
    if raw is None:  # CLI path
        ap = argparse.ArgumentParser(description="BRATS .mha -> 2D PNG slices")
        ap.add_argument("--raw", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--resolution", type=int, default=128)
        ap.add_argument("--min-brain-frac", type=float, default=0.02)
        a = ap.parse_args()
        return prepare(a.raw, a.out, a.resolution, a.min_brain_frac)
    return prepare(raw, out, resolution)


if __name__ == "__main__":
    main()
