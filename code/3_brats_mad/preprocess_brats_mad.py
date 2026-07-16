"""Preprocess BraTS 2020 into 7-channel (4 modalities + 3 tumor subregions) 256px slices for the
MAD / recursive-generation study.

Channels = [t1, t1ce, t2, flair, WT, TC, ET]:
  ch0..3  four MR modalities, each 1-99 percentile normalized on its own brain -> [0,1]
  ch4  WT  Whole Tumor      = seg>0            (labels 1,2,4)  -- common
  ch5  TC  Tumor Core       = (seg==1)|(seg==4)                -- medium
  ch6  ET  Enhancing Tumor  = seg==4                           -- SMALL/HARD = the rare class

Same subject split as the milestone-1 preprocessing (seed 0, 56 test) so the LOCKED test set is the
same held-out subjects. U-Net probe contract: in_ch=4 (modalities) -> out_ch=3 (WT,TC,ET).
"""
import glob, os, numpy as np, nibabel as nib, random

ROOT = r"C:\Users\Tonkid\Downloads\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT  = r"C:\Users\Tonkid\Downloads\tstr\data_brats_mad"
N_TEST_SUBJ = 56
GAN_TARGET  = 3000
MODS = ["t1", "t1ce", "t2", "flair"]

subjs = sorted(glob.glob(os.path.join(ROOT, "BraTS20_Training_*")))
random.seed(0); random.shuffle(subjs)                 # SAME seed/split as milestone-1
test_subj = subjs[:N_TEST_SUBJ]
train_subj = subjs[N_TEST_SUBJ:]
print(f"subjects: {len(train_subj)} train, {len(test_subj)} test")


def norm_bounds(vol):
    brain = vol[vol > 0]
    if brain.size == 0:
        return None, None
    return np.percentile(brain, 1), np.percentile(brain, 99)


def brain_slices(vol, lo, hi):
    out = []
    for z in range(vol.shape[2]):
        s = np.clip((vol[:, :, z] - lo) / (hi - lo + 1e-8), 0, 1) * 255
        if (s > 50).mean() >= 0.15:
            out.append(z)
    return out


def pad256(a):
    c = np.zeros((256, 256), np.float32); c[8:248, 8:248] = a; return c


def process(subj_list, out_dir, per_subj_cap):
    os.makedirs(out_dir, exist_ok=True)
    kept = {"WT": 0, "TC": 0, "ET": 0}; n = 0
    for si, s in enumerate(subj_list):
        if si % 10 == 0:
            print(f"    ...subject {si}/{len(subj_list)}  ({n} slices)", flush=True)
        paths = {m: glob.glob(os.path.join(s, f"*{m}*.nii*")) for m in MODS}
        paths["t1"] = [p for p in paths["t1"] if "t1ce" not in os.path.basename(p).lower()]
        fm = glob.glob(os.path.join(s, "*seg*.nii*"))
        if not all(paths[m] for m in MODS) or not fm:
            print("  skip (missing):", os.path.basename(s)); continue
        vols = {}; ok = True
        for m in MODS:
            v = nib.load(paths[m][0]).get_fdata(); lo, hi = norm_bounds(v)
            if lo is None: ok = False; break
            vols[m] = (v, lo, hi)
        if not ok: continue
        seg = nib.load(fm[0]).get_fdata()
        vref, lo_ref, hi_ref = vols["t1ce"]
        zs = brain_slices(vref, lo_ref, hi_ref)
        if per_subj_cap:
            zs = zs[:: max(1, len(zs) // per_subj_cap)][:per_subj_cap]
        name = os.path.basename(s)
        for z in zs:
            chans = []
            for m in MODS:
                v, lo, hi = vols[m]
                img = np.clip((v[:, :, z] - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
                chans.append(pad256(img))
            sl = seg[:, :, z]
            chans.append(pad256((sl > 0).astype(np.float32)))               # WT
            chans.append(pad256(((sl == 1) | (sl == 4)).astype(np.float32))) # TC
            chans.append(pad256((sl == 4).astype(np.float32)))              # ET (rare)
            arr = np.stack(chans, 0)                                        # (7,256,256)
            np.save(os.path.join(out_dir, f"{name}_{z:03d}.npy"), arr)
            n += 1
            for i, k in enumerate(["WT", "TC", "ET"]):
                kept[k] += int(arr[4 + i].sum() > 0)
    pct = {k: f"{100*v/max(1,n):.0f}%" for k, v in kept.items()}
    print(f"  {out_dir}: {n} slices | slices-with-region {kept} ({pct})")
    return n


if __name__ == "__main__":
    per = max(1, GAN_TARGET // len(train_subj) + 1)
    print("processing train ...")
    process(train_subj, os.path.join(OUT, "train"), per)
    print("processing test (LOCKED) ...")
    process(test_subj, os.path.join(OUT, "test"), None)
    print("DONE ->", OUT)
