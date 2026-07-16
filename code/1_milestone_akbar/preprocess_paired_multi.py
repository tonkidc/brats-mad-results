"""Preprocess BraTS 2020 into 5-channel (4 modalities + whole-tumor mask) 256px slices,
matching Akbar et al.: channels = [t1, t1ce, t2, flair, mask].

Each slice is saved as a (5,256,256) float32 .npy:
  ch0=t1[0,1]  ch1=t1ce[0,1]  ch2=t2[0,1]  ch3=flair[0,1]  ch4=WT mask{0,1} (seg>0)
Every modality is normalized on its OWN brain (1-99 percentile). Slice selection uses
t1ce brain content, so the z-set is identical across modalities.

Same subject split as preprocess_paired.py (seed 0, 56 test) => the locked test set is the
SAME subjects, just now with all 4 modalities. Data contract for the U-Net probe:
  in_ch=4 (4 modalities), out_ch=1 (mask)."""
import glob, os, numpy as np, nibabel as nib, random

ROOT = r"C:\Users\Tonkid\Downloads\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
OUT  = r"C:\Users\Tonkid\Downloads\tstr\data_multi"
N_TEST_SUBJ = 56          # Akbar's BraTS2020 test size
GAN_TARGET  = 3000        # ~slices for GAN/real-train (kept tractable)
MODS = ["t1", "t1ce", "t2", "flair"]   # Akbar order: T1w, T1wGD, T2w, FLAIR

subjs = sorted(glob.glob(os.path.join(ROOT, "BraTS20_Training_*")))
random.seed(0); random.shuffle(subjs)                 # SAME seed/split as 2-ch version
test_subj = subjs[:N_TEST_SUBJ]
train_subj = subjs[N_TEST_SUBJ:]
print(f"subjects: {len(train_subj)} train, {len(test_subj)} test")


def norm(vol):
    """1-99 percentile normalize a modality on its own brain -> [0,1]."""
    brain = vol[vol > 0]
    if brain.size == 0:
        return None, None, None
    lo, hi = np.percentile(brain, 1), np.percentile(brain, 99)
    return vol, lo, hi


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
    kept, tumor = 0, 0
    for si, s in enumerate(subj_list):
        if si % 10 == 0:
            print(f"    ...subject {si}/{len(subj_list)}  ({kept} slices so far)", flush=True)
        paths = {m: glob.glob(os.path.join(s, f"*{m}*.nii*")) for m in MODS}
        fm = glob.glob(os.path.join(s, "*seg*.nii*"))
        # t1ce path also matches "*t1*"; disambiguate below by exact tag
        paths["t1"] = [p for p in paths["t1"] if "t1ce" not in os.path.basename(p).lower()]
        if not all(paths[m] for m in MODS) or not fm:
            print("  skip (missing modality):", os.path.basename(s)); continue
        vols = {}
        ok = True
        for m in MODS:
            v, lo, hi = norm(nib.load(paths[m][0]).get_fdata())
            if v is None: ok = False; break
            vols[m] = (v, lo, hi)
        if not ok:
            continue
        seg = nib.load(fm[0]).get_fdata()
        vref, lo_ref, hi_ref = vols["t1ce"]                       # slice selection on t1ce
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
            chans.append(pad256((seg[:, :, z] > 0).astype(np.float32)))   # mask = last channel
            arr = np.stack(chans, 0)                                       # (5,256,256)
            np.save(os.path.join(out_dir, f"{name}_{z:03d}.npy"), arr)
            kept += 1; tumor += int(arr[-1].sum() > 0)
    print(f"  {out_dir}: {kept} slices, {tumor} with tumor ({100*tumor/max(1,kept):.0f}%)")
    return kept


if __name__ == "__main__":
    per = max(1, GAN_TARGET // len(train_subj) + 1)
    print("processing train (5-ch GAN + real baseline) ...")
    process(train_subj, os.path.join(OUT, "train"), per)
    print("processing test (real held-out, 5-ch) ...")
    process(test_subj, os.path.join(OUT, "test"), None)   # keep all test slices
    print("DONE ->", OUT)
