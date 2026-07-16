"""PILOT — Run A only (synthetic-only recursion), scaled down for a ~12 h experiment.

Reuses the SAME validated functions as retina_autophagy_local.ipynb, but with pilot knobs
(KIMG=200, N_FAKES=3000) and runs Run A generations 1..4 automatically, end to end.
Writes one row per generation to results_pilot/metrics_pilot.csv.

Pilot ≠ final: lighter training, fewer fakes, Run A only. Goal = confirm the GPU pipeline runs
and see whether Soft-Exudate (rare) recall falls BEFORE FID/mean-Dice. If it looks good, rerun
the full study (KIMG=400, 6 gens, Runs A+B) from the main notebook.

Launched in the background by retina_pilot.ipynb (so the long first-gen MIOpen warmup can't trip
VS Code's kernel watchdog). Separate output dirs => does NOT touch the main study's runs/results.
"""
import os, sys, glob, json, csv, random, time
from pathlib import Path
os.environ["MIOPEN_FIND_MODE"] = "NORMAL"
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, r"C:\Users\Tonkid\Downloads")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

print("torch", torch.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE", flush=True)
assert torch.cuda.is_available(), "GPU not visible"
DEVICE = torch.device("cuda")

# ---------------- config (paths shared with main study; outputs are pilot-only) ----------------
IDRID = r"C:\Users\Tonkid\Downloads\A. Segmentation\A. Segmentation"
IMG_TRAIN = os.path.join(IDRID, "1. Original Images", "a. Training Set")
IMG_TEST  = os.path.join(IDRID, "1. Original Images", "b. Testing Set")
GT_TRAIN  = os.path.join(IDRID, "2. All Segmentation Groundtruths", "a. Training Set")
GT_TEST   = os.path.join(IDRID, "2. All Segmentation Groundtruths", "b. Testing Set")

PROJ    = r"C:\Users\Tonkid\Downloads\retina_autophagy"
PATCHES = os.path.join(PROJ, "patches")            # reused (real data); prep once
RUNS    = os.path.join(PROJ, "runs_pilot")         # pilot-only
RESULTS = os.path.join(PROJ, "results_pilot")      # pilot-only
for d in (PROJ, PATCHES, RUNS, RESULTS):
    os.makedirs(d, exist_ok=True)
REAL_TRAIN = os.path.join(PATCHES, "train")
CSV = os.path.join(RESULTS, "metrics_pilot.csv")

LESIONS  = ["MA", "HE", "EX", "SE"]
LES_DIRS = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages",
            "EX": "3. Hard Exudates",  "SE": "4. Soft Exudates"}
RARE = "SE"; RARE_IDX = LESIONS.index(RARE)

# ---- FIXED-within-pilot knobs ----
RES = 256; WORK_W = 2048; PATCH_PER = 100; LESION_FR = 0.70
N_FAKES = 5000        # matches full study
KIMG    = 200         # pilot (full study = 400)
BATCH   = 16
N_GEN_PILOT = 4       # Run A generations 1..4
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

splits = None         # set by prep()

# ============================ prep helpers (verbatim from main nb) ============================
def _load_rgb(path, work_w=WORK_W):
    im = Image.open(path).convert("RGB")
    h = int(round(im.height * work_w / im.width))
    im = im.resize((work_w, h), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)

def _load_mask(gt_dir, stem, suffix, size_wh):
    p = os.path.join(gt_dir, LES_DIRS[suffix], f"{stem}_{suffix}.tif")
    if not os.path.exists(p):
        return np.zeros((size_wh[1], size_wh[0]), dtype=bool)
    m = Image.open(p).convert("L").resize(size_wh, Image.NEAREST)  # force 1-channel (some .tif are RGBA)
    return np.asarray(m) > 0

def load_image_and_masks(img_path, gt_dir):
    rgb = _load_rgb(img_path); H, W = rgb.shape[:2]
    stem = Path(img_path).stem
    masks = np.stack([_load_mask(gt_dir, stem, s, (W, H)) for s in LESIONS], 0)
    return rgb, masks

def fov_bbox(rgb, thr=12):
    gray = rgb.mean(2); ys, xs = np.where(gray > thr)
    if len(xs) == 0: return 0, 0, rgb.shape[0], rgb.shape[1]
    return ys.min(), xs.min(), ys.max(), xs.max()

def _to_patch_array(rgb_crop, mask_crop):
    rgb = rgb_crop.astype(np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    msk = np.where(mask_crop, 1.0, -1.0).astype(np.float32)
    return np.concatenate([rgb, msk], 0).astype(np.float32)

def extract_patches_for_image(rgb, masks, n, lesion_fr, res=RES, rng=None):
    rng = rng or np.random
    H, W = rgb.shape[:2]; y0, x0, y1, x1 = fov_bbox(rgb); half = res // 2
    les_union = masks.any(0); ly, lx = np.where(les_union)
    out = []; tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        if len(ly) and rng.random() < lesion_fr:
            k = rng.randint(len(ly))
            cy = int(ly[k] + rng.randint(-half//2, half//2 + 1))
            cx = int(lx[k] + rng.randint(-half//2, half//2 + 1))
        else:
            cy = rng.randint(y0 + half, max(y0 + half + 1, y1 - half))
            cx = rng.randint(x0 + half, max(x0 + half + 1, x1 - half))
        cy = int(np.clip(cy, half, H - half)); cx = int(np.clip(cx, half, W - half))
        rc = rgb[cy-half:cy+half, cx-half:cx+half]
        if rc.shape[:2] != (res, res): continue
        if (rc.mean(2) > 12).mean() < 0.5: continue
        mc = masks[:, cy-half:cy+half, cx-half:cx+half]
        out.append(_to_patch_array(rc, mc))
    return out

def build_split(stems, img_dir, gt_dir, out_dir, n_per, lesion_fr, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed); cov = {s: 0 for s in LESIONS}; idx = 0
    for stem in stems:
        rgb, masks = load_image_and_masks(os.path.join(img_dir, stem + ".jpg"), gt_dir)
        for p in extract_patches_for_image(rgb, masks, n_per, lesion_fr, rng=rng):
            np.save(os.path.join(out_dir, f"{stem}_p{idx:06d}.npy"), p)
            for j, s in enumerate(LESIONS): cov[s] += int((p[3 + j] > 0).any())
            idx += 1
    print(f"  {out_dir}: {idx} patches | lesion coverage {cov}", flush=True)
    return idx

def prep():
    """Build leakage-safe split + patches once; reuse if already present."""
    global splits
    train_stems_all = sorted(Path(p).stem for p in glob.glob(os.path.join(IMG_TRAIN, "*.jpg")))
    test_stems      = sorted(Path(p).stem for p in glob.glob(os.path.join(IMG_TEST,  "*.jpg")))
    rng = np.random.RandomState(SEED); perm = rng.permutation(len(train_stems_all))
    train_stems = [train_stems_all[i] for i in sorted(perm[:44])]
    val_stems   = [train_stems_all[i] for i in sorted(perm[44:])]
    splits = {"train": train_stems, "val": val_stems, "test": test_stems}
    json.dump(splits, open(os.path.join(PROJ, "splits.json"), "w"), indent=2)
    print("split: train", len(train_stems), "val", len(val_stems), "test(LOCKED)", len(test_stems), flush=True)

    if len(glob.glob(os.path.join(REAL_TRAIN, "*.npy"))) >= 44 * PATCH_PER * 0.5:
        print("patches already present -> skip prep", flush=True)
    else:
        build_split(train_stems, IMG_TRAIN, GT_TRAIN, REAL_TRAIN,                   PATCH_PER, LESION_FR, seed=1)
        build_split(val_stems,   IMG_TRAIN, GT_TRAIN, os.path.join(PATCHES, "val"),  PATCH_PER, LESION_FR, seed=2)
        build_split(test_stems,  IMG_TEST,  GT_TEST,  os.path.join(PATCHES, "test"), PATCH_PER, LESION_FR, seed=3)
    # leakage assert
    def stems_in(d): return {Path(f).stem.rsplit("_p", 1)[0] for f in glob.glob(os.path.join(d, "*.npy"))}
    tr = stems_in(REAL_TRAIN); te = stems_in(os.path.join(PATCHES, "test"))
    assert tr.isdisjoint(te), "LEAK: a test image appears in train!"
    print("leakage check OK", flush=True)

# ============================ generator / fakes ============================
from stylegan2_ada_retina.train import Config, train
from stylegan2_ada_retina.generate import load_generator

def train_generator(data_dirs, out_dir, kimg=KIMG, batch=BATCH):
    cfg = Config(
        data=data_dirs, out=out_dir, dataset="retina", color_channels=3,
        resolution=RES, batch=batch, kimg=kimg,
        lr=0.0015, r1_gamma=10.0, p_init=0.5, ada_kimg=100, ada_target=0.6,
        ema_kimg=10.0, sample_every_kimg=20, ckpt_every_kimg=max(50, kimg),
        fid_every_kimg=0, amp=True, r1_cudnn=None, workers=0, seed=SEED,
        wandb_project="retina-autophagy",
    )
    train(cfg)
    return os.path.join(out_dir, "checkpoints", "final.pt")

@torch.no_grad()
def generate_fakes(ckpt_path, out_dir, n=N_FAKES, batch=BATCH, seed=0, psi=1.0):
    os.makedirs(out_dir, exist_ok=True)
    G, cfg = load_generator(ckpt_path, DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(seed); done = 0
    while done < n:
        b = min(batch, n - done)
        z = torch.randn(b, cfg["z_dim"], device=DEVICE, generator=g)
        img = G(z, truncation_psi=psi, noise_mode="const")
        rgb = img[:, :3].clamp(-1, 1); msk = torch.where(img[:, 3:] > 0, 1.0, -1.0)
        arr = torch.cat([rgb, msk], 1).cpu().numpy().astype(np.float32)
        for i in range(b): np.save(os.path.join(out_dir, f"fake_{done+i:06d}.npy"), arr[i])
        done += b
    print(f"  generated {n} fakes -> {out_dir}", flush=True)
    return out_dir

# ============================ U-Net probe + locked-test eval ============================
class DoubleConv(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        # GroupNorm, not Batch/InstanceNorm: this ROCm build cannot compile MIOpen's batch-norm
        # spatial kernel ('type_traits' HIPRTC error) and InstanceNorm uses the same backend.
        s.net = nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.GroupNorm(8,o), nn.ReLU(True),
                              nn.Conv2d(o,o,3,padding=1), nn.GroupNorm(8,o), nn.ReLU(True))
    def forward(s,x): return s.net(x)

class UNet(nn.Module):
    def __init__(s, n_cls=4, ch=(32,64,128,256)):
        super().__init__()
        s.d1=DoubleConv(3,ch[0]); s.d2=DoubleConv(ch[0],ch[1])
        s.d3=DoubleConv(ch[1],ch[2]); s.d4=DoubleConv(ch[2],ch[3]); s.pool=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(ch[3],ch[2],2,2); s.c3=DoubleConv(ch[3],ch[2])
        s.u2=nn.ConvTranspose2d(ch[2],ch[1],2,2); s.c2=DoubleConv(ch[2],ch[1])
        s.u1=nn.ConvTranspose2d(ch[1],ch[0],2,2); s.c1=DoubleConv(ch[1],ch[0])
        s.out=nn.Conv2d(ch[0], n_cls, 1)
    def forward(s,x):
        x1=s.d1(x); x2=s.d2(s.pool(x1)); x3=s.d3(s.pool(x2)); x4=s.d4(s.pool(x3))
        y=s.c3(torch.cat([s.u3(x4),x3],1)); y=s.c2(torch.cat([s.u2(y),x2],1))
        y=s.c1(torch.cat([s.u1(y),x1],1)); return s.out(y)

class FakeSeg(torch.utils.data.Dataset):
    def __init__(s, dirs):
        dirs = [dirs] if isinstance(dirs,str) else dirs
        s.files=[]; [s.files.extend(glob.glob(os.path.join(d,"*.npy"))) for d in dirs]
    def __len__(s): return len(s.files)
    def __getitem__(s,i):
        a=np.load(s.files[i]); rgb=(a[:3]+1)/2; tgt=(a[3:7]>0).astype(np.float32)
        return torch.from_numpy(rgb).float(), torch.from_numpy(tgt).float()

def dice_loss(logits, tgt, eps=1.):
    p=torch.sigmoid(logits); num=2*(p*tgt).sum((0,2,3))+eps; den=(p+tgt).sum((0,2,3))+eps
    return (1-num/den).mean()

def train_unet(fake_dirs, epochs=15, bs=16, lr=1e-3):
    dl=torch.utils.data.DataLoader(FakeSeg(fake_dirs), batch_size=bs, shuffle=True, drop_last=True)
    net=UNet().to(DEVICE); opt=torch.optim.Adam(net.parameters(), lr)
    for ep in range(epochs):
        net.train(); tot=0
        for x,y in dl:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            out=net(x); loss=F.binary_cross_entropy_with_logits(out,y)+dice_loss(out,y)
            loss.backward(); opt.step(); tot+=loss.item()
        if ep%5==0 or ep==epochs-1: print(f"    unet ep{ep} loss {tot/max(1,len(dl)):.3f}", flush=True)
    return net

@torch.no_grad()
def eval_on_locked_test(net, stride=192, thr=0.5):
    net.eval()
    inter=np.zeros(4); psum=np.zeros(4); gsum=np.zeros(4); tp_se=fn_se=0.0
    for stem in splits["test"]:
        rgb, masks = load_image_and_masks(os.path.join(IMG_TEST, stem + ".jpg"), GT_TEST)
        H,W=rgb.shape[:2]; x=torch.from_numpy(((rgb.astype(np.float32)/127.5-1)+1)/2)
        x=x.permute(2,0,1).unsqueeze(0)
        prob=torch.zeros(1,4,H,W); cnt=torch.zeros(1,1,H,W)
        ys=list(range(0,max(1,H-RES+1),stride))+[H-RES]; xs=list(range(0,max(1,W-RES+1),stride))+[W-RES]
        for yy in ys:
            for xx in xs:
                patch=x[:,:,yy:yy+RES,xx:xx+RES].to(DEVICE)
                p=torch.sigmoid(net(patch)).cpu()
                prob[:,:,yy:yy+RES,xx:xx+RES]+=p; cnt[:,:,yy:yy+RES,xx:xx+RES]+=1
        prob/=cnt.clamp(min=1); pred=(prob[0].numpy()>thr); gt=masks
        inter+=(pred&gt).sum((1,2)); psum+=pred.sum((1,2)); gsum+=gt.sum((1,2))
        tp_se+=(pred[RARE_IDX]&gt[RARE_IDX]).sum(); fn_se+=((~pred[RARE_IDX])&gt[RARE_IDX]).sum()
    dice=(2*inter+1)/(psum+gsum+1); se_recall=tp_se/max(1.0,(tp_se+fn_se))
    return {"dice_per_class":dict(zip(LESIONS,dice.round(4))),
            "dice_mean":float(dice.mean()), "dice_SE":float(dice[RARE_IDX]),
            "se_recall":float(se_recall)}

# ============================ FID / KID ============================
from stylegan2_ada_retina.metrics.fid import _Inception
_INC=None
def _inc():
    global _INC
    if _INC is None: _INC=_Inception(DEVICE)
    return _INC

@torch.no_grad()
def _features(dirs, num=2000, batch=32):
    dirs=[dirs] if isinstance(dirs,str) else dirs
    files=[]; [files.extend(glob.glob(os.path.join(d,"*.npy"))) for d in dirs]
    random.shuffle(files); files=files[:num]; feats=[]
    for i in range(0,len(files),batch):
        arr=np.stack([np.load(f) for f in files[i:i+batch]])
        rgb=torch.from_numpy(((arr[:,:3]+1)/2)).float().to(DEVICE)
        feats.append(_inc().features(rgb).cpu().numpy())
    return np.concatenate(feats,0)

def _fid(fr, ff):
    from scipy import linalg
    mu1,mu2=fr.mean(0),ff.mean(0); s1,s2=np.cov(fr,rowvar=False),np.cov(ff,rowvar=False)
    def sq(a):
        r=linalg.sqrtm(a); r=r[0] if isinstance(r,tuple) else r
        return r.real if np.iscomplexobj(r) else r
    cov=sq(s1@s2)
    if not np.isfinite(cov).all():
        o=np.eye(s1.shape[0])*1e-6; cov=sq((s1+o)@(s2+o))
    d=mu1-mu2; return float(d@d+np.trace(s1)+np.trace(s2)-2*np.trace(cov))

def _kid(fr, ff, deg=3, n_sub=100, sub=1000):
    d=fr.shape[1]
    def k(a,b): return (a@b.T/d+1)**deg
    vals=[]
    for _ in range(n_sub):
        r=fr[np.random.choice(len(fr),min(sub,len(fr)),replace=False)]
        f=ff[np.random.choice(len(ff),min(sub,len(ff)),replace=False)]
        m=len(r); n=len(f); krr=k(r,r); kff=k(f,f); krf=k(r,f)
        np.fill_diagonal(krr,0); np.fill_diagonal(kff,0)
        vals.append(krr.sum()/(m*(m-1))+kff.sum()/(n*(n-1))-2*krf.mean())
    return float(np.mean(vals))

def fid_kid(fake_dir, real_dir=None):
    real_dir=real_dir or REAL_TRAIN
    fr=_features(real_dir); ff=_features(fake_dir)
    return _fid(fr,ff), _kid(fr,ff)

# ============================ driver ============================
FIELDS = ["run","gen","fid","kid","dice_mean","dice_SE","se_recall","dice_MA","dice_HE","dice_EX"]
def _append(row):
    new = not os.path.exists(CSV)
    with open(CSV,"a",newline="") as f:
        w=csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow({k:row.get(k,"") for k in FIELDS})

def gen_dir(gen):   return os.path.join(RUNS, "A", f"gen{gen}")
def fakes_dir(gen): return os.path.join(gen_dir(gen), "fakes")

def run_gen(gen):
    t0=time.time(); out=gen_dir(gen); os.makedirs(out, exist_ok=True)
    data = REAL_TRAIN if gen == 1 else fakes_dir(gen-1)      # Run A: gen1=real, else prev fakes
    ck = os.path.join(out, "checkpoints", "final.pt")
    if not os.path.exists(ck):
        ck = train_generator(data, out, kimg=KIMG)
    fdir = fakes_dir(gen)
    if len(glob.glob(os.path.join(fdir,"*.npy"))) < N_FAKES:
        generate_fakes(ck, fdir, seed=1000+gen)
    net = train_unet(fdir); seg = eval_on_locked_test(net); del net; torch.cuda.empty_cache()
    fid, kid = fid_kid(fdir)
    dpc = seg["dice_per_class"]
    row = {"run":"A","gen":gen,"fid":round(fid,3),"kid":round(kid,5),
           "dice_mean":round(seg["dice_mean"],4),"dice_SE":round(seg["dice_SE"],4),
           "se_recall":round(seg["se_recall"],4),
           "dice_MA":float(dpc["MA"]),"dice_HE":float(dpc["HE"]),"dice_EX":float(dpc["EX"])}
    _append(row)
    print(f"[GEN {gen} DONE in {(time.time()-t0)/60:.0f} min] {row}", flush=True)
    return row

def main():
    print("=== RETINA PILOT — Run A, gens 1..%d, KIMG=%d, N_FAKES=%d ===" % (N_GEN_PILOT, KIMG, N_FAKES), flush=True)
    prep()
    # gen-1 topline for context (U-Net trained on REAL patches)
    print("-- topline: U-Net on REAL patches --", flush=True)
    net_real = train_unet(REAL_TRAIN); m_real = eval_on_locked_test(net_real)
    del net_real; torch.cuda.empty_cache()
    print("real topline:", m_real, flush=True)
    with open(os.path.join(RESULTS, "real_topline.json"), "w") as f: json.dump(m_real, f, indent=2)
    for gen in range(1, N_GEN_PILOT + 1):
        run_gen(gen)
    print("=== PILOT DONE -> %s ===" % CSV, flush=True)

if __name__ == "__main__":
    main()
