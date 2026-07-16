"""Reusable U-Net "probe" for the recursive-generation (MAD) study.

One tool, both datasets, both runs:
  * datasets: BraTS validation (in=1 gray -> out=1 tumor)  OR  IDRiD retina (in=3 RGB -> out=4 lesions)
  * runs:     Run 1 synthetic-only = [genN_fakes]   |   Run 2 real+fakes = [real_train, genN_fakes]

It trains a FRESH U-Net on the given paired data, then scores Dice on a LOCKED real test set
(never used for training or model selection). Reports overall Dice + PER-CLASS Dice & recall,
so the rare-class metric (e.g. Soft Exudates) falls straight out.

Data contract: every folder holds (C,H,W) float32 .npy where the FIRST `in_ch` channels are the
image and the NEXT `out_ch` channels are binary masks. Values may be [0,1] or [-1,1]; both handled.
  BraTS: (2,H,W) = [image, tumor_mask]                      in_ch=1, out_ch=1
  Retina:(7,H,W) = [R,G,B, MA,HE,EX,SE]                     in_ch=3, out_ch=4

Usage (Python):
    from tstr.unet import train_probe
    r = train_probe(["...gen1_fakes"], test_dir="...test", out_dir="...probe_gen1",
                    in_ch=3, out_ch=4, class_names=["MA","HE","EX","SE"], epochs=40)
    print(r["dice"], r["per_class_dice"], r["per_class_recall"])
"""
import os
os.environ.setdefault("MIOPEN_FIND_MODE", "NORMAL")   # 256px speed on ROCm (before torch)
os.environ.pop("MIOPEN_FIND_ENFORCE", None)
import glob, json, argparse, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------- data -----------------------------
class SegDataset(Dataset):
    def __init__(self, dirs, in_ch, out_ch, res=256, augment=False):
        if isinstance(dirs, (str, os.PathLike)):
            dirs = [dirs]
        self.files = []
        for d in dirs:
            self.files += sorted(glob.glob(os.path.join(str(d), "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy in {dirs}")
        self.in_ch, self.out_ch, self.res, self.augment = in_ch, out_ch, res, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        arr = np.load(self.files[i]).astype(np.float32)          # (C,H,W)
        img, msk = arr[: self.in_ch], arr[self.in_ch : self.in_ch + self.out_ch]
        if img.min() < -0.01:                                    # [-1,1] -> [0,1]; mask -> {0,1}
            img = (img + 1.0) * 0.5
            msk = (msk > 0.0).astype(np.float32)
        else:
            msk = (msk > 0.5).astype(np.float32)
        img = np.clip(img, 0, 1)
        if img.shape[-1] != self.res:
            ti = F.interpolate(torch.from_numpy(img)[None], (self.res, self.res), mode="area")[0].numpy()
            tm = F.interpolate(torch.from_numpy(msk)[None], (self.res, self.res), mode="nearest")[0].numpy()
            img, msk = ti, (tm > 0.5).astype(np.float32)
        if self.augment:
            # geometric (image + mask together) -- horizontal flip only (keeps anatomy plausible)
            if np.random.rand() < 0.5:
                img, msk = img[:, :, ::-1].copy(), msk[:, :, ::-1].copy()
            # INTENSITY augmentation (image only) -- the key to closing the synthetic->real gap.
            # Forces the net to learn tumor shape/context, not fake-specific brightness/texture.
            brain = img > 0.03
            g = np.random.uniform(0.6, 1.6)                              # gamma
            img = np.clip(img, 0, 1) ** g
            img = np.clip(img * np.random.uniform(0.75, 1.3) +
                          np.random.uniform(-0.1, 0.1), 0, 1)            # contrast + brightness
            if np.random.rand() < 0.5:                                   # gaussian noise
                img = np.clip(img + np.random.randn(*img.shape).astype(np.float32) * np.random.uniform(0.01, 0.05), 0, 1)
            img = (img * brain).astype(np.float32)                       # keep background black
        return torch.from_numpy(img), torch.from_numpy(msk)


# ----------------------------- model -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co), nn.LeakyReLU(0.1, True),
            nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co), nn.LeakyReLU(0.1, True))
    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base);      self.d2 = DoubleConv(base, base * 2)
        self.d3 = DoubleConv(base * 2, base * 4); self.d4 = DoubleConv(base * 4, base * 8)
        self.bott = DoubleConv(base * 8, base * 16); self.pool = nn.MaxPool2d(2)
        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2); self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2);  self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2);  self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2);      self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(self.pool(x1)); x3 = self.d3(self.pool(x2)); x4 = self.d4(self.pool(x3))
        xb = self.bott(self.pool(x4))
        y = self.c4(torch.cat([self.u4(xb), x4], 1)); y = self.c3(torch.cat([self.u3(y), x3], 1))
        y = self.c2(torch.cat([self.u2(y), x2], 1)); y = self.c1(torch.cat([self.u1(y), x1], 1))
        return self.out(y)


# ----------------------------- loss / metrics -----------------------------
def _bce_dice(logits, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=[0, 2, 3]); denom = p.sum([0, 2, 3]) + target.sum([0, 2, 3])
    dice = (1 - (2 * inter + eps) / (denom + eps)).mean()
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device, out_ch):
    """Per-class Dice and recall on binarized preds, averaged over images."""
    model.eval()
    d_sum = np.zeros(out_ch); r_sum = np.zeros(out_ch); n = 0
    for img, msk in loader:
        img, msk = img.to(device), msk.to(device)
        pred = (torch.sigmoid(model(img)) > 0.5).float()
        inter = (pred * msk).sum([2, 3])                          # (B,out_ch)
        dice = (2 * inter + 1e-6) / (pred.sum([2, 3]) + msk.sum([2, 3]) + 1e-6)
        recall = (inter + 1e-6) / (msk.sum([2, 3]) + 1e-6)
        d_sum += dice.sum(0).cpu().numpy(); r_sum += recall.sum(0).cpu().numpy(); n += img.shape[0]
    return d_sum / n, r_sum / n


# ----------------------------- probe -----------------------------
def train_probe(train_dirs, test_dir, out_dir, in_ch=1, out_ch=1, class_names=None,
                epochs=40, batch=16, lr=1e-3, res=256, base=32, device="cuda", seed=0, log_every=5):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    class_names = class_names or [f"c{i}" for i in range(out_ch)]
    tr = SegDataset(train_dirs, in_ch, out_ch, res, augment=True)
    te = SegDataset(test_dir, in_ch, out_ch, res, augment=False)
    print(f"probe: train {len(tr)} from {train_dirs} | test {len(te)} | in{in_ch}->out{out_ch}")
    trl = DataLoader(tr, batch, shuffle=True, drop_last=True, num_workers=0)
    tel = DataLoader(te, batch, shuffle=False, num_workers=0)

    model = UNet(in_ch, out_ch, base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for img, msk in trl:
            img, msk = img.to(device), msk.to(device)
            opt.zero_grad(set_to_none=True)
            loss = _bce_dice(model(img), msk); loss.backward(); opt.step(); tot += float(loss)
        if ep % log_every == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  loss {tot/len(trl):.4f}")

    pcd, pcr = evaluate(model, tel, device, out_ch)               # ONE honest evaluation
    result = {"dice": float(pcd.mean()),
              "per_class_dice": {c: float(v) for c, v in zip(class_names, pcd)},
              "per_class_recall": {c: float(v) for c, v in zip(class_names, pcr)},
              "n_test": len(te), "train_dirs": [str(d) for d in (train_dirs if isinstance(train_dirs, list) else [train_dirs])]}
    torch.save({"model": model.state_dict(), **result}, os.path.join(out_dir, "unet.pt"))
    json.dump(result, open(os.path.join(out_dir, "probe.json"), "w"), indent=2)
    print(f"\n  PROBE mean Dice (locked test) = {result['dice']:.4f}")
    for c in class_names:
        print(f"    {c:>4}: dice {result['per_class_dice'][c]:.3f}  recall {result['per_class_recall'][c]:.3f}")
    print(f"  saved -> {out_dir}\\probe.json")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="comma-separated train dir(s)")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in_ch", type=int, default=1)
    ap.add_argument("--out_ch", type=int, default=1)
    ap.add_argument("--classes", default="", help="comma-separated class names")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args()
    train_probe(a.train.split(","), a.test, a.out, in_ch=a.in_ch, out_ch=a.out_ch,
                class_names=(a.classes.split(",") if a.classes else None),
                epochs=a.epochs, batch=a.batch, res=a.res)
