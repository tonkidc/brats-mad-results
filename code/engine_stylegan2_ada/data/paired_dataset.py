"""PairedSliceDataset: loads (image..., mask) pairs saved as (C,H,W) .npy by the
preprocess scripts. The LAST channel is always the binary whole-tumor mask; every
channel before it is an image modality. Returns a float32 tensor (C,H,W) in [-1,1]
plus an empty label. Channel count is detected from the data, so the SAME class
trains both the 2-channel (T1Gd+mask) and the 5-channel (t1,t1ce,t2,flair+mask)
paired StyleGAN2-ADA that generate image+mask jointly."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PairedSliceDataset(Dataset):
    def __init__(self, root, resolution=256, xflip=False):
        self.root = Path(root)
        self.resolution = resolution
        self.xflip = xflip
        self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy pairs in {self.root!s}. Run preprocess_paired.py first.")
        self._n = len(self.files)
        self._nc = int(np.load(self.files[0]).shape[0])   # detect channels from data

    @property
    def num_channels(self):
        return self._nc

    def __len__(self):
        return self._n * (2 if self.xflip else 1)

    def __getitem__(self, idx):
        flip = self.xflip and (idx >= self._n)
        arr = np.load(self.files[idx % self._n]).astype(np.float32)  # (C,H,W): imgs[0,1] + mask{0,1} (last)
        if flip:
            arr = arr[:, :, ::-1].copy()
        t = torch.from_numpy(arr)
        if t.shape[-1] != self.resolution:                # resize: area for images, nearest for mask
            img = F.interpolate(t[None, :-1], size=(self.resolution, self.resolution), mode="area")
            msk = F.interpolate(t[None, -1:], size=(self.resolution, self.resolution), mode="nearest")
            t = torch.cat([img, msk], dim=1)[0]
        t = t * 2.0 - 1.0                                 # -> [-1,1]; mask -> {-1,1}
        return t, torch.zeros(0, dtype=torch.float32)
