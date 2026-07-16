"""SliceDataset: loads the 2D grayscale PNG slices produced by prepare_brats.

Returns (image, label) where image is a float32 tensor of shape (1, H, W) in [-1, 1]
and label is an empty (0,) tensor (this is an unconditional GAN -- BRATS test set has
no usable labels). Optional x-flip doubles the effective dataset size.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SliceDataset(Dataset):
    def __init__(self, root, resolution=128, xflip=False):
        self.root = Path(root)
        self.resolution = resolution
        self.xflip = xflip
        self.files = sorted(self.root.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(
                f"No .png slices in {self.root!s}. Run prepare_brats first."
            )
        self._n = len(self.files)

    @property
    def num_channels(self):
        return 1

    def __len__(self):
        return self._n * (2 if self.xflip else 1)

    def _load(self, idx):
        from PIL import Image
        path = self.files[idx % self._n]
        img = Image.open(path).convert("L")
        if img.size != (self.resolution, self.resolution):
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32)  # (H, W) in [0,255]
        return arr

    def __getitem__(self, idx):
        flip = self.xflip and (idx >= self._n)
        arr = self._load(idx)
        if flip:
            arr = arr[:, ::-1].copy()
        arr = arr / 127.5 - 1.0  # -> [-1, 1]
        img = torch.from_numpy(arr)[None, :, :]  # (1, H, W)
        label = torch.zeros(0, dtype=torch.float32)
        return img, label


class RetinaPatchDataset(Dataset):
    """Loads pre-extracted 7-channel IDRiD patches saved as .npy by prepare_idrid.

    Each .npy is a float32 array of shape (7, H, W): channels 0..2 = RGB fundus in
    [-1, 1], channels 3..6 = lesion masks {MA, HE, EX, SE} in {-1, +1}. Returns
    (image, empty_label) exactly like SliceDataset so the training loop is unchanged;
    num_channels is 7, which flows straight into Generator/Discriminator img_channels.
    """

    def __init__(self, root, resolution=256, xflip=True, color_channels=3):
        # root may be a single dir or a list of dirs (Run B trains on real + prev fakes).
        roots = [root] if isinstance(root, (str, Path)) else list(root)
        self.roots = [Path(r) for r in roots]
        self.resolution = resolution
        self.xflip = xflip
        self.color_channels = color_channels
        self.files = []
        for r in self.roots:
            self.files.extend(sorted(Path(r).glob("*.npy")))
        if not self.files:
            raise FileNotFoundError(
                f"No .npy patches in {[str(r) for r in self.roots]}. "
                f"Run prepare_idrid / generate_fakes first."
            )
        self._n = len(self.files)

    @property
    def num_channels(self):
        return 7

    def __len__(self):
        return self._n * (2 if self.xflip else 1)

    def __getitem__(self, idx):
        flip = self.xflip and (idx >= self._n)
        arr = np.load(self.files[idx % self._n]).astype(np.float32)  # (7, H, W)
        if arr.shape[-1] != self.resolution:
            # patches are pre-sized; guard only, no silent resample of masks
            raise ValueError(f"patch {self.files[idx % self._n]} is {arr.shape}, expected *,{self.resolution},{self.resolution}")
        if flip:
            arr = arr[:, :, ::-1].copy()
        img = torch.from_numpy(arr)  # (7, H, W)
        label = torch.zeros(0, dtype=torch.float32)
        return img, label
