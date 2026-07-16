"""Adaptive Discriminator Augmentation (ADA), portable subset.

Only differentiable, grid_sample-free transforms are used so this runs on DirectML/MPS
as well as CUDA/CPU. Every transform is applied per-sample with probability `p` and is
written functionally (no in-place ops on graph tensors) so it is safe to apply to the
*generator's output* -- gradients flow back through it to G.

  always available : x-flip, 90-degree rotation, integer translation, brightness,
                     contrast, additive gaussian noise, cutout

`p` is tuned during training by the ADA controller in train.py from the sign of the
discriminator's real-logit output, targeting `ada_target` (default 0.6).
"""

import torch


class AugmentPipe:
    def __init__(self, p=0.0, allow_geometric=True):
        self.p = float(p)
        self.allow_geometric = allow_geometric

    def _fire(self, B, device):
        return torch.rand(B, device=device) < self.p

    def __call__(self, x):
        if self.p <= 0.0:
            return x
        B, C, H, W = x.shape
        dev = x.device

        # --- x-flip ---
        m = self._fire(B, dev).view(B, 1, 1, 1)
        x = torch.where(m, torch.flip(x, dims=[3]), x)

        # --- 90-degree rotation (square images only; H == W) ---
        if self.allow_geometric and H == W:
            k = (torch.rand(B, device=dev) * 4).floor().long()
            fire = self._fire(B, dev)
            for j in (1, 2, 3):
                sel = (fire & (k == j)).view(B, 1, 1, 1)
                x = torch.where(sel, torch.rot90(x, j, dims=[2, 3]), x)

        # --- integer translation (roll), per-sample, built via stack (no in-place) ---
        if self.allow_geometric:
            fire = self._fire(B, dev)
            maxshift = max(1, H // 8)
            rows = []
            for i in range(B):
                if bool(fire[i]):
                    sh = int(torch.randint(-maxshift, maxshift + 1, (1,)).item())
                    sw = int(torch.randint(-maxshift, maxshift + 1, (1,)).item())
                    rows.append(torch.roll(x[i], shifts=(sh, sw), dims=(1, 2)))
                else:
                    rows.append(x[i])
            x = torch.stack(rows, dim=0)

        # --- brightness ---
        m = self._fire(B, dev).view(B, 1, 1, 1)
        b = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.4
        x = x + torch.where(m, b, torch.zeros_like(b))

        # --- contrast ---
        m = self._fire(B, dev).view(B, 1, 1, 1)
        c = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.5
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = torch.where(m, (x - mean) * c + mean, x)

        # --- additive gaussian noise ---
        m = self._fire(B, dev).view(B, 1, 1, 1)
        n = torch.randn_like(x) * 0.1
        x = x + torch.where(m, n, torch.zeros_like(n))

        # --- cutout (single square ~H/2); mask is a fresh non-grad tensor ---
        fire = self._fire(B, dev)
        if bool(fire.any()):
            side = max(1, H // 2)
            mask = torch.ones(B, 1, H, W, device=dev)
            for i in range(B):
                if bool(fire[i]):
                    cy = int(torch.randint(0, H, (1,)).item())
                    cx = int(torch.randint(0, W, (1,)).item())
                    y0, y1 = max(0, cy - side // 2), min(H, cy + side // 2)
                    x0, x1 = max(0, cx - side // 2), min(W, cx + side // 2)
                    mask[i, :, y0:y1, x0:x1] = 0.0
            x = x * mask

        return x.clamp(-1.5, 1.5)
