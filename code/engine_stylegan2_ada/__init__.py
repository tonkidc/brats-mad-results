"""Pure-PyTorch StyleGAN2 + Adaptive Discriminator Augmentation (ADA).

No custom CUDA/HIP kernels: runs on CUDA, DirectML (AMD), MPS (Apple) or CPU.
Public surface used by the BRATS notebook:

    from stylegan2_ada.device import pick_device, probe, describe
    from stylegan2_ada.data.dataset import SliceDataset
    from stylegan2_ada.train import Config, train
    from stylegan2_ada.generate import load_generator, sample, interpolate
    from stylegan2_ada.metrics.fid import compute_fid
"""

__version__ = "0.1.0"

# ROCm/MIOpen: default to FAST find mode (picks kernels from heuristics/find-db
# without searching) — good for 128px, fast startup.
#
# For >=256px, set MIOPEN_FIND_MODE=NORMAL *before importing this package* (e.g. a
# first notebook cell: `import os; os.environ["MIOPEN_FIND_MODE"]="NORMAL"`). FAST's
# immediate mode gives the fast rocBLAS-GEMM conv solvers ZERO workspace, so at 256px
# it falls back to slow solvers (~21 min/kimg here). NORMAL does a real Find(),
# allocates workspace, and picks the fast solvers -> ~2.5 min/kimg (~10x), and also
# lets the R1 double-backward find a working solver instead of the uncompilable im2col
# one. Verified NaN-clean over 40 fp16 steps on gfx1101. Cost: a one-time ~5 min
# search per conv shape, cached to ~/.miopen. train.py auto-keeps R1 on MIOpen when
# NORMAL is set (see r1_cudnn).
#
# Do NOT set MIOPEN_FIND_ENFORCE=SEARCH_DB_UPDATE: it *forces* a full exhaustive
# re-tune per conv shape every run and has produced broken fp16 kernel picks (NaN)
# on RDNA3 Windows. NORMAL is the milder, safe search mode.
# Must be set before the first GPU conv; import-time is early enough.
import os as _os
_os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
if _os.environ.get("MIOPEN_FIND_ENFORCE") == "SEARCH_DB_UPDATE":
    del _os.environ["MIOPEN_FIND_ENFORCE"]  # scrub the old bad default from live shells
