"""Device selection and capability probing.

Auto-selects the best available backend: cuda -> DirectML (AMD) -> MPS (Apple) -> cpu.
Because DirectML and MPS have gaps (mixed precision, grid_sample backward, higher-order
autograd for gradient penalties), we *probe* the chosen device and let the training loop
turn features on/off accordingly instead of crashing mid-run.
"""

from dataclasses import dataclass
import torch


@dataclass
class Caps:
    device: object      # torch.device (or DirectML device object)
    kind: str           # 'cuda' | 'dml' | 'mps' | 'cpu'
    name: str           # human-readable device name
    amp: bool           # torch.autocast mixed precision usable
    geometric_aug: bool # continuous grid_sample-based augments usable
    r1: bool            # double-backward (R1 gradient penalty) usable


def _has_directml():
    try:
        import torch_directml  # noqa: F401
        return True
    except Exception:
        return False


def _kind(device):
    t = getattr(device, "type", str(device))
    s = str(device).lower()
    if t == "privateuseone" or "dml" in s or "directml" in s or "privateuse" in s:
        return "dml"
    return t  # 'cuda' | 'mps' | 'cpu'


def pick_device(prefer=None):
    """Return a torch device. `prefer` in {'cuda','dml','mps','cpu'} jumps the queue."""
    order = ([prefer] if prefer else []) + ["cuda", "dml", "mps", "cpu"]
    seen = set()
    for k in order:
        if k in seen:
            continue
        seen.add(k)
        if k == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if k == "dml" and _has_directml():
            import torch_directml
            return torch_directml.device()
        if k == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
        if k == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def _device_name(device, kind):
    try:
        if kind == "cuda":
            return torch.cuda.get_device_name(device)
        if kind == "dml":
            import torch_directml
            try:
                return torch_directml.device_name(torch_directml.default_device())
            except Exception:
                return "DirectML device"
        if kind == "mps":
            return "Apple MPS"
    except Exception:
        pass
    return kind.upper()


def probe(device):
    """Return a Caps describing what the given device can safely do."""
    kind = _kind(device)
    # Mixed precision: only reliable on CUDA here. DirectML/MPS autocast is flaky.
    amp = (kind == "cuda")
    # Continuous geometric augments use grid_sample, whose backward is missing on MPS
    # and unreliable on DirectML. Our AugmentPipe still runs flip/rot90/roll/color there.
    geometric_aug = kind in ("cuda", "cpu")
    # R1 needs a gradient-of-gradient. Double-backward is unsupported on MPS/DirectML.
    r1 = kind in ("cuda", "cpu")
    return Caps(
        device=device,
        kind=kind,
        name=_device_name(device, kind),
        amp=amp,
        geometric_aug=geometric_aug,
        r1=r1,
    )


def describe(caps):
    """Pretty one-block summary for notebook startup."""
    lines = [
        f"Device      : {caps.name}  ({caps.kind})",
        f"Mixed prec. : {'on' if caps.amp else 'off'}",
        f"Geometric aug (grid_sample): {'yes' if caps.geometric_aug else 'no (flip/rot/shift/color only)'}",
        f"R1 penalty (double-backward): {'yes' if caps.r1 else 'no (skipped for stability)'}",
    ]
    return "\n".join(lines)
