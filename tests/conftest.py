"""Test compatibility for trusted e3nn constants under PyTorch 2.6+."""

try:
    import torch

    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
except ImportError:
    pass
