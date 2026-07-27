"""Device selection utilities."""

import torch


def get_device() -> torch.device:
    """Return the best available compute device.

    Device priority is CUDA, then Apple Metal Performance Shaders, then CPU.

    Returns:
        The detected PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

