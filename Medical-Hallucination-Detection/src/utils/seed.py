"""Random seed utilities."""

import os
import random

import numpy as np
import torch


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value to use.
        deterministic: Whether to request deterministic PyTorch behavior where
            supported. This can reduce nondeterminism at the cost of speed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

