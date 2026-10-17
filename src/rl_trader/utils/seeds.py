"""Deterministic seed hooks for research reproducibility."""

from __future__ import annotations

import os
import random


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and (when available) PyTorch RNGs."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
