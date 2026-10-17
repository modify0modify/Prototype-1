"""Target portfolio types shared across research and risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Model-proposed portfolio weights (pre-risk, pre-execution)."""

    weights: np.ndarray
    as_of: str | None = None
