"""Action projection: continuous long-only target weights + cash residual."""

from __future__ import annotations

import numpy as np


def project_target_weights(
    action: np.ndarray,
    *,
    max_weight: float,
) -> np.ndarray:
    """Map a raw action to feasible long-only target weights.

    Rules (v1):
    - each asset weight clipped to ``[0, max_weight]``
    - if the sum of asset weights exceeds 1, scale so the sum equals 1
    - remainder (``1 - sum``) is implicit cash (not returned)
    """
    if max_weight <= 0.0 or max_weight > 1.0:
        raise ValueError(f"max_weight must be in (0, 1]; got {max_weight}")
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    if raw.size == 0:
        raise ValueError("action must contain at least one asset weight")
    if not np.isfinite(raw).all():
        raise ValueError("action contains non-finite values")
    weights = np.clip(raw, 0.0, max_weight)
    total = float(np.sum(weights))
    if total > 1.0:
        weights = weights / total
    return weights
