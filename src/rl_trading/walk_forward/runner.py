"""Walk-forward validation stubs.

Look-ahead bias note
--------------------
Each fold trains only on past windows and evaluates on a future hold-out that
never leaks into feature fitting or hyperparameter selection for that fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """Single walk-forward train/test split."""

    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    """Ordered list of expanding or rolling folds."""

    folds: tuple[WalkForwardFold, ...]


def build_walk_forward_plan(
    start: date,
    end: date,
    *,
    n_folds: int = 3,
) -> WalkForwardPlan:
    """Build a placeholder walk-forward plan (stub geometry)."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if end <= start:
        raise ValueError("end must be after start")
    # Stub: single empty fold container; real logic partitions calendars.
    fold = WalkForwardFold(
        train_start=start,
        train_end=start,
        test_start=start,
        test_end=end,
    )
    return WalkForwardPlan(folds=tuple([fold] * n_folds))
