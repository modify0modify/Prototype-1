"""Walk-forward validation public API (geometry + runner)."""

from __future__ import annotations

from datetime import date

from rl_trader.validation.runner import WalkForwardRunResult, run_walk_forward
from rl_trader.validation.windows import (
    WalkForwardFold,
    WalkForwardPlan,
    build_walk_forward_plan,
    build_walk_forward_plan_dates,
    slice_fold_episodes,
)

__all__ = [
    "WalkForwardFold",
    "WalkForwardPlan",
    "WalkForwardRunResult",
    "build_walk_forward_plan",
    "build_walk_forward_plan_dates",
    "run_walk_forward",
    "slice_fold_episodes",
]


# Keep date-only helper name used by the legacy script.
def build_stub_plan(start: date, end: date, *, n_folds: int = 3) -> WalkForwardPlan:
    """Compatibility wrapper around date-based plan builder."""
    return build_walk_forward_plan_dates(start, end, n_folds=n_folds)
