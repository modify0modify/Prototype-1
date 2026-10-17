"""Validation helpers (feature match, walk-forward)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rl_trader.validation.features import FeatureMismatchError, assert_features_match
from rl_trader.validation.windows import (
    WalkForwardFold,
    WalkForwardPlan,
    build_walk_forward_plan,
    build_walk_forward_plan_dates,
    slice_fold_episodes,
)

if TYPE_CHECKING:
    from rl_trader.validation.runner import WalkForwardRunResult

__all__ = [
    "FeatureMismatchError",
    "WalkForwardFold",
    "WalkForwardPlan",
    "WalkForwardRunResult",
    "assert_features_match",
    "build_walk_forward_plan",
    "build_walk_forward_plan_dates",
    "run_walk_forward",
    "slice_fold_episodes",
]


def __getattr__(name: str) -> Any:
    """Lazy-load runner symbols to avoid circular imports with agents."""
    if name in {"run_walk_forward", "WalkForwardRunResult"}:
        from rl_trader.validation import runner as _runner

        return getattr(_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
