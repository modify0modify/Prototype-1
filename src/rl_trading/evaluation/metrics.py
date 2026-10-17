"""Backtest / OOS evaluation stubs.

Look-ahead bias note
--------------------
Evaluation windows must be strictly out-of-sample relative to training and
feature-fit periods. Do not recompute features with future statistics when
scoring historical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Summary metrics for an evaluation run (stub fields)."""

    total_return: float
    sharpe: float | None
    max_drawdown: float | None
    notes: str = "stub metrics — not computed"


def evaluate_equity_curve(curve: Any) -> EvalResult:
    """Compute performance metrics from an equity curve (stub)."""
    _ = curve
    return EvalResult(
        total_return=0.0,
        sharpe=None,
        max_drawdown=None,
        notes="evaluate_equity_curve stub",
    )
