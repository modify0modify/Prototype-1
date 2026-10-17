"""Aggregate walk-forward fold metrics with dispersion and stability flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from rl_trader.config.models import BacktestSettings


@dataclass
class FoldSummaryRow:
    """One row of the fold-summary CSV."""

    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    agent_cumulative_return: float
    agent_sharpe: float
    agent_max_drawdown: float
    agent_turnover: float
    agent_transaction_costs: float
    buy_hold_cumulative_return: float
    underperforms_buy_hold: bool
    exceeds_max_dd_threshold: bool
    unstable: bool
    instability_reasons: str
    scaler_mean_fingerprint: str
    retrain_mode: str
    algorithm: str
    benchmark_best_name: str
    benchmark_best_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateReport:
    """Cross-fold aggregates — means plus dispersion; no live-edge claims."""

    n_folds: int
    mean_agent_return: float
    median_agent_return: float
    std_agent_return: float
    mad_agent_return: float
    mean_agent_sharpe: float
    median_agent_sharpe: float
    std_agent_sharpe: float
    mad_agent_sharpe: float
    mean_agent_max_dd: float
    mean_agent_turnover: float
    mean_agent_costs: float
    n_underperform_buy_hold: int
    n_exceed_max_dd: int
    n_unstable: int
    folds_underperform_buy_hold: tuple[int, ...]
    folds_exceed_max_dd: tuple[int, ...]
    folds_unstable: tuple[int, ...]
    test_periods_non_overlapping: bool
    combined_test_return: float | None
    regime_performance: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def flag_unstable_folds(
    rows: list[FoldSummaryRow],
    settings: BacktestSettings,
) -> list[FoldSummaryRow]:
    """Mark folds with extreme sharpe / return relative to cross-fold MAD."""
    if len(rows) < 2:
        return rows
    sharpes = np.array([r.agent_sharpe for r in rows], dtype=np.float64)
    rets = np.array([r.agent_cumulative_return for r in rows], dtype=np.float64)
    med_s = float(np.median(sharpes))
    med_r = float(np.median(rets))
    mad_s = max(_mad(sharpes), 1e-6)
    mad_r = max(_mad(rets), 1e-6)
    updated: list[FoldSummaryRow] = []
    for r in rows:
        reasons: list[str] = []
        if abs(r.agent_sharpe - med_s) > settings.instability_sharpe_mad_mult * mad_s:
            reasons.append("sharpe_outlier")
        if (
            abs(r.agent_cumulative_return - med_r)
            > settings.instability_return_mad_mult * mad_r
        ):
            reasons.append("return_outlier")
        updated.append(
            FoldSummaryRow(
                **{
                    **r.to_dict(),
                    "unstable": bool(reasons),
                    "instability_reasons": ",".join(reasons),
                }
            )
        )
    return updated


def concatenate_nonoverlapping_returns(
    fold_test_starts: list[int],
    fold_test_ends: list[int],
    fold_returns: list[np.ndarray],
) -> tuple[np.ndarray, bool]:
    """Concatenate daily returns only when test windows do not overlap.

    Returns ``(combined_returns, was_safe)``. If any overlap is detected,
    returns empty array and ``was_safe=False`` — callers must not invent a
    combined equity curve from overlapping OOS periods.
    """
    order = sorted(range(len(fold_test_starts)), key=lambda i: fold_test_starts[i])
    prev_end = -1
    chunks: list[np.ndarray] = []
    for i in order:
        start = fold_test_starts[i]
        end = fold_test_ends[i]
        if start < prev_end:
            return np.array([], dtype=np.float64), False
        chunks.append(np.asarray(fold_returns[i], dtype=np.float64))
        prev_end = end
    if not chunks:
        return np.array([], dtype=np.float64), True
    return np.concatenate(chunks), True


def build_aggregate_report(
    rows: list[FoldSummaryRow],
    *,
    settings: BacktestSettings,
    test_starts: list[int],
    test_ends: list[int],
    fold_returns: list[np.ndarray],
    regime_performance: dict[str, Any],
) -> AggregateReport:
    """Build mean + dispersion summary; refuse overlapping combined returns."""
    rets = np.array([r.agent_cumulative_return for r in rows], dtype=np.float64)
    sharpes = np.array([r.agent_sharpe for r in rows], dtype=np.float64)
    dds = np.array([r.agent_max_drawdown for r in rows], dtype=np.float64)
    turns = np.array([r.agent_turnover for r in rows], dtype=np.float64)
    costs = np.array([r.agent_transaction_costs for r in rows], dtype=np.float64)

    combined, safe = concatenate_nonoverlapping_returns(
        test_starts, test_ends, fold_returns
    )
    combined_ret: float | None
    if safe and combined.size:
        combined_ret = float(np.prod(1.0 + combined) - 1.0)
    else:
        combined_ret = None

    under = tuple(r.fold_id for r in rows if r.underperforms_buy_hold)
    dd_hit = tuple(r.fold_id for r in rows if r.exceeds_max_dd_threshold)
    unstable = tuple(r.fold_id for r in rows if r.unstable)

    notes = (
        "Research walk-forward summary. Dispersion reported alongside means. "
        "No live trading edge is claimed. Combined test return is only computed "
        "when successive test windows are non-overlapping."
    )
    if not safe:
        notes += " WARNING: overlapping test windows — combined return withheld."

    return AggregateReport(
        n_folds=len(rows),
        mean_agent_return=float(np.mean(rets)) if rets.size else 0.0,
        median_agent_return=float(np.median(rets)) if rets.size else 0.0,
        std_agent_return=float(np.std(rets, ddof=0)) if rets.size else 0.0,
        mad_agent_return=_mad(rets),
        mean_agent_sharpe=float(np.mean(sharpes)) if sharpes.size else 0.0,
        median_agent_sharpe=float(np.median(sharpes)) if sharpes.size else 0.0,
        std_agent_sharpe=float(np.std(sharpes, ddof=0)) if sharpes.size else 0.0,
        mad_agent_sharpe=_mad(sharpes),
        mean_agent_max_dd=float(np.mean(dds)) if dds.size else 0.0,
        mean_agent_turnover=float(np.mean(turns)) if turns.size else 0.0,
        mean_agent_costs=float(np.mean(costs)) if costs.size else 0.0,
        n_underperform_buy_hold=len(under),
        n_exceed_max_dd=len(dd_hit),
        n_unstable=len(unstable),
        folds_underperform_buy_hold=under,
        folds_exceed_max_dd=dd_hit,
        folds_unstable=unstable,
        test_periods_non_overlapping=safe,
        combined_test_return=combined_ret,
        regime_performance=regime_performance,
        notes=notes,
    )
