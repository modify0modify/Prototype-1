"""Write walk-forward artifacts under ``reports/walk_forward/``."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from rl_trader.validation.aggregate import AggregateReport, FoldSummaryRow
from rl_trader.validation.policy_eval import PolicyEvalResult


def write_fold_summary_csv(path: Path, rows: Sequence[FoldSummaryRow]) -> Path:
    """Write fold-summary.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0].to_dict().keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return path


def write_trade_log(path: Path, result: PolicyEvalResult) -> Path:
    """Write per-fold trade log CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bar",
        "timestamp",
        "nav_before",
        "nav_after",
        "turnover",
        "cost",
        "period_return",
        "target_weights",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.trade_log:
            d = row.to_dict()
            d["target_weights"] = json.dumps(d["target_weights"])
            writer.writerow(d)
    return path


def write_equity_csv(path: Path, result: PolicyEvalResult) -> Path:
    """Write equity + drawdown series."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "equity", "drawdown"])
        for ts, eq, dd in zip(
            result.timestamps, result.equity_curve, result.drawdowns, strict=False
        ):
            writer.writerow([str(ts)[:32], float(eq), float(dd)])
    return path


def write_equity_drawdown_charts(
    path_equity: Path,
    path_dd: Path,
    *,
    fold_id: int,
    result: PolicyEvalResult,
) -> tuple[Path, Path]:
    """Save equity and drawdown PNG charts for one fold."""
    path_equity.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(result.equity_curve, color="#1f4e5f", lw=1.5)
    ax.set_title(f"Fold {fold_id} equity (synthetic/research)")
    ax.set_xlabel("bar")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_equity, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.fill_between(
        np.arange(result.drawdowns.size),
        result.drawdowns,
        color="#8b3a3a",
        alpha=0.7,
    )
    ax.set_title(f"Fold {fold_id} drawdown")
    ax.set_xlabel("bar")
    ax.set_ylabel("drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_dd, dpi=120)
    plt.close(fig)
    return path_equity, path_dd


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_aggregate_report_md(path: Path, report: AggregateReport) -> Path:
    """Human-readable aggregate report (no winner / live-edge claims)."""
    d = report.to_dict()
    lines = [
        "# Walk-forward aggregate report",
        "",
        f"_Folds: {d['n_folds']}_",
        "",
        f"**Notes:** {d['notes']}",
        "",
        "## Dispersion (not only means)",
        "",
        "| metric | mean | median | std | MAD |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| cumulative_return | {d['mean_agent_return']:.6g} | "
            f"{d['median_agent_return']:.6g} | {d['std_agent_return']:.6g} | "
            f"{d['mad_agent_return']:.6g} |"
        ),
        (
            f"| sharpe | {d['mean_agent_sharpe']:.6g} | "
            f"{d['median_agent_sharpe']:.6g} | {d['std_agent_sharpe']:.6g} | "
            f"{d['mad_agent_sharpe']:.6g} |"
        ),
        "",
        f"- mean max drawdown: `{d['mean_agent_max_dd']:.6g}`",
        f"- mean turnover: `{d['mean_agent_turnover']:.6g}`",
        f"- mean transaction costs: `{d['mean_agent_costs']:.6g}`",
        f"- test periods non-overlapping: `{d['test_periods_non_overlapping']}`",
        f"- combined test return (safe concat only): `{d['combined_test_return']}`",
        "",
        "## Flags",
        "",
        f"- underperform buy-and-hold folds: `{list(d['folds_underperform_buy_hold'])}` "
        f"(n={d['n_underperform_buy_hold']})",
        f"- max DD threshold folds: `{list(d['folds_exceed_max_dd'])}` "
        f"(n={d['n_exceed_max_dd']})",
        f"- unstable folds: `{list(d['folds_unstable'])}` (n={d['n_unstable']})",
        "",
        "## Performance by market regime",
        "",
        "```json",
        json.dumps(d["regime_performance"], indent=2, sort_keys=True),
        "```",
        "",
        "Research artifact only — not investment advice.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_stability_report_md(
    path: Path, rows: Sequence[FoldSummaryRow], report: AggregateReport
) -> Path:
    """Model-stability report highlighting unstable / DD / underperform folds."""
    lines = [
        "# Model stability report",
        "",
        "Folds flagged by MAD-based outliers, buy-and-hold underperformance, "
        "or configured max-drawdown threshold.",
        "",
        f"- unstable: `{list(report.folds_unstable)}`",
        f"- underperform buy-and-hold: `{list(report.folds_underperform_buy_hold)}`",
        f"- exceed max DD: `{list(report.folds_exceed_max_dd)}`",
        "",
        "| fold | return | sharpe | max_dd | unstable | reasons |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.fold_id} | {r.agent_cumulative_return:.6g} | "
            f"{r.agent_sharpe:.6g} | {r.agent_max_drawdown:.6g} | "
            f"{r.unstable} | {r.instability_reasons or '-'} |"
        )
    lines.extend(["", "No production recommendation is implied.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_benchmark_comparison_csv(
    path: Path, fold_id: int, benchmarks: list[dict[str, Any]]
) -> Path:
    """Per-fold benchmark metrics comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not benchmarks:
        path.write_text("fold_id,strategy\n", encoding="utf-8")
        return path
    keys = ["fold_id"] + list(benchmarks[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for b in benchmarks:
            row = {"fold_id": fold_id, **b}
            writer.writerow(row)
    return path
