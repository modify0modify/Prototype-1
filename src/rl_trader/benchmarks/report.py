"""Write benchmark comparison reports (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_trader.benchmarks.harness import StrategyRunResult


def results_to_comparison_table(results: list[StrategyRunResult]) -> list[dict[str, Any]]:
    """Flatten metrics for JSON / Markdown tables."""
    return [r.metrics.to_dict() for r in results]


def write_benchmark_report(
    results: list[StrategyRunResult],
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> tuple[Path, Path]:
    """Persist JSON + Markdown comparison under ``output_dir``.

    Returns ``(json_path, markdown_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    assumptions = results[0].assumptions if results else {}
    payload: dict[str, Any] = {
        "disclaimer": (
            "Deterministic research baselines for RL comparison only. "
            "Not investment advice. Not a recommendation to buy or sell."
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": stamp,
        "shared_assumptions": assumptions,
        "strategies": [r.metrics.strategy for r in results],
        "metrics": results_to_comparison_table(results),
    }
    json_path = output_dir / f"benchmarks_{stamp}.json"
    md_path = output_dir / f"benchmarks_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    # Convenience pointers for latest run.
    (output_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark comparison",
        "",
        f"_Generated: {payload['generated_at']}_",
        "",
        f"**Disclaimer:** {payload['disclaimer']}",
        "",
        "## Shared assumptions",
        "",
        "```json",
        json.dumps(payload.get("shared_assumptions", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Metrics",
        "",
    ]
    metrics: list[dict[str, Any]] = payload.get("metrics", [])
    if not metrics:
        lines.append("_No strategies evaluated._")
        lines.append("")
        return "\n".join(lines)

    columns = [
        "strategy",
        "cumulative_return",
        "annualised_return",
        "annualised_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "turnover",
        "transaction_costs",
        "number_of_trades",
        "win_rate",
        "profit_factor",
        "exposure",
        "worst_day",
        "best_day",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in metrics:
        cells: list[str] = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                cells.append(f"{val:.6g}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("No RL training was run for this report.")
    lines.append("")
    return "\n".join(lines)
