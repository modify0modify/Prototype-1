"""Compare PPO / SAC / TD3 on identical splits, costs, and rewards.

Candidates are selected using **validation** performance only. Test metrics are
reported once per selected candidate. This module does **not** declare a
winning algorithm from training curves.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_trader.agents.random_control import RandomControlResult, evaluate_random_policy
from rl_trader.agents.registry import SUPPORTED_ALGORITHMS, AlgorithmName
from rl_trader.agents.splits import SplitEpisodes
from rl_trader.agents.trainer import (
    TrainResult,
    cost_assumptions_dict,
    reward_params_dict,
    train_agent,
)
from rl_trader.config.models import (
    EnvironmentSettings,
    FeatureSettings,
    TrainingSettings,
)


@dataclass(frozen=True, slots=True)
class AlgorithmComparisonRow:
    """One algorithm's validation-selected candidate + one-shot test report."""

    algorithm: str
    model_id: str
    output_dir: str
    best_val_mean_reward: float | None
    test_mean_reward: float | None
    model_selected_on: str
    test_used_for_selection: bool
    algorithm_hyperparameters: dict[str, Any]
    total_timesteps_run: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    """Cross-algorithm comparison artifact (no declared winner)."""

    generated_at: str
    algorithms: tuple[str, ...]
    shared_cost_assumptions: dict[str, Any]
    shared_reward_parameters: dict[str, Any]
    rows: tuple[AlgorithmComparisonRow, ...]
    random_control: dict[str, Any]
    selection_rule: str
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "algorithms": list(self.algorithms),
            "shared_cost_assumptions": self.shared_cost_assumptions,
            "shared_reward_parameters": self.shared_reward_parameters,
            "rows": [r.to_dict() for r in self.rows],
            "random_control": self.random_control,
            "selection_rule": self.selection_rule,
            "disclaimer": self.disclaimer,
            "winner_declared": False,
        }


def compare_algorithms(
    splits: SplitEpisodes,
    *,
    training: TrainingSettings,
    environment: EnvironmentSettings,
    features: FeatureSettings | None = None,
    algorithms: Sequence[str] | None = None,
    output_dir: Path | None = None,
    random_episodes: int = 3,
) -> ComparisonReport:
    """Train each algorithm on the same splits; write a comparison report.

    Selection uses validation only. Test is evaluated once per candidate after
    selection. A random-policy control is included. No algorithm is declared
    "best" from training performance.
    """
    algos_list: list[AlgorithmName] = []
    if algorithms is None:
        algos_list.extend(SUPPORTED_ALGORITHMS)
    else:
        for raw in algorithms:
            key = raw.lower().strip()
            if key not in SUPPORTED_ALGORITHMS:
                raise ValueError(f"unsupported algorithm {raw!r}")
            algos_list.append(cast(AlgorithmName, key))
    algos = tuple(algos_list)

    out_root = Path(output_dir or training.comparison_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[AlgorithmComparisonRow] = []
    train_results: list[TrainResult] = []

    for algo in algos:
        model_id = f"{algo}-{training.model_id}"
        algo_out = out_root / "checkpoints" / algo
        algo_training = training.model_copy(
            update={
                "algorithm": algo,
                "model_id": model_id,
                "checkpoint_dir": out_root / "checkpoints",
            }
        )
        result = train_agent(
            splits,
            training=algo_training,
            environment=environment,
            features=features,
            output_dir=algo_out,
            algorithm=algo,
        )
        train_results.append(result)
        rows.append(
            AlgorithmComparisonRow(
                algorithm=algo,
                model_id=result.metadata.model_id,
                output_dir=str(result.output_dir),
                best_val_mean_reward=result.best_val_mean_reward,
                test_mean_reward=result.test_mean_reward,
                model_selected_on=result.manifest.model_selected_on,
                test_used_for_selection=False,
                algorithm_hyperparameters=dict(result.manifest.algorithm_hyperparameters),
                total_timesteps_run=result.manifest.total_timesteps_run,
            )
        )

    random_result: RandomControlResult = evaluate_random_policy(
        splits,
        environment=environment,
        training=training,
        split="test",
        n_episodes=random_episodes,
        seed=training.seed,
    )

    report = ComparisonReport(
        generated_at=datetime.now(UTC).isoformat(),
        algorithms=tuple(str(a) for a in algos),
        shared_cost_assumptions=cost_assumptions_dict(environment),
        shared_reward_parameters=reward_params_dict(environment),
        rows=tuple(rows),
        random_control=random_result.to_dict(),
        selection_rule=(
            "Per-algorithm candidate selected by best validation mean reward "
            "(EvalCallback). Test mean reward reported once after selection. "
            "No cross-algorithm winner is declared from training or validation."
        ),
        disclaimer=(
            "Research comparison only. Not investment advice. "
            "Do not interpret validation ranking as a production recommendation. "
            "Random-policy control checks for env implementation artifacts."
        ),
    )
    _write_report(report, out_root)
    return report


def _write_report(report: ComparisonReport, output_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"agent_comparison_{stamp}.json"
    md_path = output_dir / f"agent_comparison_{stamp}.md"
    payload = report.to_dict()
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    (output_dir / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output_dir / "latest.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, md_path


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Agent algorithm comparison",
        "",
        f"_Generated: {payload['generated_at']}_",
        "",
        f"**Disclaimer:** {payload['disclaimer']}",
        "",
        f"**Selection rule:** {payload['selection_rule']}",
        "",
        f"**Winner declared:** `{payload['winner_declared']}`",
        "",
        "## Shared cost assumptions",
        "",
        "```json",
        json.dumps(payload["shared_cost_assumptions"], indent=2, sort_keys=True),
        "```",
        "",
        "## Shared reward parameters",
        "",
        "```json",
        json.dumps(payload["shared_reward_parameters"], indent=2, sort_keys=True),
        "```",
        "",
        "## Validation-selected candidates (test reported once)",
        "",
        "| algorithm | val_mean_reward | test_mean_reward | timesteps | selected_on |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {algorithm} | {val} | {test} | {steps} | {sel} |".format(
                algorithm=row["algorithm"],
                val=_fmt(row["best_val_mean_reward"]),
                test=_fmt(row["test_mean_reward"]),
                steps=row["total_timesteps_run"],
                sel=row["model_selected_on"],
            )
        )
    lines.extend(
        [
            "",
            "## Random-policy control",
            "",
            "```json",
            json.dumps(payload["random_control"], indent=2, sort_keys=True),
            "```",
            "",
            "No algorithm is declared best from these figures.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
