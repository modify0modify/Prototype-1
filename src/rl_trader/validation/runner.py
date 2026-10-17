"""Walk-forward runner: per-fold train/eval, benchmarks, artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_trader.agents.registry import get_algorithm_spec
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import SplitEpisodes
from rl_trader.agents.trainer import train_agent
from rl_trader.benchmarks.harness import (
    resolve_env_settings_for_benchmarks,
    run_all_benchmarks,
)
from rl_trader.benchmarks.panel import PricePanel
from rl_trader.benchmarks.strategies import (
    BuyAndHoldSpyStrategy,
    CashOnlyStrategy,
    EqualWeightBuyAndHoldStrategy,
)
from rl_trader.config.models import (
    BacktestSettings,
    BenchmarkSettings,
    EnvironmentSettings,
    FeatureSettings,
    TrainingSettings,
)
from rl_trader.environment.episode import EpisodeData
from rl_trader.validation.aggregate import (
    FoldSummaryRow,
    build_aggregate_report,
    flag_unstable_folds,
)
from rl_trader.validation.artifacts import (
    write_aggregate_report_md,
    write_benchmark_comparison_csv,
    write_equity_csv,
    write_equity_drawdown_charts,
    write_fold_summary_csv,
    write_json,
    write_stability_report_md,
    write_trade_log,
)
from rl_trader.validation.policy_eval import (
    EqualWeightPolicy,
    PolicyEvalResult,
    WeightPolicy,
    evaluate_policy_on_episode,
)
from rl_trader.validation.regime import (
    fit_regime_bucketizer,
    label_episode_regimes,
    merge_regime_tables,
    regime_performance,
)
from rl_trader.validation.windows import (
    WalkForwardFold,
    WalkForwardPlan,
    build_walk_forward_plan,
    slice_fold_episodes,
)


@dataclass(frozen=True, slots=True)
class WalkForwardRunResult:
    """Top-level walk-forward run artifacts."""

    output_dir: Path
    plan: WalkForwardPlan
    fold_rows: tuple[FoldSummaryRow, ...]
    aggregate_path: Path
    summary_csv: Path


def _scaler_fingerprint(scaler: FeatureScaler) -> str:
    payload = {
        "mean": scaler.mean_.tolist() if scaler.mean_ is not None else [],
        "scale": scaler.scale_.tolist() if scaler.scale_ is not None else [],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _align_training_dates(
    training: TrainingSettings, train: EpisodeData, val: EpisodeData, test: EpisodeData
) -> TrainingSettings:
    def _d(ts: object) -> Any:
        from datetime import date

        if isinstance(ts, date):
            return ts
        return date.fromisoformat(str(ts)[:10])

    return training.model_copy(
        update={
            "train_start": _d(train.timestamps[0]),
            "train_end": _d(train.timestamps[-1]),
            "val_start": _d(val.timestamps[0]),
            "val_end": _d(val.timestamps[-1]),
            "test_start": _d(test.timestamps[0]),
            "test_end": _d(test.timestamps[-1]),
        }
    )


def _load_sb3_policy(directory: Path, algorithm: str, device: str = "cpu") -> WeightPolicy:
    spec = get_algorithm_spec(algorithm)
    model_path = directory / spec.model_filename
    return spec.sb3_cls.load(str(model_path), device=device)  # type: ignore[return-value]


def _run_fold_benchmarks(
    test_episode: EpisodeData,
    env_settings: EnvironmentSettings,
    benchmark_settings: BenchmarkSettings,
) -> list[dict[str, Any]]:
    panel = PricePanel(
        timestamps=test_episode.timestamps,
        prices=test_episode.prices,
        symbols=test_episode.symbols,
    )
    # Compact suite for fold speed + always include buy-and-hold baselines.
    strategies = [
        CashOnlyStrategy(),
        EqualWeightBuyAndHoldStrategy(max_weight=env_settings.max_weight),
    ]
    spy = benchmark_settings.spy_symbol.upper()
    if spy in panel.symbols or panel.symbols:
        # BuyAndHoldSpyStrategy needs SPY in panel; fall back to first symbol name.
        strategies.append(
            BuyAndHoldSpyStrategy(
                spy_symbol=spy if spy in panel.symbols else panel.symbols[0],
                max_weight=env_settings.max_weight,
            )
        )
    results = run_all_benchmarks(
        panel,
        env_settings=env_settings,
        benchmark_settings=benchmark_settings,
        strategies=strategies,
    )
    return [r.metrics.to_dict() for r in results]


def run_walk_forward(
    episode: EpisodeData,
    *,
    backtest: BacktestSettings,
    environment: EnvironmentSettings,
    training: TrainingSettings | None = None,
    features: FeatureSettings | None = None,
    benchmarks: BenchmarkSettings | None = None,
    output_dir: Path | None = None,
) -> WalkForwardRunResult:
    """Execute walk-forward validation and write artifacts.

    Per fold: independent scaler fit on train, optional agent train (scratch or
    fine-tune), validation-selected checkpoint, one-shot test eval, benchmarks
    on the same test panel and cost model.
    """
    plan = build_walk_forward_plan(episode, backtest)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(output_dir or (backtest.walk_forward_dir / f"wf_{stamp}"))
    out.mkdir(parents=True, exist_ok=True)
    folds_dir = out / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    env_settings = resolve_env_settings_for_benchmarks(
        environment=environment,
        benchmarks=benchmarks or BenchmarkSettings(),
        initial_cash=backtest.initial_cash,
    )
    # Sync bps from backtest when provided (still same cost model family).
    env_settings = env_settings.model_copy(
        update={
            "commission_bps": backtest.commission_bps,
            "slippage_bps": backtest.slippage_bps,
            "initial_cash": backtest.initial_cash,
        }
    )
    bench_settings = benchmarks or BenchmarkSettings()
    base_training = training or TrainingSettings()
    feature_settings = features or FeatureSettings()

    rows: list[FoldSummaryRow] = []
    fold_returns: list[np.ndarray] = []
    test_starts: list[int] = []
    test_ends: list[int] = []
    regime_tables: list[dict[str, dict[str, float]]] = []
    prev_checkpoint: Path | None = None
    scaler_fingerprints: list[str] = []

    for fold in plan.folds:
        fold_result = _run_single_fold(
            episode=episode,
            fold=fold,
            backtest=backtest,
            env_settings=env_settings,
            base_training=base_training,
            feature_settings=feature_settings,
            bench_settings=bench_settings,
            folds_dir=folds_dir,
            prev_checkpoint=prev_checkpoint,
        )
        rows.append(fold_result["row"])
        fold_returns.append(fold_result["daily_returns"])
        test_starts.append(fold.test_start_idx)
        test_ends.append(fold.test_end_idx)
        regime_tables.append(fold_result["regime_table"])
        scaler_fingerprints.append(fold_result["scaler_fp"])
        if fold_result["checkpoint"] is not None:
            prev_checkpoint = fold_result["checkpoint"]

    # Independent scalers should not all share an identical fingerprint on
    # non-stationary synthetic data — recorded for stability report.
    rows = flag_unstable_folds(rows, backtest)
    regime_merged = merge_regime_tables(regime_tables)
    aggregate = build_aggregate_report(
        rows,
        settings=backtest,
        test_starts=test_starts,
        test_ends=test_ends,
        fold_returns=fold_returns,
        regime_performance=regime_merged,
    )

    summary_csv = write_fold_summary_csv(out / "fold_summary.csv", rows)
    write_json(out / "aggregate_report.json", aggregate.to_dict())
    agg_md = write_aggregate_report_md(out / "aggregate_report.md", aggregate)
    write_stability_report_md(out / "model_stability_report.md", rows, aggregate)
    write_json(
        out / "model_stability_report.json",
        {
            "folds_unstable": list(aggregate.folds_unstable),
            "folds_underperform_buy_hold": list(aggregate.folds_underperform_buy_hold),
            "folds_exceed_max_dd": list(aggregate.folds_exceed_max_dd),
            "scaler_fingerprints": scaler_fingerprints,
            "scaler_all_identical": (
                len(set(scaler_fingerprints)) <= 1 and len(scaler_fingerprints) > 1
            ),
            "window_config": {
                "window_mode": plan.window_mode,
                "train_bars": plan.train_bars,
                "val_bars": plan.val_bars,
                "test_bars": plan.test_bars,
                "step_bars": plan.step_bars,
                "retrain_mode": backtest.retrain_mode,
                "max_dd_threshold": backtest.max_dd_threshold,
            },
        },
    )
    write_json(
        out / "plan.json",
        {
            "window_mode": plan.window_mode,
            "train_bars": plan.train_bars,
            "val_bars": plan.val_bars,
            "test_bars": plan.test_bars,
            "step_bars": plan.step_bars,
            "n_bars": plan.n_bars,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train": [f.train_start_idx, f.train_end_idx],
                    "val": [f.val_start_idx, f.val_end_idx],
                    "test": [f.test_start_idx, f.test_end_idx],
                    "train_dates": [f.train_start.isoformat(), f.train_end.isoformat()],
                    "val_dates": [f.val_start.isoformat(), f.val_end.isoformat()],
                    "test_dates": [f.test_start.isoformat(), f.test_end.isoformat()],
                }
                for f in plan.folds
            ],
        },
    )
    # Convenience copies
    (out / "latest_pointer.txt").write_text(str(out.resolve()) + "\n", encoding="utf-8")
    return WalkForwardRunResult(
        output_dir=out,
        plan=plan,
        fold_rows=tuple(rows),
        aggregate_path=agg_md,
        summary_csv=summary_csv,
    )


def _run_single_fold(
    *,
    episode: EpisodeData,
    fold: WalkForwardFold,
    backtest: BacktestSettings,
    env_settings: EnvironmentSettings,
    base_training: TrainingSettings,
    feature_settings: FeatureSettings,
    bench_settings: BenchmarkSettings,
    folds_dir: Path,
    prev_checkpoint: Path | None,
) -> dict[str, Any]:
    fold.assert_no_leak()
    train_ep, val_ep, test_ep = slice_fold_episodes(episode, fold)
    fold_dir = folds_dir / f"fold_{fold.fold_id:03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # --- Scaler: TRAIN ONLY, independent per fold ---
    scaler = FeatureScaler()
    train_scaled = scaler.fit_transform(train_ep, split="train")
    val_scaled = scaler.transform(val_ep)
    test_scaled = scaler.transform(test_ep)
    scaler.save(fold_dir)
    scaler_fp = _scaler_fingerprint(scaler)

    bucketizer = fit_regime_bucketizer(
        train_ep,
        feature_index=backtest.regime_feature_index,
        n_buckets=backtest.n_regime_buckets,
    )
    test_regime = label_episode_regimes(test_ep, bucketizer)
    # Labels align to decision bars; returns use bars [0..T-2] roughly T-1.
    regime_for_returns = test_regime[:-1] if test_regime.size > 1 else test_regime

    checkpoint: Path | None = None
    policy: WeightPolicy
    if backtest.train_agent:
        splits = SplitEpisodes(
            train=train_scaled,
            validation=val_scaled,
            test=test_scaled,
            train_indices=tuple(range(train_scaled.n_bars)),
            val_indices=tuple(range(val_scaled.n_bars)),
            test_indices=tuple(range(test_scaled.n_bars)),
        )
        training = _align_training_dates(
            base_training.model_copy(
                update={
                    "algorithm": backtest.algorithm,
                    "model_id": f"wf-fold-{fold.fold_id}",
                    "total_timesteps": backtest.train_timesteps,
                    "n_steps": min(64, max(16, backtest.train_timesteps // 2)),
                    "batch_size": 32,
                    "n_epochs": 1,
                    "buffer_size": max(1_000, backtest.train_timesteps),
                    "learning_starts": min(64, backtest.train_timesteps // 2),
                    "eval_freq": max(16, backtest.train_timesteps // 2),
                    "early_stopping_patience": 2,
                    "min_evals_before_stop": 1,
                    "evaluate_test_after_training": False,
                    "normalize_features": False,  # already scaled externally
                    "seed": base_training.seed + fold.fold_id,
                }
            ),
            train_scaled,
            val_scaled,
            test_scaled,
        )
        ckpt_dir = fold_dir / "checkpoint"
        spec = get_algorithm_spec(backtest.algorithm)
        prev_weights = (
            prev_checkpoint / spec.model_filename
            if prev_checkpoint is not None
            else None
        )
        if (
            backtest.retrain_mode == "fine_tune"
            and prev_weights is not None
            and prev_weights.exists()
        ):
            from rl_trader.agents.trainer import make_vec_env

            ckpt_dir.mkdir(parents=True, exist_ok=True)
            train_env = make_vec_env(train_scaled, env_settings, seed=training.seed)
            model = spec.sb3_cls.load(str(prev_weights), device=training.device)
            model.set_env(train_env)
            model.learn(total_timesteps=backtest.train_timesteps, progress_bar=False)
            model.save(str((ckpt_dir / spec.model_filename).with_suffix("")))
            train_env.close()
            scaler.save(ckpt_dir)
            checkpoint = ckpt_dir
        else:
            result = train_agent(
                splits,
                training=training,
                environment=env_settings,
                features=feature_settings,
                output_dir=ckpt_dir,
                algorithm=backtest.algorithm,
            )
            checkpoint = result.output_dir
        policy = _load_sb3_policy(ckpt_dir, backtest.algorithm, device=training.device)
    else:
        policy = EqualWeightPolicy(test_scaled.n_assets, env_settings.max_weight)

    agent_eval: PolicyEvalResult = evaluate_policy_on_episode(
        policy,
        test_scaled,
        env_settings,
        trading_days_per_year=backtest.trading_days_per_year,
        strategy_name=f"agent_{backtest.algorithm}",
        seed=base_training.seed + 1000 + fold.fold_id,
        regime_labels=regime_for_returns,
    )

    bench_metrics = _run_fold_benchmarks(test_ep, env_settings, bench_settings)
    write_benchmark_comparison_csv(
        fold_dir / "benchmark_comparison.csv", fold.fold_id, bench_metrics
    )
    write_json(fold_dir / "benchmarks.json", {"benchmarks": bench_metrics})
    write_json(fold_dir / "agent_metrics.json", agent_eval.metrics.to_dict())
    write_trade_log(fold_dir / "trade_log.csv", agent_eval)
    write_equity_csv(fold_dir / "equity_curve.csv", agent_eval)
    write_equity_drawdown_charts(
        fold_dir / "equity_curve.png",
        fold_dir / "drawdown.png",
        fold_id=fold.fold_id,
        result=agent_eval,
    )

    def _metric_float(row: dict[str, Any], key: str, default: float) -> float:
        raw: Any = row.get(key, default)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                return default
        return default

    bh = next(
        (
            b
            for b in bench_metrics
            if b.get("strategy") == "equal_weight_buy_and_hold"
        ),
        bench_metrics[0] if bench_metrics else {"cumulative_return": 0.0, "strategy": "none"},
    )
    best_bench = max(
        bench_metrics,
        key=lambda b: _metric_float(b, "cumulative_return", -1e9),
    )
    under = (
        agent_eval.metrics.cumulative_return
        < _metric_float(bh, "cumulative_return", 0.0) - backtest.underperform_eps
    )
    exceeds_dd = agent_eval.metrics.max_drawdown > backtest.max_dd_threshold
    regime_table = regime_performance(
        agent_eval.daily_returns, agent_eval.regime_labels
    )
    write_json(fold_dir / "regime_performance.json", regime_table)

    row = FoldSummaryRow(
        fold_id=fold.fold_id,
        train_start=fold.train_start.isoformat(),
        train_end=fold.train_end.isoformat(),
        val_start=fold.val_start.isoformat(),
        val_end=fold.val_end.isoformat(),
        test_start=fold.test_start.isoformat(),
        test_end=fold.test_end.isoformat(),
        agent_cumulative_return=agent_eval.metrics.cumulative_return,
        agent_sharpe=agent_eval.metrics.sharpe,
        agent_max_drawdown=agent_eval.metrics.max_drawdown,
        agent_turnover=agent_eval.metrics.turnover,
        agent_transaction_costs=agent_eval.metrics.transaction_costs,
        buy_hold_cumulative_return=_metric_float(bh, "cumulative_return", 0.0),
        underperforms_buy_hold=under,
        exceeds_max_dd_threshold=exceeds_dd,
        unstable=False,
        instability_reasons="",
        scaler_mean_fingerprint=scaler_fp,
        retrain_mode=backtest.retrain_mode,
        algorithm=backtest.algorithm if backtest.train_agent else "equal_weight",
        benchmark_best_name=str(best_bench.get("strategy", "")),
        benchmark_best_return=_metric_float(best_bench, "cumulative_return", 0.0),
    )
    write_json(fold_dir / "fold_summary.json", row.to_dict())
    return {
        "row": row,
        "daily_returns": agent_eval.daily_returns,
        "regime_table": regime_table,
        "scaler_fp": scaler_fp,
        "checkpoint": checkpoint,
    }
