"""Walk-forward validation tests — synthetic fixtures only (labelled)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_trader.agents.scaling import FeatureScaler
from rl_trader.config.models import BacktestSettings, EnvironmentSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.validation.aggregate import (
    FoldSummaryRow,
    concatenate_nonoverlapping_returns,
    flag_unstable_folds,
)
from rl_trader.validation.runner import run_walk_forward
from rl_trader.validation.windows import (
    WalkForwardFold,
    build_walk_forward_plan,
    slice_fold_episodes,
)


def _synthetic_episode(n_bars: int = 80, seed: int = 7) -> EpisodeData:
    """Clearly labelled synthetic panel for walk-forward unit tests."""
    raw = make_synthetic_episode(
        n_bars=n_bars,
        n_assets=2,
        n_market=2,
        n_fundamental=1,
        n_regime=2,
        seed=seed,
        price_path="drift",
    )
    return EpisodeData(
        timestamps=raw.timestamps,
        prices=raw.prices,
        market_features=raw.market_features,
        fundamental_features=raw.fundamental_features,
        regime_features=raw.regime_features,
        symbols=("SYN_A", "SYN_B"),
    )


def _wf_settings(**overrides: object) -> BacktestSettings:
    base: dict[str, object] = {
        "train_bars": 20,
        "val_bars": 8,
        "test_bars": 8,
        "step_bars": 8,
        "walk_forward_folds": 3,
        "window_mode": "rolling",
        "retrain_mode": "scratch",
        "max_dd_threshold": 0.5,
        "train_agent": False,
        "train_timesteps": 64,
        "algorithm": "ppo",
    }
    base.update(overrides)
    return BacktestSettings(**base)  # type: ignore[arg-type]


def test_no_train_test_leak_in_plan() -> None:
    episode = _synthetic_episode(80)
    plan = build_walk_forward_plan(episode, _wf_settings())
    assert len(plan.folds) >= 2
    for fold in plan.folds:
        fold.assert_no_leak()
        train_idx = set(range(fold.train_start_idx, fold.train_end_idx))
        test_idx = set(range(fold.test_start_idx, fold.test_end_idx))
        assert not (train_idx & test_idx)
        assert fold.train_end_idx <= fold.val_start_idx
        assert fold.val_end_idx <= fold.test_start_idx
    plan.assert_non_overlapping_tests()


def test_step_bars_less_than_test_rejected() -> None:
    with pytest.raises(ValueError, match="step_bars"):
        BacktestSettings(train_bars=20, val_bars=8, test_bars=10, step_bars=5)


def test_per_fold_scaler_independence() -> None:
    episode = _synthetic_episode(90, seed=11)
    plan = build_walk_forward_plan(episode, _wf_settings(walk_forward_folds=3))
    fingerprints: list[str] = []
    for fold in plan.folds:
        train, _val, _test = slice_fold_episodes(episode, fold)
        scaler = FeatureScaler().fit(train, split="train")
        assert scaler.fitted_on == "train"
        fp = tuple(np.round(scaler.mean_, 6))  # type: ignore[arg-type]
        fingerprints.append(str(fp))
        with pytest.raises(ValueError, match="split='train'"):
            FeatureScaler().fit(train, split="validation")  # type: ignore[arg-type]
    # Non-stationary synthetic features → independent fits differ across folds.
    assert len(set(fingerprints)) >= 2


def test_non_overlapping_test_aggregation() -> None:
    safe_rets, safe = concatenate_nonoverlapping_returns(
        [0, 10, 20],
        [10, 20, 30],
        [np.array([0.01, 0.02]), np.array([-0.01]), np.array([0.0, 0.03])],
    )
    assert safe is True
    assert safe_rets.size == 5

    empty, ok = concatenate_nonoverlapping_returns(
        [0, 5],
        [10, 15],
        [np.array([0.1]), np.array([0.2])],
    )
    assert ok is False
    assert empty.size == 0


def test_underperform_and_dd_flags(tmp_path: Path) -> None:
    episode = _synthetic_episode(70, seed=3)
    # Crash path to encourage DD flags on some folds.
    crash = make_synthetic_episode(
        n_bars=70, n_assets=2, n_regime=2, seed=3, price_path="crash"
    )
    episode = EpisodeData(
        timestamps=crash.timestamps,
        prices=crash.prices,
        market_features=crash.market_features,
        fundamental_features=crash.fundamental_features,
        regime_features=crash.regime_features,
        symbols=("SYN_A", "SYN_B"),
    )
    result = run_walk_forward(
        episode,
        backtest=_wf_settings(
            max_dd_threshold=0.01,  # very tight → likely flag
            walk_forward_folds=2,
            train_bars=20,
            val_bars=8,
            test_bars=8,
            step_bars=8,
        ),
        environment=EnvironmentSettings(initial_cash=10_000.0, max_weight=0.5),
        output_dir=tmp_path / "wf",
    )
    assert result.summary_csv.exists()
    assert (result.output_dir / "aggregate_report.md").exists()
    assert (result.output_dir / "model_stability_report.md").exists()
    # Flags are always computed (boolean); crash path usually trips DD threshold.
    assert all(isinstance(r.exceeds_max_dd_threshold, bool) for r in result.fold_rows)
    assert all(isinstance(r.underperforms_buy_hold, bool) for r in result.fold_rows)
    assert any(r.exceeds_max_dd_threshold for r in result.fold_rows)
    for fold in result.plan.folds:
        fold_dir = result.output_dir / "folds" / f"fold_{fold.fold_id:03d}"
        assert (fold_dir / "trade_log.csv").exists()
        assert (fold_dir / "equity_curve.csv").exists()
        assert (fold_dir / "equity_curve.png").exists()
        assert (fold_dir / "drawdown.png").exists()
        assert (fold_dir / "benchmark_comparison.csv").exists()


def test_instability_and_dispersion_flags() -> None:
    rows = [
        FoldSummaryRow(
            fold_id=0,
            train_start="2023-01-01",
            train_end="2023-01-10",
            val_start="2023-01-11",
            val_end="2023-01-15",
            test_start="2023-01-16",
            test_end="2023-01-20",
            agent_cumulative_return=0.01,
            agent_sharpe=0.5,
            agent_max_drawdown=0.05,
            agent_turnover=1.0,
            agent_transaction_costs=10.0,
            buy_hold_cumulative_return=0.02,
            underperforms_buy_hold=True,
            exceeds_max_dd_threshold=False,
            unstable=False,
            instability_reasons="",
            scaler_mean_fingerprint="a",
            retrain_mode="scratch",
            algorithm="equal_weight",
            benchmark_best_name="cash",
            benchmark_best_return=0.0,
        ),
        FoldSummaryRow(
            fold_id=1,
            train_start="2023-01-09",
            train_end="2023-01-18",
            val_start="2023-01-19",
            val_end="2023-01-23",
            test_start="2023-01-24",
            test_end="2023-01-28",
            agent_cumulative_return=0.02,
            agent_sharpe=0.6,
            agent_max_drawdown=0.04,
            agent_turnover=1.1,
            agent_transaction_costs=11.0,
            buy_hold_cumulative_return=0.01,
            underperforms_buy_hold=False,
            exceeds_max_dd_threshold=False,
            unstable=False,
            instability_reasons="",
            scaler_mean_fingerprint="b",
            retrain_mode="scratch",
            algorithm="equal_weight",
            benchmark_best_name="cash",
            benchmark_best_return=0.0,
        ),
        FoldSummaryRow(
            fold_id=2,
            train_start="2023-01-17",
            train_end="2023-01-26",
            val_start="2023-01-27",
            val_end="2023-01-31",
            test_start="2023-02-01",
            test_end="2023-02-05",
            agent_cumulative_return=5.0,  # extreme outlier
            agent_sharpe=50.0,
            agent_max_drawdown=0.03,
            agent_turnover=1.0,
            agent_transaction_costs=9.0,
            buy_hold_cumulative_return=0.01,
            underperforms_buy_hold=False,
            exceeds_max_dd_threshold=False,
            unstable=False,
            instability_reasons="",
            scaler_mean_fingerprint="c",
            retrain_mode="scratch",
            algorithm="equal_weight",
            benchmark_best_name="cash",
            benchmark_best_return=0.0,
        ),
    ]
    flagged = flag_unstable_folds(rows, _wf_settings(instability_sharpe_mad_mult=1.0))
    assert flagged[2].unstable is True
    assert "sharpe_outlier" in flagged[2].instability_reasons or "return_outlier" in (
        flagged[2].instability_reasons
    )


def test_expanding_window_mode() -> None:
    episode = _synthetic_episode(90)
    plan = build_walk_forward_plan(
        episode, _wf_settings(window_mode="expanding", walk_forward_folds=3)
    )
    assert plan.folds[0].train_start_idx == 0
    assert plan.folds[1].train_start_idx == 0
    assert plan.folds[1].train_end_idx > plan.folds[0].train_end_idx


def test_fold_assert_detects_overlap() -> None:
    bad = WalkForwardFold(
        fold_id=0,
        train_start_idx=0,
        train_end_idx=10,
        val_start_idx=5,  # overlaps train
        val_end_idx=12,
        test_start_idx=12,
        test_end_idx=20,
        train_start=__import__("datetime").date(2023, 1, 1),
        train_end=__import__("datetime").date(2023, 1, 10),
        val_start=__import__("datetime").date(2023, 1, 6),
        val_end=__import__("datetime").date(2023, 1, 12),
        test_start=__import__("datetime").date(2023, 1, 13),
        test_end=__import__("datetime").date(2023, 1, 20),
    )
    with pytest.raises(ValueError):
        bad.assert_no_leak()
