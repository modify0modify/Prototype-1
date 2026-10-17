"""Regression tests for adversarial overfitting / leakage audit fixes.

Fixtures are synthetic and labelled as such. These tests lock the critical and
high-severity corrections from ``reports/audits/overfitting_audit.md``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.config.models import BacktestSettings, EnvironmentSettings, FeatureSettings
from rl_trader.data.validation import assert_survivorship_policy, validate_ohlcv_frame
from rl_trader.environment.reward import compute_reward
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.features.fundamental_features import FundamentalEvent, align_fundamentals_asof
from rl_trader.features.technical import compute_technical_features
from rl_trader.validation.windows import build_walk_forward_plan


def test_same_bar_execution_requires_explicit_allow() -> None:
    with pytest.raises(ValueError, match="allow_same_bar_execution"):
        EnvironmentSettings(execution_delay_bars=0, allow_same_bar_execution=False)
    ok = EnvironmentSettings(execution_delay_bars=0, allow_same_bar_execution=True)
    assert ok.execution_delay_bars == 0
    default = EnvironmentSettings()
    assert default.execution_delay_bars >= 1


def test_reward_does_not_double_count_embedded_costs() -> None:
    settings = EnvironmentSettings(
        costs_already_in_return=True,
        transaction_cost_penalty=1.0,
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
    )
    bd = compute_reward(
        portfolio_return=-0.01,  # already net of costs
        drawdown=0.0,
        turnover=0.1,
        transaction_cost_frac=0.005,
        weights=np.array([0.2, 0.2]),
        recent_returns=np.array([0.0, 0.0]),
        settings=settings,
    )
    assert bd.transaction_cost_penalty == 0.0
    assert bd.total == pytest.approx(-0.01)

    explicit = EnvironmentSettings(
        costs_already_in_return=False,
        transaction_cost_penalty=1.0,
        drawdown_penalty=0.0,
        turnover_penalty=0.0,
        allow_same_bar_execution=True,
        execution_delay_bars=0,
    )
    bd2 = compute_reward(
        portfolio_return=0.0,
        drawdown=0.0,
        turnover=0.0,
        transaction_cost_frac=0.005,
        weights=np.array([0.2, 0.2]),
        recent_returns=np.array([0.0]),
        settings=explicit,
    )
    assert bd2.transaction_cost_penalty == pytest.approx(0.005)


def test_paper_broker_applies_costs_by_default(tmp_path: Path) -> None:
    """C3 fix: PaperBroker is enabled and records cost metadata on fills."""
    from datetime import UTC, datetime

    from rl_trader.brokers.types import OrderRequest, OrderSide, Quote

    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    broker = PaperBroker(
        config=PaperBrokerConfig(
            data_dir=tmp_path / "pb",
            initial_cash=10_000.0,
            commission_bps=10.0,
            spread_bps=2.0,
            slippage_bps=1.0,
        ),
        now_fn=lambda: now,
    )
    assert broker.incomplete is False
    broker.connect()
    broker.set_quote(
        Quote(symbol="SPY", bid=99.9, ask=100.1, last=100.0, timestamp=now)
    )
    order = broker.submit_order(
        OrderRequest(symbol="SPY", side=OrderSide.BUY, qty=1.0)
    )
    assert order.commission > 0.0
    assert broker.get_account().cash < 10_000.0
    broker.disconnect()


def test_fundamentals_not_visible_on_publication_day() -> None:
    pub = date(2023, 2, 15)
    idx = pd.bdate_range("2023-02-10", periods=10, tz="UTC")
    events = [
        FundamentalEvent(
            publication_date=pub,
            revenue_growth=0.1,
            estimated_intrinsic_value=100.0,
        )
    ]
    aligned = align_fundamentals_asof(idx, events)
    on_pub = aligned.loc[aligned.index.date == pub, "revenue_growth"]
    assert on_pub.isna().all()
    after = aligned.loc[aligned.index.date > pub, "revenue_growth"]
    assert after.notna().any()


def test_atr_uses_adjusted_high_low_not_raw_mixed_units() -> None:
    idx = pd.bdate_range("2023-01-03", periods=40, tz="UTC")
    # Simulate a 2-for-1 split mid-window: raw high/low jump, adj stays smooth.
    close_adj = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    high_adj = close_adj + 1.0
    low_adj = close_adj - 1.0
    # Raw unadjusted would be ~2x before split — wrong for ATR with close_adj.
    high_raw = high_adj.copy()
    high_raw.iloc[:20] = high_adj.iloc[:20] * 2.0
    low_raw = low_adj.copy()
    low_raw.iloc[:20] = low_adj.iloc[:20] * 2.0
    ohlcv = pd.DataFrame(
        {
            "open": close_adj,
            "high": high_raw,
            "low": low_raw,
            "close": close_adj * 2.0,  # unadjusted
            "close_adj": close_adj,
            "high_adj": high_adj,
            "low_adj": low_adj,
            "volume": 1_000_000,
        },
        index=idx,
    )
    feats = compute_technical_features(ohlcv, FeatureSettings())
    # ATR should be finite and small (adj ranges ~2), not inflated by raw*2.
    atr = feats["atr"].dropna()
    assert atr.size > 0
    assert float(atr.iloc[-1]) < 5.0


def test_expanding_walk_forward_does_not_reuse_prior_test_as_val() -> None:
    ep = make_synthetic_episode(n_bars=100, n_assets=2, seed=1)
    settings = BacktestSettings(
        train_bars=20,
        val_bars=8,
        test_bars=8,
        step_bars=8,
        walk_forward_folds=3,
        window_mode="expanding",
        train_agent=False,
    )
    plan = build_walk_forward_plan(ep, settings)
    assert len(plan.folds) >= 2
    f0, f1 = plan.folds[0], plan.folds[1]
    # Prior test must not intersect next validation (selection leakage).
    prior_test = set(range(f0.test_start_idx, f0.test_end_idx))
    next_val = set(range(f1.val_start_idx, f1.val_end_idx))
    assert not (prior_test & next_val)
    assert f1.train_end_idx == f0.test_end_idx


def test_survivorship_policy_requires_acknowledgement() -> None:
    with pytest.raises(ValueError, match="survivor"):
        assert_survivorship_policy(
            point_in_time_universe=False,
            survivorship_bias_acknowledged=False,
            require_acknowledgement=True,
        )
    assert_survivorship_policy(
        point_in_time_universe=False,
        survivorship_bias_acknowledged=True,
        require_acknowledgement=True,
    )


def test_delisting_warning_when_series_ends_early() -> None:
    idx = pd.bdate_range("2023-01-03", periods=10, tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000,
        },
        index=idx,
    )
    report = validate_ohlcv_frame(
        frame,
        start=date(2023, 1, 3),
        end=date(2023, 3, 31),
        allow_missing_bars=True,
    )
    assert any("delisting" in w.lower() or "survivor" in w.lower() for w in report.warnings)


def test_train_agent_refuses_silent_synthetic(tmp_path: Path) -> None:
    import runpy
    import sys

    script = Path("scripts/train_agent.py")
    argv = [
        str(script),
        "--config",
        "configs/research.yaml",
        "--timesteps",
        "64",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    old = sys.argv
    try:
        sys.argv = argv
        try:
            runpy.run_path(str(script), run_name="__main__")
            rc = 0
        except SystemExit as exc:
            rc = int(exc.code or 0)
    finally:
        sys.argv = old
    assert rc == 2


def test_audit_document_exists() -> None:
    path = Path("reports/audits/overfitting_audit.md")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "critical" in text.lower()
    assert "Paper trading" in text or "paper trading" in text
