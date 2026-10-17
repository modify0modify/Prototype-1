"""Regression guards for leakage, determinism, compatibility, and safety."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    save_checkpoint_metadata,
)
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import OrderRequest, OrderSide, OrderStatus, OrderType, Quote
from rl_trader.config import is_live_trading_blocked
from rl_trader.config.models import RiskSettings
from rl_trader.config.settings import AppSettings
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.features.engineering import FeatureSpec
from rl_trader.features.fundamental_features import FundamentalEvent
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import KillSwitch, RiskDecision, RiskEngine, RiskReviewContext
from rl_trader.validation.features import FeatureMismatchError


def test_no_lookahead_leakage() -> None:
    """Mutating a future price must not change past synthetic features."""
    import pandas as pd

    from rl_trader.config.models import FeatureSettings
    from rl_trader.features.technical import compute_technical_features

    idx = pd.bdate_range("2023-01-03", periods=40, tz="UTC")
    close = pd.Series(np.linspace(100, 140, 40), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": np.full(40, 1e6),
        },
        index=idx,
    )
    settings = FeatureSettings(
        include_regime=False,
        include_portfolio_state=False,
        include_fundamentals=False,
    )
    base = compute_technical_features(ohlcv, settings, benchmark_close=close)
    # Mutate last bar only.
    ohlcv2 = ohlcv.copy()
    ohlcv2.iloc[-1, ohlcv2.columns.get_loc("close")] = 999.0
    ohlcv2.iloc[-1, ohlcv2.columns.get_loc("adj_close")] = 999.0
    mutated = compute_technical_features(
        ohlcv2, settings, benchmark_close=ohlcv2["adj_close"]
    )
    # All rows except the last must match (no look-ahead into the mutated bar).
    for col in base.columns:
        left = base[col].iloc[:-1].to_numpy(dtype=float)
        right = mutated[col].iloc[:-1].to_numpy(dtype=float)
        # Allow NaN equality.
        mask = np.isfinite(left) & np.isfinite(right)
        assert np.allclose(left[mask], right[mask], equal_nan=True)


def test_deterministic_fixture_results() -> None:
    """Same seed → identical synthetic episode prices (labelled synthetic)."""
    a = make_synthetic_episode(n_bars=30, n_assets=2, seed=42)
    b = make_synthetic_episode(n_bars=30, n_assets=2, seed=42)
    assert np.allclose(a.prices, b.prices)
    c = make_synthetic_episode(n_bars=30, n_assets=2, seed=43)
    assert not np.allclose(a.prices, c.prices)


def test_saved_model_compatibility(tmp_path: Path) -> None:
    specs = (
        FeatureSpec(name="returns_1d", lookback=1, version="1"),
        FeatureSpec(name="vol_20d", lookback=20, version="1"),
    )
    save_checkpoint_metadata(
        tmp_path,
        CheckpointMetadata(
            model_id="compat",
            algorithm="stub",
            feature_specs=specs,
            training_period=TrainingPeriod(start=date(2023, 1, 1), end=date(2023, 6, 1)),
            seed=1,
            package_version="0.1.0",
        ),
    )
    loaded = load_checkpoint_for_features(tmp_path, specs)
    assert loaded.model_id == "compat"
    with pytest.raises(FeatureMismatchError):
        load_checkpoint_for_features(
            tmp_path,
            (FeatureSpec(name="returns_1d", lookback=5, version="1"),),
        )


def test_no_duplicate_orders(tmp_path: Path) -> None:
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    broker = PaperBroker(
        PaperBrokerConfig(
            data_dir=tmp_path / "d",
            initial_cash=20_000.0,
            commission_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            rng_seed=1,
        ),
        now_fn=lambda: now,
    )
    broker.connect()
    broker.set_quote(
        Quote(symbol="AAA", bid=99.9, ask=100.1, last=100.0, timestamp=now)
    )
    req = OrderRequest(
        symbol="AAA",
        side=OrderSide.BUY,
        qty=5.0,
        order_type=OrderType.MARKET,
        idempotency_key="dup-guard",
    )
    o1 = broker.submit_order(req)
    o2 = broker.submit_order(req)
    assert o1.order_id == o2.order_id
    assert o1.status is OrderStatus.FILLED
    pos_qty = sum(p.qty for p in broker.get_positions())
    assert pos_qty == pytest.approx(5.0)


def test_no_live_activation_without_approval() -> None:
    settings = AppSettings(_env_file=None)
    assert is_live_trading_blocked(settings) is True


def test_no_order_after_kill_switch() -> None:
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    ks = KillSwitch(reset_phrase="MANUAL_KILL_SWITCH_RESET")
    ks.trip("regression-test")
    eng = RiskEngine(
        settings=RiskSettings(
            max_position_size=0.5,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
            daily_loss_limit=0.05,
            max_drawdown_limit=0.2,
            configuration_version="reg",
        ),
        kill_switch=ks,
    )
    ctx = RiskReviewContext(
        symbols=("AAPL",),
        prices=np.array([100.0]),
        prior_prices=np.array([100.0]),
        price_timestamps=(now - timedelta(seconds=5),),
        now=now,
        portfolio_value=10_000.0,
        cash=10_000.0,
        current_weights=np.array([0.0]),
        day_pnl_frac=0.0,
        week_pnl_frac=0.0,
        peak_nav=10_000.0,
        current_nav=10_000.0,
        realized_volatility=0.1,
        bid_ask_spread_bps=np.array([5.0]),
        sector_by_symbol={"AAPL": "tech"},
        market_is_open=True,
        orders_submitted_today=0,
        last_order_fingerprint=None,
        pending_order_ids=(),
        consecutive_losses=0,
        database_available=True,
        account_confirmed=True,
        trading_mode="paper",
        model_id="ks",
        config_version="reg",
    )
    result = eng.review(TargetPortfolio(weights=np.array([0.2])), context=ctx)
    assert result.decision is RiskDecision.REJECTED
    assert result.portfolio is None


def test_no_fundamentals_before_publication() -> None:
    """Fundamental feature must be NaN on/before publication (next-day lag)."""
    import pandas as pd

    from rl_trader.config.models import FeatureSettings
    from rl_trader.features.pipeline import build_feature_matrix

    idx = pd.bdate_range("2023-01-03", periods=20, tz="UTC")
    close = pd.Series(np.linspace(100, 110, 20), index=idx)
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": np.full(20, 1e6),
        },
        index=idx,
    )
    pub = idx[10].date()
    events = [
        FundamentalEvent(
            publication_date=pub,
            revenue_growth=0.15,
            estimated_intrinsic_value=120.0,
        )
    ]
    settings = FeatureSettings(
        include_regime=False,
        include_portfolio_state=False,
        include_fundamentals=True,
    )
    features, _ = build_feature_matrix(
        ohlcv, settings, symbol="SYN", fundamental_events=events
    )
    before = features.loc[features.index.date <= pub, "revenue_growth"]
    assert before.isna().all()
    after = features.loc[features.index.date > pub, "revenue_growth"]
    assert np.allclose(after.to_numpy(dtype=float), 0.15)


def test_no_scaler_fit_on_val_test() -> None:
    ep = make_synthetic_episode(n_bars=20, n_assets=2, seed=7)
    scaler = FeatureScaler()
    with pytest.raises(ValueError, match="train"):
        scaler.fit(ep, split="val")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="train"):
        scaler.fit(ep, split="test")  # type: ignore[arg-type]
    scaler.fit(ep, split="train")
    assert scaler.fitted_on == "train"
