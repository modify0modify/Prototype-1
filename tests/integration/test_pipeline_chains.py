"""Integration chains: data→features→env→agent→risk→PaperBroker→reconcile."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_trader.agents.base import StubAgent
from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import Quote
from rl_trader.config.models import FeatureSettings, RiskSettings
from rl_trader.data.cleaning import clean_ohlcv
from rl_trader.environment import PortfolioEnv
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.features.pipeline import build_feature_matrix
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import RiskDecision, RiskEngine, RiskReviewContext
from rl_trader.risk.decisions import DecisionRecord, RiskResult


def _synthetic_ohlcv(n: int = 80) -> pd.DataFrame:
    """Labelled synthetic OHLCV — not market data."""
    idx = pd.bdate_range("2023-01-03", periods=n, tz="UTC")
    rng = np.random.default_rng(11)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, size=n))
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
    volume = rng.integers(1_000_000, 2_000_000, size=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "adj_close": close,
        },
        index=idx,
    )


def test_data_to_features_chain() -> None:
    raw = _synthetic_ohlcv()
    cleaned = clean_ohlcv(raw).frame
    assert not cleaned.empty
    settings = FeatureSettings(
        lookback_bars=20,
        causal_only=True,
        include_regime=True,
        include_portfolio_state=False,
        include_fundamentals=False,
    )
    features, specs = build_feature_matrix(cleaned, settings, symbol="SYN")
    assert features.shape[0] == cleaned.shape[0]
    assert len(specs) >= 1
    assert len(features.columns) > 0


def test_features_to_environment_chain() -> None:
    """Synthetic features feed a PortfolioEnv observation without crash."""
    ep = make_synthetic_episode(n_bars=25, n_assets=2, seed=3)
    env = PortfolioEnv(episode=ep)
    obs, info = env.reset(seed=3)
    assert obs.shape[0] > 0
    assert np.isfinite(obs).all()
    action = np.array([0.3, 0.3], dtype=np.float64)
    obs2, reward, terminated, truncated, step_info = env.step(action)
    assert np.isfinite(reward)
    assert obs2.shape == obs.shape
    _ = info, terminated, truncated, step_info


def test_environment_to_agent_chain() -> None:
    ep = make_synthetic_episode(n_bars=20, n_assets=2, seed=5)
    env = PortfolioEnv(episode=ep)
    obs, _ = env.reset(seed=5)
    agent = StubAgent(n_assets=2)
    target = agent.predict_target(obs)
    assert isinstance(target, TargetPortfolio)
    assert target.weights.shape == (2,)
    assert pytest.approx(float(np.sum(target.weights)), abs=1e-6) == 1.0


def test_agent_to_risk_engine_chain() -> None:
    agent = StubAgent(n_assets=2)
    target = agent.predict_target(np.zeros(8, dtype=np.float32))
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    ctx = RiskReviewContext(
        symbols=("A0", "A1"),
        prices=np.array([100.0, 100.0]),
        prior_prices=np.array([100.0, 100.0]),
        price_timestamps=(now - timedelta(seconds=5), now - timedelta(seconds=5)),
        now=now,
        portfolio_value=100_000.0,
        cash=100_000.0,
        current_weights=np.array([0.0, 0.0]),
        day_pnl_frac=0.0,
        week_pnl_frac=0.0,
        peak_nav=100_000.0,
        current_nav=100_000.0,
        realized_volatility=0.1,
        bid_ask_spread_bps=np.array([5.0, 5.0]),
        sector_by_symbol={"A0": "eq", "A1": "eq"},
        market_is_open=True,
        orders_submitted_today=0,
        last_order_fingerprint=None,
        pending_order_ids=(),
        consecutive_losses=0,
        database_available=True,
        account_confirmed=True,
        trading_mode="paper",
        model_id="chain",
        config_version="c1",
    )
    eng = RiskEngine(
        settings=RiskSettings(
            max_position_size=0.6,
            max_portfolio_exposure=1.0,
            max_sector_exposure=1.0,
            min_cash_reserve=0.0,
            max_order_notional=1_000_000.0,
            max_daily_turnover=2.0,
            max_orders_per_day=50,
            daily_loss_limit=0.5,
            max_drawdown_limit=0.9,
            max_volatility_target=1.0,
            max_consecutive_losses=50,
            max_data_age_seconds=300.0,
            abnormal_price_jump=0.5,
            max_spread_bps=500.0,
            configuration_version="c1",
        )
    )
    result = eng.review(target, context=ctx)
    assert result.decision in {RiskDecision.APPROVED, RiskDecision.RESIZED}
    assert result.portfolio is not None


def test_risk_to_paper_broker_chain(tmp_path: Path) -> None:
    now = datetime(2024, 6, 3, 15, 30, tzinfo=UTC)
    broker = PaperBroker(
        PaperBrokerConfig(
            data_dir=tmp_path / "b",
            initial_cash=100_000.0,
            commission_bps=1.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            rng_seed=9,
        ),
        now_fn=lambda: now,
    )
    broker.connect()
    for sym in ("A0", "A1"):
        broker.set_quote(
            Quote(symbol=sym, bid=99.9, ask=100.1, last=100.0, timestamp=now)
        )
    portfolio = TargetPortfolio(weights=np.array([0.4, 0.4]))
    record = DecisionRecord(
        decision_status=RiskDecision.APPROVED,
        original_order=(0.4, 0.4),
        approved_order=(0.4, 0.4),
        violated_rules=(),
        reason="ok",
        timestamp=now,
        configuration_version="c1",
        model_identifier="chain",
    )
    risk_result = RiskResult(
        decision=RiskDecision.APPROVED,
        portfolio=portfolio,
        reason="ok",
        record=record,
        details={"symbols": ("A0", "A1")},
    )
    fill = broker.submit_target(risk_result)
    assert fill.mode == "paper"
    assert broker.get_account().cash < 100_000.0


def test_fills_to_portfolio_reconciliation(tmp_path: Path) -> None:
    now = datetime(2024, 6, 3, 15, 30, tzinfo=UTC)
    broker = PaperBroker(
        PaperBrokerConfig(
            data_dir=tmp_path / "r",
            initial_cash=50_000.0,
            commission_bps=5.0,
            rng_seed=4,
        ),
        now_fn=lambda: now,
    )
    broker.connect()
    broker.set_quote(
        Quote(symbol="AAA", bid=49.9, ask=50.1, last=50.0, timestamp=now)
    )
    from rl_trader.brokers.types import OrderRequest, OrderSide, OrderStatus, OrderType

    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=10.0,
            order_type=OrderType.MARKET,
            idempotency_key="recon-1",
        )
    )
    assert order.status is OrderStatus.FILLED
    recon = broker.reconcile()
    assert recon["ok"] is True
    account = broker.get_account()
    pos = broker.get_positions()
    mtm = sum(p.market_value for p in pos)
    assert account.equity == pytest.approx(account.cash + mtm, rel=1e-5)
