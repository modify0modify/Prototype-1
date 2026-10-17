"""Unit: reward calculation, transaction costs, portfolio accounting.

Uses labelled synthetic fixtures only (no network).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from rl_trader.brokers.paper import PaperBroker, PaperBrokerConfig
from rl_trader.brokers.types import OrderRequest, OrderSide, OrderStatus, OrderType, Quote
from rl_trader.config.models import EnvironmentSettings
from rl_trader.environment.costs import estimate_trade_costs
from rl_trader.environment.reward import compute_reward


def _env_settings(**overrides: object) -> EnvironmentSettings:
    base: dict[str, object] = {
        "initial_cash": 100_000.0,
        "max_weight": 0.5,
        "commission_bps": 10.0,
        "bid_ask_spread_bps": 4.0,
        "slippage_bps": 5.0,
        "market_impact_coeff": 0.05,
        "execution_delay_bars": 0,
        "allow_same_bar_execution": True,
        "recent_returns_lookback": 5,
        "drawdown_penalty": 1.0,
        "turnover_penalty": 1.0,
        "transaction_cost_penalty": 1.0,
        "concentration_penalty": 0.5,
        "volatility_penalty": 0.5,
        "costs_already_in_return": False,
        "max_drawdown_terminate": 0.5,
        "min_portfolio_value": 1.0,
    }
    base.update(overrides)
    return EnvironmentSettings(**base)  # type: ignore[arg-type]


def test_reward_breakdown_penalties() -> None:
    """Synthetic: each penalty term reduces total vs raw portfolio return."""
    settings = _env_settings()
    recent = np.array([0.01, -0.02, 0.015, 0.0, 0.005], dtype=np.float64)
    br = compute_reward(
        portfolio_return=0.02,
        drawdown=0.10,
        turnover=0.20,
        transaction_cost_frac=0.001,
        weights=np.array([0.5, 0.5]),
        recent_returns=recent,
        settings=settings,
    )
    assert br.portfolio_return == pytest.approx(0.02)
    assert br.drawdown_penalty > 0.0
    assert br.turnover_penalty > 0.0
    assert br.transaction_cost_penalty > 0.0
    assert br.concentration_penalty > 0.0
    assert br.volatility_penalty > 0.0
    assert br.total < br.portfolio_return


def test_transaction_cost_components() -> None:
    """Synthetic: commission/spread/slippage/impact all positive on trade."""
    settings = _env_settings()
    costs = estimate_trade_costs(
        np.array([10_000.0, -5_000.0]),
        portfolio_value=100_000.0,
        settings=settings,
    )
    assert costs.commission > 0.0
    assert costs.spread > 0.0
    assert costs.slippage > 0.0
    assert costs.market_impact > 0.0
    assert costs.total == pytest.approx(
        costs.commission + costs.spread + costs.slippage + costs.market_impact
    )
    flat = estimate_trade_costs(
        np.zeros(2), portfolio_value=100_000.0, settings=settings
    )
    assert flat.total == 0.0


def test_portfolio_accounting_after_fill(tmp_path: Path) -> None:
    """Synthetic paper fill: cash + mark-to-market equity = consistent NAV."""
    now = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
    broker = PaperBroker(
        PaperBrokerConfig(
            data_dir=tmp_path / "ledger",
            initial_cash=10_000.0,
            commission_bps=10.0,
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
    order = broker.submit_order(
        OrderRequest(
            symbol="AAA",
            side=OrderSide.BUY,
            qty=10.0,
            order_type=OrderType.MARKET,
            idempotency_key="acct-1",
        )
    )
    assert order.status is OrderStatus.FILLED
    account = broker.get_account()
    positions = broker.get_positions()
    assert len(positions) == 1
    pos_value = positions[0].market_value
    assert account.equity == pytest.approx(account.cash + pos_value, rel=1e-6)
    assert account.cash < 10_000.0
    # Weight of position ≈ market_value / equity
    weight = pos_value / account.equity
    assert 0.0 < weight <= 1.0
    recon = broker.reconcile()
    assert recon.get("ok") is True
