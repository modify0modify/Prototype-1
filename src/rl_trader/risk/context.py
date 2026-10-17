"""Inputs to a deterministic risk review (no RL agent types)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

from rl_trader.portfolio.targets import TargetPortfolio

TradingMode = Literal["research", "paper", "live"]


@dataclass(frozen=True, slots=True)
class RiskReviewContext:
    """Market / account state accompanying a proposed target.

    Fail-closed rules treat missing critical fields as rejection for live mode.
    Research callers should still supply prices and timestamps.
    """

    symbols: tuple[str, ...]
    prices: np.ndarray
    price_timestamps: tuple[datetime, ...]
    now: datetime
    portfolio_value: float
    cash: float
    current_weights: np.ndarray
    day_pnl_frac: float = 0.0
    week_pnl_frac: float = 0.0
    peak_nav: float = 0.0
    current_nav: float = 0.0
    realized_volatility: float = 0.0
    bid_ask_spread_bps: np.ndarray | None = None
    prior_prices: np.ndarray | None = None
    sector_by_symbol: dict[str, str] = field(default_factory=dict)
    market_is_open: bool = True
    orders_submitted_today: int = 0
    last_order_fingerprint: str | None = None
    pending_order_ids: tuple[str, ...] = ()
    consecutive_losses: int = 0
    database_available: bool | None = None
    account_confirmed: bool | None = None
    trading_mode: TradingMode = "research"
    model_id: str = "unspecified"
    config_version: str | None = None

    def __post_init__(self) -> None:
        prices = np.asarray(self.prices, dtype=np.float64).reshape(-1)
        weights = np.asarray(self.current_weights, dtype=np.float64).reshape(-1)
        n = len(self.symbols)
        if prices.shape[0] != n:
            raise ValueError(f"prices length {prices.shape[0]} != n_symbols {n}")
        if weights.shape[0] != n:
            raise ValueError(f"current_weights length {weights.shape[0]} != n_symbols {n}")
        if len(self.price_timestamps) != n:
            raise ValueError("price_timestamps must align with symbols")
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "current_weights", weights)
        if self.bid_ask_spread_bps is not None:
            spreads = np.asarray(self.bid_ask_spread_bps, dtype=np.float64).reshape(-1)
            if spreads.shape[0] != n:
                raise ValueError("bid_ask_spread_bps must align with symbols")
            object.__setattr__(self, "bid_ask_spread_bps", spreads)
        if self.prior_prices is not None:
            prior = np.asarray(self.prior_prices, dtype=np.float64).reshape(-1)
            if prior.shape[0] != n:
                raise ValueError("prior_prices must align with symbols")
            object.__setattr__(self, "prior_prices", prior)
        peak = float(self.peak_nav) if self.peak_nav > 0 else float(self.portfolio_value)
        nav = float(self.current_nav) if self.current_nav > 0 else float(self.portfolio_value)
        object.__setattr__(self, "peak_nav", peak)
        object.__setattr__(self, "current_nav", nav)


@dataclass(frozen=True, slots=True)
class ProposedOrder:
    """Agent-proposed target weights (risk never imports agents)."""

    target: TargetPortfolio
    model_id: str = "unspecified"
