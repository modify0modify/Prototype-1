"""Simulated trading cost model (research — no broker)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_trader.config.models import EnvironmentSettings


@dataclass(frozen=True, slots=True)
class TradeCosts:
    """Breakdown of costs charged against cash for one rebalance."""

    commission: float
    spread: float
    slippage: float
    market_impact: float

    @property
    def total(self) -> float:
        """Sum of all cost components."""
        return self.commission + self.spread + self.slippage + self.market_impact


def _bps_to_frac(bps: float) -> float:
    return bps / 10_000.0


def estimate_trade_costs(
    trade_notional: np.ndarray,
    *,
    portfolio_value: float,
    settings: EnvironmentSettings,
) -> TradeCosts:
    """Estimate commissions, half-spread, slippage, and market-impact.

    ``trade_notional`` is signed dollar flow per asset (buy > 0, sell < 0).
    Market impact uses a simple square-root-of-turnover approximation:
    ``impact = coeff * sqrt(sum(|Δ$| / NAV)) * sum(|Δ$|)``.
    """
    abs_notional = np.abs(np.asarray(trade_notional, dtype=np.float64))
    traded = float(np.sum(abs_notional))
    if traded <= 0.0 or portfolio_value <= 0.0:
        return TradeCosts(0.0, 0.0, 0.0, 0.0)

    commission = _bps_to_frac(settings.commission_bps) * traded
    # Full bid-ask spread paid as round-trip half-spread on each side of the trade.
    spread = _bps_to_frac(settings.bid_ask_spread_bps) * 0.5 * traded
    slippage = _bps_to_frac(settings.slippage_bps) * traded
    turnover = traded / portfolio_value
    impact = settings.market_impact_coeff * float(np.sqrt(turnover)) * traded
    return TradeCosts(
        commission=float(commission),
        spread=float(spread),
        slippage=float(slippage),
        market_impact=float(impact),
    )
