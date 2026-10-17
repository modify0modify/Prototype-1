"""In-environment trade simulation (no broker adapters)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_trader.config.models import EnvironmentSettings
from rl_trader.environment.costs import TradeCosts, estimate_trade_costs


@dataclass(frozen=True, slots=True)
class SimulatedFill:
    """Result of converting desired weights into share trades."""

    pre_trade_shares: np.ndarray
    requested_weights: np.ndarray
    approved_weights: np.ndarray
    approved_shares: np.ndarray
    trades_shares: np.ndarray
    trade_notional: np.ndarray
    costs: TradeCosts
    cash_after: float
    rejected: bool
    reject_reason: str | None = None


def simulate_rebalance(
    *,
    cash: float,
    shares: np.ndarray,
    prices: np.ndarray,
    target_weights: np.ndarray,
    settings: EnvironmentSettings,
) -> SimulatedFill:
    """Rebalance toward ``target_weights`` at ``prices`` with cash constraints.

    Long + cash only. Buys that would overdraw cash (after estimated costs) are
    scaled down proportionately. Fully rejected trades leave the portfolio
    unchanged when prices are invalid or NAV is non-positive.
    """
    shares = np.asarray(shares, dtype=np.float64).copy()
    prices = np.asarray(prices, dtype=np.float64)
    target_weights = np.asarray(target_weights, dtype=np.float64)
    pre_trade_shares = shares.copy()

    if (
        not np.isfinite(prices).all()
        or np.any(prices <= 0.0)
        or not np.isfinite(target_weights).all()
        or not np.isfinite(cash)
    ):
        return SimulatedFill(
            pre_trade_shares=pre_trade_shares,
            requested_weights=target_weights,
            approved_weights=_current_weights(cash, shares, prices),
            approved_shares=shares,
            trades_shares=np.zeros_like(shares),
            trade_notional=np.zeros_like(shares),
            costs=TradeCosts(0.0, 0.0, 0.0, 0.0),
            cash_after=cash,
            rejected=True,
            reject_reason="invalid_prices_or_cash",
        )

    holdings_value = float(np.dot(shares, prices))
    nav = cash + holdings_value
    if not np.isfinite(nav) or nav <= 0.0:
        return SimulatedFill(
            pre_trade_shares=pre_trade_shares,
            requested_weights=target_weights,
            approved_weights=np.zeros_like(target_weights),
            approved_shares=shares,
            trades_shares=np.zeros_like(shares),
            trade_notional=np.zeros_like(shares),
            costs=TradeCosts(0.0, 0.0, 0.0, 0.0),
            cash_after=cash,
            rejected=True,
            reject_reason="non_positive_nav",
        )

    # Cap single-name weight again for safety (also enforced in action projection).
    capped = np.clip(target_weights, 0.0, settings.max_weight)
    if float(np.sum(capped)) > 1.0:
        capped = capped / float(np.sum(capped))

    desired_value = capped * nav
    current_value = shares * prices
    trade_value = desired_value - current_value

    # First-pass cost estimate on unconstrained trade notionals.
    costs = estimate_trade_costs(
        trade_value, portfolio_value=nav, settings=settings
    )
    buy_mask = trade_value > 0.0
    sell_mask = trade_value < 0.0
    buy_notional = float(np.sum(trade_value[buy_mask])) if np.any(buy_mask) else 0.0
    sell_notional = (
        float(-np.sum(trade_value[sell_mask])) if np.any(sell_mask) else 0.0
    )
    cash_if_unscaled = cash + sell_notional - buy_notional - costs.total

    rejected = False
    reject_reason: str | None = None
    if cash_if_unscaled < -1e-9 and buy_notional > 0.0:
        # Scale buys so cash remains non-negative after costs (approx).
        # Available for buys ≈ cash + sell proceeds - costs_estimate.
        available = cash + sell_notional - costs.total
        if available <= 0.0:
            trade_value = trade_value.copy()
            trade_value[buy_mask] = 0.0
            rejected = True
            reject_reason = "insufficient_cash_for_buys"
        else:
            scale = min(1.0, available / buy_notional)
            trade_value = trade_value.copy()
            trade_value[buy_mask] = trade_value[buy_mask] * scale
            if scale < 1.0 - 1e-12:
                rejected = True
                reject_reason = "buys_scaled_for_cash"

    # Recompute costs on the (possibly scaled) trades.
    costs = estimate_trade_costs(
        trade_value, portfolio_value=nav, settings=settings
    )
    trade_shares = np.zeros_like(shares)
    nonzero = prices > 0.0
    trade_shares[nonzero] = trade_value[nonzero] / prices[nonzero]

    new_shares = shares + trade_shares
    # Enforce non-negative holdings (long-only).
    if np.any(new_shares < -1e-9):
        new_shares = np.maximum(new_shares, 0.0)
        trade_shares = new_shares - shares
        trade_value = trade_shares * prices
        costs = estimate_trade_costs(
            trade_value, portfolio_value=nav, settings=settings
        )
        rejected = True
        reject_reason = reject_reason or "short_sale_blocked"

    cash_delta = -float(np.sum(trade_value)) - costs.total
    new_cash = cash + cash_delta
    if new_cash < -1e-8:
        # Last-resort: abort trades.
        return SimulatedFill(
            pre_trade_shares=pre_trade_shares,
            requested_weights=target_weights,
            approved_weights=_current_weights(cash, shares, prices),
            approved_shares=shares,
            trades_shares=np.zeros_like(shares),
            trade_notional=np.zeros_like(shares),
            costs=TradeCosts(0.0, 0.0, 0.0, 0.0),
            cash_after=cash,
            rejected=True,
            reject_reason="cash_constraint_abort",
        )
    new_cash = max(0.0, new_cash)

    approved_weights = _current_weights(new_cash, new_shares, prices)
    return SimulatedFill(
        pre_trade_shares=pre_trade_shares,
        requested_weights=target_weights,
        approved_weights=approved_weights,
        approved_shares=new_shares,
        trades_shares=trade_shares,
        trade_notional=trade_value,
        costs=costs,
        cash_after=new_cash,
        rejected=rejected,
        reject_reason=reject_reason,
    )


def _current_weights(cash: float, shares: np.ndarray, prices: np.ndarray) -> np.ndarray:
    nav = cash + float(np.dot(shares, prices))
    if nav <= 0.0 or not np.isfinite(nav):
        return np.zeros_like(shares, dtype=np.float64)
    weights = np.asarray((shares * prices) / nav, dtype=np.float64)
    return weights


def portfolio_nav(cash: float, shares: np.ndarray, prices: np.ndarray) -> float:
    """Mark-to-market portfolio value."""
    return float(cash + np.dot(shares, prices))
