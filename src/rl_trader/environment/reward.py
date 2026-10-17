"""Configurable risk-adjusted reward for the portfolio environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_trader.config.models import EnvironmentSettings


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Components of the step reward (for logging / debugging)."""

    portfolio_return: float
    drawdown_penalty: float
    turnover_penalty: float
    transaction_cost_penalty: float
    concentration_penalty: float
    volatility_penalty: float
    total: float


def compute_reward(
    *,
    portfolio_return: float,
    drawdown: float,
    turnover: float,
    transaction_cost_frac: float,
    weights: np.ndarray,
    recent_returns: np.ndarray,
    settings: EnvironmentSettings,
) -> RewardBreakdown:
    """Risk-adjusted reward — uses only information known at the reward time.

    ``reward = r - λ_dd·DD - λ_to·turnover [- λ_tc·cost_frac]
               - λ_conc·concentration - λ_vol·realized_vol``

    When ``settings.costs_already_in_return`` is True (default), ``r`` already
    nets simulated transaction costs, so the explicit ``λ_tc`` term is skipped
    to avoid double-counting (reward-hacking / distorted learning signal).
    """
    w = np.asarray(weights, dtype=np.float64)
    exposure = float(np.sum(w))
    # Herfindahl on invested weights (cash excluded); 0 when flat cash.
    if exposure > 1e-12:
        invested = w / exposure
        concentration = float(np.sum(invested**2))
    else:
        concentration = 0.0

    recent = np.asarray(recent_returns, dtype=np.float64)
    finite = recent[np.isfinite(recent)]
    if finite.size >= 2:
        realized_vol = float(np.std(finite, ddof=0))
    else:
        realized_vol = 0.0

    dd_pen = settings.drawdown_penalty * max(0.0, float(drawdown))
    to_pen = settings.turnover_penalty * max(0.0, float(turnover))
    if settings.costs_already_in_return:
        tc_pen = 0.0
    else:
        tc_pen = settings.transaction_cost_penalty * max(
            0.0, float(transaction_cost_frac)
        )
    conc_pen = settings.concentration_penalty * concentration
    vol_pen = settings.volatility_penalty * realized_vol

    total = (
        float(portfolio_return)
        - dd_pen
        - to_pen
        - tc_pen
        - conc_pen
        - vol_pen
    )
    return RewardBreakdown(
        portfolio_return=float(portfolio_return),
        drawdown_penalty=dd_pen,
        turnover_penalty=to_pen,
        transaction_cost_penalty=tc_pen,
        concentration_penalty=conc_pen,
        volatility_penalty=vol_pen,
        total=float(total),
    )
