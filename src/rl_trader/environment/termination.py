"""Safe episode termination checks."""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from rl_trader.config.models import EnvironmentSettings


class TerminationReason(StrEnum):
    """Why an episode ended (empty string means still running)."""

    NONE = ""
    DATA_EXHAUSTED = "data_exhausted"
    INVALID_PORTFOLIO_VALUE = "invalid_portfolio_value"
    MAX_DRAWDOWN = "max_drawdown_breached"
    NON_FINITE_OBSERVATION = "non_finite_observation"


def check_termination(
    *,
    next_index: int,
    n_bars: int,
    portfolio_value: float,
    drawdown: float,
    observation: np.ndarray,
    settings: EnvironmentSettings,
) -> TerminationReason:
    """Return the first matching termination reason, or ``NONE``."""
    if not np.isfinite(portfolio_value) or portfolio_value < settings.min_portfolio_value:
        return TerminationReason.INVALID_PORTFOLIO_VALUE
    if drawdown >= settings.max_drawdown_terminate:
        return TerminationReason.MAX_DRAWDOWN
    if not np.isfinite(observation).all():
        return TerminationReason.NON_FINITE_OBSERVATION
    if next_index >= n_bars:
        return TerminationReason.DATA_EXHAUSTED
    return TerminationReason.NONE
