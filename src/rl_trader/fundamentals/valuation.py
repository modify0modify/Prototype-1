"""Fundamental valuation config helpers (terminal growth < WACC)."""

from __future__ import annotations

from rl_trader.config.models import FundamentalValuationSettings


def validate_valuation_assumptions(
    wacc: float,
    terminal_growth_rate: float,
) -> FundamentalValuationSettings:
    """Validate valuation inputs; rejects terminal growth ≥ WACC."""
    return FundamentalValuationSettings(wacc=wacc, terminal_growth_rate=terminal_growth_rate)
