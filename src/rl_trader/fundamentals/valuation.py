"""Fundamental valuation config helpers and public DCF entrypoints."""

from __future__ import annotations

from rl_trader.config.models import FundamentalValuationSettings
from rl_trader.fundamentals.dcf import DCFResult, run_dcf
from rl_trader.fundamentals.statements import AnnualFinancials


def validate_valuation_assumptions(
    wacc: float,
    terminal_growth_rate: float,
) -> FundamentalValuationSettings:
    """Validate valuation inputs; rejects terminal growth ≥ WACC."""
    return FundamentalValuationSettings(wacc=wacc, terminal_growth_rate=terminal_growth_rate)


def value_equity(
    statement: AnnualFinancials,
    settings: FundamentalValuationSettings | None = None,
) -> DCFResult:
    """Convenience wrapper around :func:`run_dcf`."""
    return run_dcf(statement, settings or FundamentalValuationSettings())
