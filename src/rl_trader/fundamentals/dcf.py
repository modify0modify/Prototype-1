"""Discounted cash-flow (FCFF) valuation engine.

**Not investment advice.** Research / feature-engineering utility only.

Discount timing uses an end-of-year mid-year-unadjusted convention:
cash flow in forecast year ``t`` (t=1..N) is discounted by ``(1+WACC)^t``.
Terminal value is computed at horizon N and discounted by ``(1+WACC)^N``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from rl_trader.config.models import FundamentalValuationSettings
from rl_trader.fundamentals.fcff import (
    FCFFMethod,
    compute_fcff,
    resolve_fcff_method,
)
from rl_trader.fundamentals.sensitivity import build_sensitivity_matrix
from rl_trader.fundamentals.statements import AnnualFinancials, DataProvenance


@dataclass(frozen=True, slots=True)
class DCFResult:
    """Full DCF output for one valuation date / financial period."""

    symbol: str
    valuation_date: date
    financial_period_end: date
    publication_date: date
    currency: str
    fcff_method: str
    forecast_cash_flows: tuple[float, ...]
    present_values: tuple[float, ...]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    estimated_value_per_share: float
    diluted_shares_used: float
    sensitivity_matrix: dict[str, Any]
    assumptions: dict[str, Any]
    warnings: tuple[str, ...] = ()
    base_fcff: float = 0.0
    kind: Literal["historical_dcf"] = "historical_dcf"

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        payload = asdict(self)
        payload["valuation_date"] = self.valuation_date.isoformat()
        payload["financial_period_end"] = self.financial_period_end.isoformat()
        payload["publication_date"] = self.publication_date.isoformat()
        return payload


def _forecast_growth_path(
    horizon: int,
    initial_growth: float,
    decline: float,
    terminal_growth: float,
) -> list[float]:
    """Declining growth rates approaching (but not forcing) terminal g."""
    rates: list[float] = []
    g = initial_growth
    for _ in range(horizon):
        rates.append(g)
        g = max(terminal_growth, g - decline)
    return rates


def run_dcf(
    statement: AnnualFinancials,
    settings: FundamentalValuationSettings,
    *,
    valuation_date: date | None = None,
) -> DCFResult:
    """Run an annual FCFF DCF on a single statement.

    Parameters
    ----------
    statement:
        Latest **known** annual financials (caller must enforce publication lag).
    settings:
        WACC, terminal growth, horizon, dilution, etc.
    valuation_date:
        Decision / valuation as-of date (defaults to ``publication_date``).
    """
    warnings: list[str] = []
    if settings.terminal_growth_rate >= settings.wacc:
        raise ValueError(
            "invalid terminal growth: g must be strictly less than WACC "
            f"(g={settings.terminal_growth_rate}, WACC={settings.wacc})"
        )

    method = resolve_fcff_method(statement, settings.fcff_method)
    base_fcff, method_used, method_warnings = compute_fcff(statement, method)
    warnings.extend(method_warnings)

    if statement.reported_free_cash_flow is not None:
        warnings.append(
            "reported_free_cash_flow present; model still uses configured FCFF method "
            f"({method_used.value}) for forecasts"
        )

    growth_path = _forecast_growth_path(
        settings.forecast_horizon_years,
        settings.initial_fcff_growth,
        settings.growth_decline_per_year,
        settings.terminal_growth_rate,
    )

    forecast: list[float] = []
    prev = base_fcff
    for g in growth_path:
        prev = prev * (1.0 + g)
        forecast.append(prev)

    wacc = settings.wacc
    pvs: list[float] = []
    for t, cf in enumerate(forecast, start=1):
        pvs.append(cf / ((1.0 + wacc) ** t))

    fcf_final = forecast[-1]
    terminal_value = fcf_final * (1.0 + settings.terminal_growth_rate) / (
        wacc - settings.terminal_growth_rate
    )
    n = settings.forecast_horizon_years
    pv_terminal = terminal_value / ((1.0 + wacc) ** n)
    enterprise_value = sum(pvs) + pv_terminal

    cash = statement.cash_and_equivalents
    if settings.include_short_term_investments_in_cash:
        cash += statement.short_term_investments
    else:
        warnings.append("short_term_investments excluded from cash bridge")

    equity_value = enterprise_value + cash - statement.debt

    shares = statement.diluted_shares
    if settings.annual_share_dilution > 0.0:
        shares = shares * ((1.0 + settings.annual_share_dilution) ** n)
        warnings.append(
            f"applied annual_share_dilution={settings.annual_share_dilution} over {n} years"
        )
    if shares <= 0.0:
        raise ValueError("diluted share count must be positive")

    per_share = equity_value / shares
    as_of = valuation_date or statement.publication_date

    assumptions: dict[str, Any] = {
        "wacc": wacc,
        "terminal_growth_rate": settings.terminal_growth_rate,
        "forecast_horizon_years": n,
        "initial_fcff_growth": settings.initial_fcff_growth,
        "growth_decline_per_year": settings.growth_decline_per_year,
        "growth_path": growth_path,
        "fcff_method": method_used.value,
        "annual_share_dilution": settings.annual_share_dilution,
        "include_short_term_investments_in_cash": (
            settings.include_short_term_investments_in_cash
        ),
        "base_fcff_provenance": DataProvenance.REPORTED.value,
        "forecast_provenance": DataProvenance.FORECAST.value,
        "currency": statement.currency,
        "disclaimer": "Not investment advice. Research/feature use only.",
    }

    sensitivity = build_sensitivity_matrix(
        forecast_cash_flows=tuple(forecast),
        base_wacc=wacc,
        base_g=settings.terminal_growth_rate,
        wacc_deltas=settings.sensitivity_wacc_deltas,
        g_deltas=settings.sensitivity_g_deltas,
        cash=cash,
        debt=statement.debt,
        shares=shares,
    )

    return DCFResult(
        symbol=statement.symbol,
        valuation_date=as_of,
        financial_period_end=statement.fiscal_period_end,
        publication_date=statement.publication_date,
        currency=statement.currency,
        fcff_method=method_used.value,
        forecast_cash_flows=tuple(forecast),
        present_values=tuple(pvs),
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        estimated_value_per_share=per_share,
        diluted_shares_used=shares,
        sensitivity_matrix=sensitivity,
        assumptions=assumptions,
        warnings=tuple(warnings),
        base_fcff=base_fcff,
    )


# Re-export method enum for callers
__all__ = ["DCFResult", "FCFFMethod", "run_dcf"]
