"""Bridge DCF valuation outputs into publication-lagged feature events."""

from __future__ import annotations

from datetime import date

from rl_trader.config.models import FundamentalValuationSettings
from rl_trader.features.fundamental_features import FundamentalEvent
from rl_trader.fundamentals.asof import run_dcf_asof
from rl_trader.fundamentals.dcf import DCFResult, run_dcf
from rl_trader.fundamentals.fcff import compute_fcff, resolve_fcff_method
from rl_trader.fundamentals.statements import AnnualFinancials


def dcf_result_to_event(result: DCFResult, statement: AnnualFinancials) -> FundamentalEvent:
    """Map a DCF result + statement into a publication-lagged feature event."""
    method = resolve_fcff_method(statement, result.fcff_method)
    fcff, _, _ = compute_fcff(statement, method)
    revenue = statement.revenue
    op_margin = statement.ebit / revenue if revenue else None
    fcf_margin = fcff / revenue if revenue else None
    return FundamentalEvent(
        publication_date=result.publication_date,
        revenue_growth=None,
        fcf_growth=None,
        operating_margin=op_margin,
        fcf_margin=fcf_margin,
        cash=statement.cash_and_equivalents + statement.short_term_investments,
        debt=statement.debt,
        diluted_shares=result.diluted_shares_used,
        estimated_intrinsic_value=result.estimated_value_per_share,
    )


def events_from_statements(
    statements: list[AnnualFinancials],
    settings: FundamentalValuationSettings,
) -> list[FundamentalEvent]:
    """One feature event per statement; invisible before publication_date."""
    events: list[FundamentalEvent] = []
    for statement in sorted(statements, key=lambda s: s.publication_date):
        result = run_dcf(statement, settings, valuation_date=statement.publication_date)
        events.append(dcf_result_to_event(result, statement))
    return events


def intrinsic_value_asof(
    statements: list[AnnualFinancials],
    settings: FundamentalValuationSettings,
    as_of: date,
) -> float | None:
    """Leakage-safe per-share intrinsic value known at ``as_of``."""
    result = run_dcf_asof(statements, settings, as_of)
    return None if result is None else result.estimated_value_per_share
