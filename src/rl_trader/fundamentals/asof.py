"""Point-in-time financial statement selection (look-ahead prevention).

Only statements with ``publication_date <= as_of`` may inform a decision at
``as_of``. Future fiscal filings must not leak into earlier trading dates.
"""

from __future__ import annotations

from datetime import date

from rl_trader.config.models import FundamentalValuationSettings
from rl_trader.fundamentals.dcf import DCFResult, run_dcf
from rl_trader.fundamentals.statements import AnnualFinancials


def select_statements_asof(
    statements: list[AnnualFinancials],
    as_of: date,
) -> list[AnnualFinancials]:
    """Filter to statements published on or before ``as_of``, sorted by period."""
    known = [s for s in statements if s.publication_date <= as_of]
    return sorted(known, key=lambda s: (s.fiscal_period_end, s.publication_date))


def latest_statement_asof(
    statements: list[AnnualFinancials],
    as_of: date,
) -> AnnualFinancials | None:
    """Return the latest known annual statement as of ``as_of``, or None."""
    known = select_statements_asof(statements, as_of)
    return known[-1] if known else None


def run_dcf_asof(
    statements: list[AnnualFinancials],
    settings: FundamentalValuationSettings,
    as_of: date,
) -> DCFResult | None:
    """Run DCF on the latest statement known at ``as_of`` (leakage-safe)."""
    latest = latest_statement_asof(statements, as_of)
    if latest is None:
        return None
    return run_dcf(latest, settings, valuation_date=as_of)
