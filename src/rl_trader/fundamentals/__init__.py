"""Fundamental analysis and FCFF DCF (research — not investment advice)."""

from rl_trader.fundamentals.asof import (
    latest_statement_asof,
    run_dcf_asof,
    select_statements_asof,
)
from rl_trader.fundamentals.dcf import DCFResult, run_dcf
from rl_trader.fundamentals.fcff import (
    FCFFMethod,
    compute_fcff,
    fcf_ocf_minus_capex,
    fcff_ebit_bridge,
)
from rl_trader.fundamentals.loader import (
    FundamentalSnapshot,
    load_annual_financials_json,
    load_fundamentals,
)
from rl_trader.fundamentals.statements import AnnualFinancials, DataProvenance
from rl_trader.fundamentals.valuation import validate_valuation_assumptions, value_equity

__all__ = [
    "AnnualFinancials",
    "DCFResult",
    "DataProvenance",
    "FCFFMethod",
    "FundamentalSnapshot",
    "compute_fcff",
    "fcf_ocf_minus_capex",
    "fcff_ebit_bridge",
    "latest_statement_asof",
    "load_annual_financials_json",
    "load_fundamentals",
    "run_dcf",
    "run_dcf_asof",
    "select_statements_asof",
    "validate_valuation_assumptions",
    "value_equity",
]
