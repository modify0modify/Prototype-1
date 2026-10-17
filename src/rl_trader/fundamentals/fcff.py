"""Free cash flow to firm (FCFF) calculators.

Supported definitions
---------------------
1. **EBIT bridge:** ``FCFF = EBIT × (1 - tax) + D&A - CapEx - ΔNWC``
2. **OCF − CapEx:** ``FCF = OCF - CapEx`` when operating cash flow is reliable

CapEx is treated as a cash *outflow*. If callers store CapEx as a positive
spend amount, it is subtracted. ΔNWC may be positive (use of cash) or
negative (source of cash).
"""

from __future__ import annotations

from enum import StrEnum

from rl_trader.fundamentals.statements import AnnualFinancials, DataProvenance


class FCFFMethod(StrEnum):
    """FCFF calculation method."""

    EBIT_BRIDGE = "ebit_bridge"
    OCF_CAPEX = "ocf_capex"


def fcff_ebit_bridge(
    ebit: float,
    tax_rate: float,
    depreciation_amortization: float,
    capex: float,
    change_in_nwc: float,
) -> float:
    """``EBIT*(1-t) + D&A - CapEx - ΔNWC``."""
    if not 0.0 <= tax_rate <= 1.0:
        raise ValueError(f"tax_rate must be in [0, 1], got {tax_rate}")
    nopat = ebit * (1.0 - tax_rate)
    return nopat + depreciation_amortization - capex - change_in_nwc


def fcf_ocf_minus_capex(operating_cash_flow: float, capex: float) -> float:
    """``OCF - CapEx``."""
    return operating_cash_flow - capex


def compute_fcff(
    statement: AnnualFinancials,
    method: FCFFMethod | str = FCFFMethod.EBIT_BRIDGE,
) -> tuple[float, FCFFMethod, list[str]]:
    """Compute FCFF for a statement; returns (value, method_used, warnings)."""
    warnings: list[str] = []
    method_enum = FCFFMethod(method)

    if method_enum is FCFFMethod.EBIT_BRIDGE:
        value = fcff_ebit_bridge(
            statement.ebit,
            statement.effective_tax_rate,
            statement.depreciation_amortization,
            statement.capex,
            statement.change_in_nwc,
        )
        return value, method_enum, warnings

    if method_enum is FCFFMethod.OCF_CAPEX:
        value = fcf_ocf_minus_capex(statement.operating_cash_flow, statement.capex)
        if statement.reported_free_cash_flow is not None:
            gap = abs(value - statement.reported_free_cash_flow)
            if gap > max(1.0, 0.05 * abs(statement.reported_free_cash_flow)):
                warnings.append(
                    "ocf_capex differs from reported_free_cash_flow by "
                    f"{gap:.4g}; using ocf_capex definition"
                )
        return value, method_enum, warnings

    raise ValueError(f"unsupported FCFF method: {method}")


def resolve_fcff_method(
    statement: AnnualFinancials,
    configured: str,
) -> FCFFMethod:
    """Choose method for ``auto``: prefer OCF path when OCF is reported."""
    if configured == "ebit_bridge":
        return FCFFMethod.EBIT_BRIDGE
    if configured == "ocf_capex":
        return FCFFMethod.OCF_CAPEX
    if configured == "auto":
        ocf_prov = statement.field_provenance.get(
            "operating_cash_flow", DataProvenance.REPORTED
        )
        if ocf_prov is DataProvenance.REPORTED:
            return FCFFMethod.OCF_CAPEX
        return FCFFMethod.EBIT_BRIDGE
    raise ValueError(f"unknown fcff_method config: {configured!r}")
