"""Annual aggregate financial statements (reported vs assumption tags).

Disclaimer
----------
This module is for research / feature engineering only. Outputs are **not
investment advice** and must not be treated as recommendations to buy or sell.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataProvenance(StrEnum):
    """Origin of a financial field value."""

    REPORTED = "reported"
    ASSUMPTION = "assumption"
    FORECAST = "forecast"


class AnnualFinancials(BaseModel):
    """One fiscal year of aggregate financials.

    ``publication_date`` is when the statement becomes known for trading
    decisions. ``fiscal_period_end`` is the reporting period end. Leakage
    control uses ``publication_date`` only.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    fiscal_period_end: date
    publication_date: date
    currency: str = "USD"

    revenue: Annotated[float, Field(allow_inf_nan=False)]
    ebit: Annotated[float, Field(allow_inf_nan=False)]
    effective_tax_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    depreciation_amortization: Annotated[float, Field(allow_inf_nan=False)]
    capex: Annotated[float, Field(allow_inf_nan=False)]
    change_in_nwc: Annotated[float, Field(allow_inf_nan=False)]
    operating_cash_flow: Annotated[float, Field(allow_inf_nan=False)]
    reported_free_cash_flow: Annotated[float, Field(allow_inf_nan=False)] | None = None
    cash_and_equivalents: Annotated[float, Field(allow_inf_nan=False)]
    short_term_investments: Annotated[float, Field(allow_inf_nan=False)] = 0.0
    debt: Annotated[float, Field(ge=0.0)]
    diluted_shares: Annotated[float, Field(gt=0.0)]

    # Per-field provenance (defaults to reported for required inputs)
    field_provenance: dict[str, DataProvenance] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _defaults_and_checks(self) -> AnnualFinancials:
        if self.publication_date < self.fiscal_period_end:
            # Allowed (early releases) but warn via empty provenance only — keep data
            pass
        defaults = {
            "revenue": DataProvenance.REPORTED,
            "ebit": DataProvenance.REPORTED,
            "effective_tax_rate": DataProvenance.REPORTED,
            "depreciation_amortization": DataProvenance.REPORTED,
            "capex": DataProvenance.REPORTED,
            "change_in_nwc": DataProvenance.REPORTED,
            "operating_cash_flow": DataProvenance.REPORTED,
            "cash_and_equivalents": DataProvenance.REPORTED,
            "short_term_investments": DataProvenance.REPORTED,
            "debt": DataProvenance.REPORTED,
            "diluted_shares": DataProvenance.REPORTED,
        }
        if self.reported_free_cash_flow is not None:
            defaults["reported_free_cash_flow"] = DataProvenance.REPORTED
        merged = {**defaults, **self.field_provenance}
        object.__setattr__(self, "field_provenance", merged)
        object.__setattr__(self, "symbol", self.symbol.upper())
        return self

    def missing_required_fields(self) -> list[str]:
        """Return names of fields that failed to parse (always empty if constructed)."""
        return []
