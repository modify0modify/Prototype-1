"""Unit tests for FCFF / DCF valuation (not investment advice)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from rl_trader.config.models import FundamentalValuationSettings
from rl_trader.features.dcf_bridge import events_from_statements, intrinsic_value_asof
from rl_trader.features.fundamental_features import align_fundamentals_asof
from rl_trader.fundamentals import (
    AnnualFinancials,
    FCFFMethod,
    compute_fcff,
    fcf_ocf_minus_capex,
    fcff_ebit_bridge,
    latest_statement_asof,
    load_annual_financials_json,
    run_dcf,
    run_dcf_asof,
    select_statements_asof,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "fundamentals" / "sample_annual.json"


def _stmt(**overrides: object) -> AnnualFinancials:
    base: dict[str, object] = {
        "symbol": "EXAM",
        "fiscal_period_end": date(2022, 12, 31),
        "publication_date": date(2023, 2, 20),
        "revenue": 1100.0,
        "ebit": 165.0,
        "effective_tax_rate": 0.21,
        "depreciation_amortization": 45.0,
        "capex": 55.0,
        "change_in_nwc": 10.0,
        "operating_cash_flow": 180.0,
        "reported_free_cash_flow": 125.0,
        "cash_and_equivalents": 90.0,
        "short_term_investments": 25.0,
        "debt": 180.0,
        "diluted_shares": 50.0,
    }
    base.update(overrides)
    return AnnualFinancials(**base)  # type: ignore[arg-type]


def test_fcff_ebit_bridge_positive_and_negative_dnwc() -> None:
    pos = fcff_ebit_bridge(100, 0.2, 10, 30, 15)
    # 100*0.8 + 10 - 30 - 15 = 45
    assert pos == pytest.approx(45.0)
    neg = fcff_ebit_bridge(100, 0.2, 10, 30, -15)
    # ΔNWC negative adds cash: 100*0.8 + 10 - 30 - (-15) = 75
    assert neg == pytest.approx(75.0)


def test_fcf_ocf_minus_capex() -> None:
    assert fcf_ocf_minus_capex(180.0, 55.0) == pytest.approx(125.0)


def test_invalid_terminal_growth_rejected() -> None:
    with pytest.raises(ValidationError, match="terminal_growth_rate"):
        FundamentalValuationSettings(wacc=0.08, terminal_growth_rate=0.08)
    stmt = _stmt()
    settings = FundamentalValuationSettings.model_construct(
        wacc=0.08,
        terminal_growth_rate=0.09,
        forecast_horizon_years=5,
        initial_fcff_growth=0.08,
        growth_decline_per_year=0.01,
        fcff_method="auto",
        annual_share_dilution=0.0,
        include_short_term_investments_in_cash=True,
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
        currency="USD",
        output_dir=Path("reports/valuations"),
    )
    with pytest.raises(ValueError, match="invalid terminal growth"):
        run_dcf(stmt, settings)


def test_discount_timing_end_of_year() -> None:
    stmt = _stmt(change_in_nwc=0.0)
    settings = FundamentalValuationSettings(
        wacc=0.10,
        terminal_growth_rate=0.02,
        forecast_horizon_years=2,
        initial_fcff_growth=0.0,
        growth_decline_per_year=0.0,
        fcff_method="ebit_bridge",
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    result = run_dcf(stmt, settings)
    assert len(result.forecast_cash_flows) == 2
    cf1, cf2 = result.forecast_cash_flows
    assert result.present_values[0] == pytest.approx(cf1 / 1.10)
    assert result.present_values[1] == pytest.approx(cf2 / (1.10**2))
    tv = cf2 * 1.02 / (0.10 - 0.02)
    assert result.terminal_value == pytest.approx(tv)
    assert result.pv_terminal_value == pytest.approx(tv / (1.10**2))


def test_cash_debt_adjustment_to_equity() -> None:
    settings = FundamentalValuationSettings(
        forecast_horizon_years=1,
        initial_fcff_growth=0.0,
        growth_decline_per_year=0.0,
        fcff_method="ebit_bridge",
        include_short_term_investments_in_cash=True,
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    result = run_dcf(_stmt(), settings)
    cash = 90.0 + 25.0
    assert result.equity_value == pytest.approx(result.enterprise_value + cash - 180.0)


def test_share_dilution_increases_share_count() -> None:
    base = FundamentalValuationSettings(
        forecast_horizon_years=2,
        annual_share_dilution=0.0,
        fcff_method="ebit_bridge",
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    diluted = FundamentalValuationSettings(
        forecast_horizon_years=2,
        annual_share_dilution=0.05,
        fcff_method="ebit_bridge",
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    stmt = _stmt(diluted_shares=100.0)
    r0 = run_dcf(stmt, base)
    r1 = run_dcf(stmt, diluted)
    assert r1.diluted_shares_used == pytest.approx(100.0 * (1.05**2))
    assert r1.estimated_value_per_share < r0.estimated_value_per_share


def test_missing_fields_rejected_by_model() -> None:
    with pytest.raises(ValidationError):
        AnnualFinancials(  # type: ignore[call-arg]
            symbol="X",
            fiscal_period_end=date(2022, 12, 31),
            publication_date=date(2023, 1, 1),
            revenue=1.0,
            # ebit missing
            effective_tax_rate=0.2,
            depreciation_amortization=1.0,
            capex=1.0,
            change_in_nwc=0.0,
            operating_cash_flow=1.0,
            cash_and_equivalents=1.0,
            debt=0.0,
            diluted_shares=1.0,
        )


def test_unit_consistency_scales_linearly_with_shares() -> None:
    settings = FundamentalValuationSettings(
        forecast_horizon_years=3,
        fcff_method="ebit_bridge",
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    a = run_dcf(_stmt(diluted_shares=50.0), settings)
    b = run_dcf(_stmt(diluted_shares=100.0), settings)
    assert a.enterprise_value == pytest.approx(b.enterprise_value)
    assert a.estimated_value_per_share == pytest.approx(2 * b.estimated_value_per_share)


def test_compute_fcff_methods_on_statement() -> None:
    stmt = _stmt(change_in_nwc=-5.0)
    bridge, method, _ = compute_fcff(stmt, FCFFMethod.EBIT_BRIDGE)
    assert method is FCFFMethod.EBIT_BRIDGE
    expected = 165 * (1 - 0.21) + 45 - 55 - (-5)
    assert bridge == pytest.approx(expected)
    ocf, method2, _ = compute_fcff(stmt, FCFFMethod.OCF_CAPEX)
    assert method2 is FCFFMethod.OCF_CAPEX
    assert ocf == pytest.approx(125.0)


def test_asof_prevents_future_statement_leakage() -> None:
    statements = load_annual_financials_json(FIXTURE)
    early = date(2022, 6, 1)
    known = select_statements_asof(statements, early)
    assert len(known) == 1
    assert known[0].fiscal_period_end.year == 2021
    assert latest_statement_asof(statements, early) is known[0]

    before_any = run_dcf_asof(statements, FundamentalValuationSettings(), date(2020, 1, 1))
    assert before_any is None

    settings = FundamentalValuationSettings(
        forecast_horizon_years=2,
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    v_early = intrinsic_value_asof(statements, settings, early)
    v_late = intrinsic_value_asof(statements, settings, date(2023, 3, 1))
    assert v_early is not None and v_late is not None
    # Different statements → values need not match; ensure late uses 2022 filing
    late_stmt = latest_statement_asof(statements, date(2023, 3, 1))
    assert late_stmt is not None and late_stmt.fiscal_period_end.year == 2022


def test_feature_asof_hides_dcf_before_publication() -> None:
    statements = load_annual_financials_json(FIXTURE)
    settings = FundamentalValuationSettings(
        forecast_horizon_years=2,
        sensitivity_wacc_deltas=(0.0,),
        sensitivity_g_deltas=(0.0,),
    )
    events = events_from_statements(statements, settings)
    idx = pd.bdate_range("2022-01-03", periods=40, tz="UTC")
    aligned = align_fundamentals_asof(idx, events)
    first_pub = min(e.publication_date for e in events)
    before = aligned.loc[aligned.index.date < first_pub, "estimated_intrinsic_value"]
    assert before.isna().all()
    after = aligned.loc[aligned.index.date >= first_pub, "estimated_intrinsic_value"]
    assert after.notna().any()


def test_sensitivity_matrix_present() -> None:
    result = run_dcf(
        _stmt(),
        FundamentalValuationSettings(forecast_horizon_years=2, fcff_method="ebit_bridge"),
    )
    assert "rows" in result.sensitivity_matrix
    assert result.sensitivity_matrix["unit"] == "estimated_value_per_share"
    assert result.assumptions["disclaimer"].startswith("Not investment advice")


def test_sample_fixture_disclaimer_present() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    assert "Not investment advice" in text
