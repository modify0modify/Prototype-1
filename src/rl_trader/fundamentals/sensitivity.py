"""WACC × terminal-growth sensitivity for per-share DCF values."""

from __future__ import annotations

from typing import Any


def _per_share_from_forecast(
    forecast_cash_flows: tuple[float, ...],
    wacc: float,
    g: float,
    cash: float,
    debt: float,
    shares: float,
) -> float | None:
    """Recompute equity value / shares for a (wacc, g) pair; None if invalid."""
    if wacc <= 0.0 or g >= wacc or shares <= 0.0:
        return None
    n = len(forecast_cash_flows)
    if n < 1:
        return None
    pvs = [cf / ((1.0 + wacc) ** t) for t, cf in enumerate(forecast_cash_flows, start=1)]
    fcf_final = forecast_cash_flows[-1]
    tv = fcf_final * (1.0 + g) / (wacc - g)
    pv_tv = tv / ((1.0 + wacc) ** n)
    ev = sum(pvs) + pv_tv
    equity = ev + cash - debt
    return equity / shares


def build_sensitivity_matrix(
    *,
    forecast_cash_flows: tuple[float, ...],
    base_wacc: float,
    base_g: float,
    wacc_deltas: tuple[float, ...],
    g_deltas: tuple[float, ...],
    cash: float,
    debt: float,
    shares: float,
) -> dict[str, Any]:
    """Return a matrix of estimated value/share keyed by WACC and g."""
    waccs = [base_wacc + d for d in wacc_deltas]
    gs = [base_g + d for d in g_deltas]
    rows: list[dict[str, Any]] = []
    for w in waccs:
        row: dict[str, Any] = {"wacc": w, "values": {}}
        for g in gs:
            value = _per_share_from_forecast(
                forecast_cash_flows, w, g, cash, debt, shares
            )
            row["values"][f"g={g:.4f}"] = value
        rows.append(row)
    return {
        "wacc_axis": waccs,
        "g_axis": gs,
        "base_wacc": base_wacc,
        "base_g": base_g,
        "unit": "estimated_value_per_share",
        "rows": rows,
        "note": "None cells mark invalid pairs (g >= WACC). Not investment advice.",
    }
