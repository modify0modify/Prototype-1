"""Fundamental features lagged by publication date (no backfill into earlier dates)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings, FundamentalValuationSettings
from rl_trader.features.engineering import FeatureSpec

FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "revenue_growth",
    "fcf_growth",
    "operating_margin",
    "fcf_margin",
    "cash",
    "debt",
    "diluted_shares",
    "estimated_intrinsic_value",
    "price_to_dcf_value",
)


@dataclass(frozen=True, slots=True)
class FundamentalEvent:
    """One fundamentals release known only after ``publication_date``."""

    publication_date: date
    revenue_growth: float | None = None
    fcf_growth: float | None = None
    operating_margin: float | None = None
    fcf_margin: float | None = None
    cash: float | None = None
    debt: float | None = None
    diluted_shares: float | None = None
    estimated_intrinsic_value: float | None = None


def fundamental_feature_specs(settings: FeatureSettings) -> list[FeatureSpec]:
    """Declarative specs for fundamental features."""
    v = settings.feature_version
    return [FeatureSpec(name=name, lookback=1, version=v) for name in FUNDAMENTAL_COLUMNS]


def fundamentals_to_frame(events: list[FundamentalEvent]) -> pd.DataFrame:
    """Convert release events to a DataFrame indexed by publication_date (UTC midnight)."""
    if not events:
        return pd.DataFrame(columns=list(FUNDAMENTAL_COLUMNS[:-1]))
    rows = []
    for event in sorted(events, key=lambda e: e.publication_date):
        rows.append(
            {
                "publication_date": pd.Timestamp(event.publication_date, tz="UTC"),
                "revenue_growth": event.revenue_growth,
                "fcf_growth": event.fcf_growth,
                "operating_margin": event.operating_margin,
                "fcf_margin": event.fcf_margin,
                "cash": event.cash,
                "debt": event.debt,
                "diluted_shares": event.diluted_shares,
                "estimated_intrinsic_value": event.estimated_intrinsic_value,
            }
        )
    frame = pd.DataFrame(rows).set_index("publication_date").sort_index()
    # Never forward-fill across duplicates — keep last release on same day
    return frame[~frame.index.duplicated(keep="last")]


def align_fundamentals_asof(
    index: pd.DatetimeIndex,
    events: list[FundamentalEvent],
    *,
    price: pd.Series | None = None,
    valuation: FundamentalValuationSettings | None = None,
) -> pd.DataFrame:
    """Point-in-time align fundamentals with ``merge_asof`` (backward).

    A release with ``publication_date = D`` is visible only for feature rows
    with timestamp ``>= D``. Earlier dates stay NaN — no backfill into the past.
    """
    _ = valuation  # reserved for future DCF recomputation hooks
    out = pd.DataFrame(index=index)
    for col in FUNDAMENTAL_COLUMNS:
        out[col] = np.nan

    fund = fundamentals_to_frame(events)
    if fund.empty:
        return out

    left_ts = pd.DatetimeIndex(pd.DatetimeIndex(index).tz_convert("UTC")).as_unit("ns")
    right_ts = pd.DatetimeIndex(pd.DatetimeIndex(fund.index).tz_convert("UTC")).as_unit(
        "ns"
    )
    left = pd.DataFrame({"ts": left_ts}).sort_values("ts")
    right = fund.copy()
    right.index = right_ts
    right = right.reset_index(names="ts").sort_values("ts")
    merged = pd.merge_asof(left, right, on="ts", direction="backward")
    merged = merged.set_index(index)
    for col in fund.columns:
        out[col] = merged[col].to_numpy()

    if price is not None and "estimated_intrinsic_value" in out.columns:
        px = price.astype(float).reindex(index)
        dcf = out["estimated_intrinsic_value"].astype(float)
        out["price_to_dcf_value"] = px / dcf.replace(0.0, np.nan)
    return out


def compute_fundamental_features(
    index: pd.DatetimeIndex,
    settings: FeatureSettings,
    *,
    events: list[FundamentalEvent] | None = None,
    price: pd.Series | None = None,
    valuation: FundamentalValuationSettings | None = None,
) -> pd.DataFrame:
    """Compute fundamental feature block (NaNs when unavailable)."""
    _ = settings
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    return align_fundamentals_asof(
        index, events or [], price=price, valuation=valuation
    )
