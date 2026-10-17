"""OHLCV validation and data-quality checks (no silent fills)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from rl_trader.data.calendar import expected_trading_days
from rl_trader.data.schema import RAW_OHLCV_COLUMNS


@dataclass(slots=True)
class ValidationReport:
    """Outcome of a dataset validation / quality pass."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable report."""
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def validate_ohlcv_frame(
    frame: pd.DataFrame | None,
    *,
    start: date | None = None,
    end: date | None = None,
    jump_threshold: float = 0.20,
    allow_missing_bars: bool = False,
) -> ValidationReport:
    """Validate an OHLCV-like frame.

    Detects duplicated rows, missing trading dates, zero/negative prices,
    negative volume, and large suspicious close-to-close jumps. Never
    interpolates or fills gaps.
    """
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "jump_threshold": jump_threshold,
        "allow_missing_bars": allow_missing_bars,
    }

    if frame is None:
        return ValidationReport(ok=False, errors=["frame is None"], metadata=metadata)
    if not isinstance(frame, pd.DataFrame):
        return ValidationReport(ok=False, errors=["frame is not a DataFrame"], metadata=metadata)

    metadata["n_rows"] = int(len(frame))
    if frame.empty:
        errors.append("frame is empty")
        return ValidationReport(ok=False, errors=errors, warnings=warnings, metadata=metadata)

    missing_cols = [c for c in ("open", "high", "low", "close", "volume") if c not in frame.columns]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return ValidationReport(ok=False, errors=errors, warnings=warnings, metadata=metadata)

    if not isinstance(frame.index, pd.DatetimeIndex):
        errors.append("index must be a DatetimeIndex")
        return ValidationReport(ok=False, errors=errors, warnings=warnings, metadata=metadata)

    # Duplicates
    dup_mask = frame.index.duplicated(keep=False)
    n_dup_index = int(dup_mask.sum())
    metadata["duplicate_index_rows"] = n_dup_index
    if n_dup_index:
        errors.append(f"duplicated timestamps: {n_dup_index} rows")

    if not frame.index.is_monotonic_increasing:
        errors.append("timestamps are not sorted chronologically")

    # Missing trading dates
    idx_dates = [ts.date() for ts in frame.index]
    metadata["first_date"] = idx_dates[0].isoformat()
    metadata["last_date"] = idx_dates[-1].isoformat()
    range_start = start or idx_dates[0]
    range_end = end or idx_dates[-1]
    expected = expected_trading_days(range_start, range_end)
    actual = set(idx_dates)
    missing = [d for d in expected if d not in actual]
    metadata["missing_trading_dates_count"] = len(missing)
    metadata["missing_trading_dates"] = [d.isoformat() for d in missing[:50]]
    metadata["expected_trading_days"] = len(expected)
    if missing:
        msg = f"missing {len(missing)} trading dates (not interpolated)"
        if allow_missing_bars:
            warnings.append(msg)
        else:
            errors.append(msg)

    # Price / volume integrity
    price_cols = [c for c in ("open", "high", "low", "close", "adj_close") if c in frame.columns]
    prices = frame[price_cols].apply(pd.to_numeric, errors="coerce")
    bad_price = (prices <= 0) | prices.isna()
    n_bad_price = int(bad_price.any(axis=1).sum())
    metadata["zero_or_negative_or_nan_price_rows"] = n_bad_price
    if n_bad_price:
        errors.append(f"zero/negative/NaN prices in {n_bad_price} rows")

    volume = pd.to_numeric(frame["volume"], errors="coerce")
    n_neg_vol = int((volume < 0).sum())
    metadata["negative_volume_rows"] = n_neg_vol
    if n_neg_vol:
        errors.append(f"negative volume in {n_neg_vol} rows")

    # Suspicious jumps on close (or close_adj if present)
    close_col = "close_adj" if "close_adj" in frame.columns else "close"
    close = pd.to_numeric(frame[close_col], errors="coerce")
    rets = close.pct_change()
    jump_mask = rets.abs() > jump_threshold
    jump_mask = jump_mask.fillna(False)
    jump_dates = [
        ts.date().isoformat()
        for ts, flag in zip(frame.index, jump_mask, strict=True)
        if flag
    ]
    metadata["large_jump_count"] = len(jump_dates)
    metadata["large_jump_dates"] = jump_dates[:50]
    if jump_dates:
        warnings.append(
            f"{len(jump_dates)} close moves exceed jump_threshold={jump_threshold:.0%}"
        )

    # OHLC consistency soft checks
    if {"open", "high", "low", "close"}.issubset(frame.columns):
        o = frame["open"].astype(float)
        h = frame["high"].astype(float)
        low = frame["low"].astype(float)
        c = frame["close"].astype(float)
        inconsistent = (h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low)
        n_inconsistent = int(inconsistent.sum())
        metadata["ohlc_inconsistent_rows"] = n_inconsistent
        if n_inconsistent:
            warnings.append(f"OHLC inconsistencies in {n_inconsistent} rows")

    metadata["raw_columns_present"] = [c for c in RAW_OHLCV_COLUMNS if c in frame.columns]
    ok = not errors
    return ValidationReport(ok=ok, errors=errors, warnings=warnings, metadata=metadata)
