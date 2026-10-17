"""Produce cleaned OHLCV from raw — never interpolate missing prices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rl_trader.data.schema import ADJUSTMENT_METHOD, CLEANED_OHLCV_COLUMNS, RAW_OHLCV_COLUMNS


@dataclass(slots=True)
class CleaningResult:
    """Cleaned frame plus non-destructive cleaning notes."""

    frame: pd.DataFrame
    duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)
    adjustment_method: str = ADJUSTMENT_METHOD


def clean_ohlcv(raw: pd.DataFrame) -> CleaningResult:
    """Sort, drop exact duplicate timestamps, apply adjustment factors.

    Missing trading dates are **not** filled. Rows with invalid prices are kept
    for validation to flag (cleaning does not invent prices).
    """
    warnings: list[str] = []
    if raw.empty:
        empty = pd.DataFrame(columns=list(CLEANED_OHLCV_COLUMNS))
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return CleaningResult(frame=empty, warnings=["raw frame empty"])

    frame = raw.copy()
    # Ensure canonical raw columns exist
    for col in RAW_OHLCV_COLUMNS:
        if col not in frame.columns:
            if col == "adj_close" and "close" in frame.columns:
                frame["adj_close"] = frame["close"]
                warnings.append("adj_close missing; copied from close (factor=1)")
            else:
                raise ValueError(f"raw frame missing required column {col!r}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    elif frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    normalized = [ts.normalize() for ts in frame.index]
    frame.index = pd.DatetimeIndex(normalized, tz="UTC", name="timestamp")

    before = len(frame)
    frame = frame[~frame.index.duplicated(keep="first")]
    duplicates_removed = before - len(frame)
    frame = frame.sort_index()

    close = frame["close"].astype(float)
    adj = frame["adj_close"].astype(float)
    factor = np.where(close != 0.0, adj / close, 1.0)
    frame["adjustment_factor"] = factor
    frame["open_adj"] = frame["open"].astype(float) * factor
    frame["high_adj"] = frame["high"].astype(float) * factor
    frame["low_adj"] = frame["low"].astype(float) * factor
    frame["close_adj"] = adj

    # Preserve volume as-is (unadjusted)
    frame = frame.loc[:, list(CLEANED_OHLCV_COLUMNS)]
    return CleaningResult(
        frame=frame,
        duplicates_removed=duplicates_removed,
        warnings=warnings,
        adjustment_method=ADJUSTMENT_METHOD,
    )
