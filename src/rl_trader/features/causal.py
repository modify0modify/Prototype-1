"""Causal (look-ahead-safe) pandas helpers.

Every helper uses only observations at timestamps ``<= t`` when computing the
value at ``t``. Warm-up periods yield explicit NaNs — rows are never dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Return ``series[t] / series[t-periods] - 1`` (NaN until enough history)."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    return series.pct_change(periods=periods)


def causal_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Trailing mean over ``window`` observations ending at ``t``."""
    return series.rolling(window=window, min_periods=window).mean()


def causal_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Trailing sample std over ``window`` observations ending at ``t``."""
    return series.rolling(window=window, min_periods=window).std(ddof=1)


def causal_rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Trailing minimum."""
    return series.rolling(window=window, min_periods=window).min()


def causal_rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Trailing maximum."""
    return series.rolling(window=window, min_periods=window).max()


def causal_ewm_mean(series: pd.Series, span: int) -> pd.Series:
    """Causal EWM mean (``adjust=False`` uses only past + current)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def causal_rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of ``series[t]`` within the trailing window ending at ``t``."""

    def _rank(values: np.ndarray) -> float:
        current = values[-1]
        if np.isnan(current):
            return np.nan
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            return np.nan
        return float(np.mean(valid <= current))

    return series.rolling(window=window, min_periods=window).apply(_rank, raw=True)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder true range using prior close (causal)."""
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)
