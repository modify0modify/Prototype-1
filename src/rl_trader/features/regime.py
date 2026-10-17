"""Market-regime features from a benchmark series (causal)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings
from rl_trader.features.causal import (
    causal_pct_change,
    causal_rolling_max,
    causal_rolling_mean,
    causal_rolling_percentile_rank,
    causal_rolling_std,
)
from rl_trader.features.engineering import FeatureSpec


def regime_feature_specs(settings: FeatureSettings) -> list[FeatureSpec]:
    """Declarative specs for regime features."""
    v = settings.feature_version
    return [
        FeatureSpec(name="benchmark_trend", lookback=settings.regime_trend_window, version=v),
        FeatureSpec(name="benchmark_volatility", lookback=settings.regime_vol_window, version=v),
        FeatureSpec(
            name="volatility_percentile", lookback=settings.vol_percentile_window, version=v
        ),
        FeatureSpec(name="drawdown_regime", lookback=settings.regime_trend_window, version=v),
        FeatureSpec(name="regime_bull", lookback=settings.regime_trend_window, version=v),
        FeatureSpec(name="regime_bear", lookback=settings.regime_trend_window, version=v),
        FeatureSpec(name="regime_sideways", lookback=settings.regime_trend_window, version=v),
        FeatureSpec(name="regime_high_vol", lookback=settings.vol_percentile_window, version=v),
        FeatureSpec(name="regime_low_vol", lookback=settings.vol_percentile_window, version=v),
    ]


def compute_regime_features(
    benchmark_close: pd.Series,
    settings: FeatureSettings,
    *,
    index: pd.Index,
) -> pd.DataFrame:
    """Compute regime features aligned to ``index`` (no row drops)."""
    close = benchmark_close.astype(float).reindex(index)
    out = pd.DataFrame(index=index)

    ma = causal_rolling_mean(close, settings.regime_trend_window)
    trend = close / ma - 1.0
    out["benchmark_trend"] = trend

    rets = causal_pct_change(close, 1)
    vol = causal_rolling_std(rets, settings.regime_vol_window)
    out["benchmark_volatility"] = vol
    out["volatility_percentile"] = causal_rolling_percentile_rank(
        vol, settings.vol_percentile_window
    )

    peak = causal_rolling_max(close, settings.regime_trend_window)
    dd = close / peak - 1.0
    out["drawdown_regime"] = dd

    out["regime_bull"] = (trend > settings.bull_threshold).astype(float)
    out["regime_bear"] = (trend < settings.bear_threshold).astype(float)
    out["regime_sideways"] = (
        (trend >= settings.bear_threshold) & (trend <= settings.bull_threshold)
    ).astype(float)
    # Warm-up NaNs: where trend is NaN, regime flags should be NaN not 0
    warm = trend.isna()
    for col in ("regime_bull", "regime_bear", "regime_sideways"):
        out.loc[warm, col] = np.nan

    out["regime_high_vol"] = (
        out["volatility_percentile"] >= settings.high_vol_percentile
    ).astype(float)
    out["regime_low_vol"] = (
        out["volatility_percentile"] <= settings.low_vol_percentile
    ).astype(float)
    vol_warm = out["volatility_percentile"].isna()
    out.loc[vol_warm, "regime_high_vol"] = np.nan
    out.loc[vol_warm, "regime_low_vol"] = np.nan
    return out
