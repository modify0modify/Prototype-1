"""Look-ahead-safe technical features from cleaned OHLCV."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings
from rl_trader.features.causal import (
    causal_ewm_mean,
    causal_pct_change,
    causal_rolling_max,
    causal_rolling_mean,
    causal_rolling_std,
    true_range,
)
from rl_trader.features.engineering import FeatureSpec


def _price_series(ohlcv: pd.DataFrame) -> pd.Series:
    if "close_adj" in ohlcv.columns:
        return ohlcv["close_adj"].astype(float)
    return ohlcv["close"].astype(float)


def technical_feature_specs(settings: FeatureSettings) -> list[FeatureSpec]:
    """Declarative specs for technical features."""
    specs: list[FeatureSpec] = []
    for w in settings.return_windows:
        specs.append(FeatureSpec(name=f"return_{w}d", lookback=w, version=settings.feature_version))
    for w in settings.ma_windows:
        specs.append(
            FeatureSpec(name=f"ma_ratio_{w}d", lookback=w, version=settings.feature_version)
        )
    specs.extend(
        [
            FeatureSpec(name="rsi", lookback=settings.rsi_period, version=settings.feature_version),
            FeatureSpec(name="macd", lookback=settings.macd_slow, version=settings.feature_version),
            FeatureSpec(
                name="macd_signal", lookback=settings.macd_slow, version=settings.feature_version
            ),
            FeatureSpec(
                name="macd_hist", lookback=settings.macd_slow, version=settings.feature_version
            ),
            FeatureSpec(
                name="bb_position",
                lookback=settings.bollinger_period,
                version=settings.feature_version,
            ),
            FeatureSpec(name="atr", lookback=settings.atr_period, version=settings.feature_version),
            FeatureSpec(
                name="rolling_volatility",
                lookback=settings.volatility_window,
                version=settings.feature_version,
            ),
            FeatureSpec(
                name="rolling_downside_volatility",
                lookback=settings.downside_vol_window,
                version=settings.feature_version,
            ),
            FeatureSpec(
                name="volume_zscore",
                lookback=settings.volume_zscore_window,
                version=settings.feature_version,
            ),
            FeatureSpec(
                name="rolling_max_drawdown",
                lookback=settings.drawdown_window,
                version=settings.feature_version,
            ),
            FeatureSpec(
                name="distance_from_rolling_high",
                lookback=settings.rolling_high_window,
                version=settings.feature_version,
            ),
            FeatureSpec(
                name="market_relative_return_1d",
                lookback=1,
                version=settings.feature_version,
            ),
        ]
    )
    return specs


def compute_technical_features(
    ohlcv: pd.DataFrame,
    settings: FeatureSettings,
    *,
    benchmark_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute technical features; index aligned to ``ohlcv`` (rows never dropped)."""
    close = _price_series(ohlcv)
    high = ohlcv["high"].astype(float) if "high" in ohlcv.columns else close
    low = ohlcv["low"].astype(float) if "low" in ohlcv.columns else close
    volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else pd.Series(
        np.nan, index=ohlcv.index
    )

    out = pd.DataFrame(index=ohlcv.index)

    for w in settings.return_windows:
        out[f"return_{w}d"] = causal_pct_change(close, w)

    for w in settings.ma_windows:
        ma = causal_rolling_mean(close, w)
        out[f"ma_ratio_{w}d"] = close / ma - 1.0

    # RSI (Wilder-style via EWM of gains/losses)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rsi_alpha = 1 / settings.rsi_period
    avg_gain = gain.ewm(
        alpha=rsi_alpha, adjust=False, min_periods=settings.rsi_period
    ).mean()
    avg_loss = loss.ewm(
        alpha=rsi_alpha, adjust=False, min_periods=settings.rsi_period
    ).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    ema_fast = causal_ewm_mean(close, settings.macd_fast)
    ema_slow = causal_ewm_mean(close, settings.macd_slow)
    macd = ema_fast - ema_slow
    signal = causal_ewm_mean(macd, settings.macd_signal)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    mid = causal_rolling_mean(close, settings.bollinger_period)
    std = causal_rolling_std(close, settings.bollinger_period)
    upper = mid + settings.bollinger_std * std
    lower = mid - settings.bollinger_std * std
    band = upper - lower
    out["bb_position"] = (close - lower) / band.replace(0.0, np.nan)

    tr = true_range(high, low, close)
    out["atr"] = causal_ewm_mean(tr, settings.atr_period)

    rets = causal_pct_change(close, 1)
    out["rolling_volatility"] = causal_rolling_std(rets, settings.volatility_window)
    downside = rets.where(rets < 0.0, 0.0)
    out["rolling_downside_volatility"] = causal_rolling_std(
        downside, settings.downside_vol_window
    )

    vol_mean = causal_rolling_mean(volume, settings.volume_zscore_window)
    vol_std = causal_rolling_std(volume, settings.volume_zscore_window)
    out["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0.0, np.nan)

    roll_max = causal_rolling_max(close, settings.drawdown_window)
    out["rolling_max_drawdown"] = close / roll_max - 1.0
    high_roll = causal_rolling_max(close, settings.rolling_high_window)
    out["distance_from_rolling_high"] = close / high_roll - 1.0

    if benchmark_close is not None:
        bench = benchmark_close.reindex(ohlcv.index)
        asset_ret = causal_pct_change(close, 1)
        bench_ret = causal_pct_change(bench.astype(float), 1)
        out["market_relative_return_1d"] = asset_ret - bench_ret
    else:
        out["market_relative_return_1d"] = np.nan

    return out
