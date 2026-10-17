"""Unit tests for technical / regime / portfolio feature columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings
from rl_trader.features.pipeline import build_feature_matrix, collect_feature_specs
from rl_trader.features.portfolio_state import PortfolioStateSeries
from rl_trader.features.regime import compute_regime_features
from rl_trader.features.technical import compute_technical_features


def _ohlcv(n: int = 80) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-03", periods=n, tz="UTC")
    close = pd.Series(np.linspace(100, 120, n), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_technical_columns_present() -> None:
    settings = FeatureSettings(
        include_regime=False,
        include_portfolio_state=False,
        include_fundamentals=False,
    )
    frame = compute_technical_features(
        _ohlcv(), settings, benchmark_close=_ohlcv(80)["close"]
    )
    expected = {
        "return_1d",
        "return_5d",
        "return_20d",
        "return_60d",
        "ma_ratio_5d",
        "ma_ratio_20d",
        "ma_ratio_60d",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_position",
        "atr",
        "rolling_volatility",
        "rolling_downside_volatility",
        "volume_zscore",
        "rolling_max_drawdown",
        "distance_from_rolling_high",
        "market_relative_return_1d",
    }
    assert expected.issubset(set(frame.columns))
    assert len(frame) == 80
    # Warm-up NaNs exist for long windows
    assert frame["return_60d"].isna().sum() >= 59


def test_regime_indicators() -> None:
    settings = FeatureSettings(vol_percentile_window=30, regime_trend_window=10)
    close = _ohlcv(60)["close"]
    regime = compute_regime_features(close, settings, index=close.index)
    for col in (
        "benchmark_trend",
        "benchmark_volatility",
        "volatility_percentile",
        "drawdown_regime",
        "regime_bull",
        "regime_bear",
        "regime_sideways",
        "regime_high_vol",
        "regime_low_vol",
    ):
        assert col in regime.columns


def test_portfolio_state_and_specs() -> None:
    settings = FeatureSettings()
    idx = _ohlcv(20).index
    state = PortfolioStateSeries(
        cash_pct=pd.Series(0.2, index=idx),
        position_weight=pd.Series(0.8, index=idx),
        unrealised_return=pd.Series(0.01, index=idx),
        realised_return=pd.Series(0.02, index=idx),
        current_drawdown=pd.Series(-0.05, index=idx),
        turnover=pd.Series(0.1, index=idx),
        exposure=pd.Series(0.8, index=idx),
        time_since_last_trade=pd.Series(3.0, index=idx),
    )
    features, specs = build_feature_matrix(
        _ohlcv(20),
        settings,
        symbol="SPY",
        benchmark_ohlcv=_ohlcv(20),
        portfolio_state=state,
    )
    assert "cash_pct" in features.columns
    assert features["cash_pct"].iloc[-1] == 0.2
    names = {s.name for s in collect_feature_specs(settings)}
    assert "rsi" in names
    assert "regime_bull" in names
    assert "revenue_growth" in names
    assert "recent_turnover" in names
