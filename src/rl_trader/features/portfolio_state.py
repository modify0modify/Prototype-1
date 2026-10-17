"""Portfolio-state features (causal; explicit NaNs when state unavailable)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rl_trader.config.models import FeatureSettings
from rl_trader.features.causal import causal_rolling_mean
from rl_trader.features.engineering import FeatureSpec


@dataclass(frozen=True, slots=True)
class PortfolioStateSeries:
    """Optional daily portfolio state aligned to the feature index.

    All series must be known as-of the timestamp (no look-ahead). Missing
    inputs remain NaN in the output — rows are not dropped.
    """

    cash_pct: pd.Series | None = None
    position_weight: pd.Series | None = None
    unrealised_return: pd.Series | None = None
    realised_return: pd.Series | None = None
    current_drawdown: pd.Series | None = None
    turnover: pd.Series | None = None
    exposure: pd.Series | None = None
    time_since_last_trade: pd.Series | None = None


def portfolio_feature_specs(settings: FeatureSettings) -> list[FeatureSpec]:
    """Declarative specs for portfolio-state features."""
    v = settings.feature_version
    return [
        FeatureSpec(name="cash_pct", lookback=1, version=v),
        FeatureSpec(name="position_weight", lookback=1, version=v),
        FeatureSpec(name="unrealised_return", lookback=1, version=v),
        FeatureSpec(name="realised_return", lookback=1, version=v),
        FeatureSpec(name="current_drawdown", lookback=1, version=v),
        FeatureSpec(
            name="recent_turnover", lookback=settings.turnover_window, version=v
        ),
        FeatureSpec(name="exposure", lookback=1, version=v),
        FeatureSpec(name="time_since_last_trade", lookback=1, version=v),
    ]


def compute_portfolio_state_features(
    index: pd.Index,
    settings: FeatureSettings,
    state: PortfolioStateSeries | None = None,
) -> pd.DataFrame:
    """Build portfolio-state feature frame; NaNs when state not provided."""
    out = pd.DataFrame(index=index)
    empty = pd.Series(np.nan, index=index, dtype=float)
    state = state or PortfolioStateSeries()

    def _align(series: pd.Series | None) -> pd.Series:
        if series is None:
            return empty.copy()
        return series.astype(float).reindex(index)

    out["cash_pct"] = _align(state.cash_pct)
    out["position_weight"] = _align(state.position_weight)
    out["unrealised_return"] = _align(state.unrealised_return)
    out["realised_return"] = _align(state.realised_return)
    out["current_drawdown"] = _align(state.current_drawdown)
    turnover = _align(state.turnover)
    out["recent_turnover"] = causal_rolling_mean(turnover, settings.turnover_window)
    out["exposure"] = _align(state.exposure)
    out["time_since_last_trade"] = _align(state.time_since_last_trade)
    return out
