"""Synthetic episode builders for offline tests and smoke demos (no network)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from rl_trader.environment.episode import EpisodeData


def make_synthetic_episode(
    *,
    n_bars: int = 30,
    n_assets: int = 2,
    n_market: int = 3,
    n_fundamental: int = 2,
    n_regime: int = 2,
    seed: int = 0,
    price_path: str = "drift",
) -> EpisodeData:
    """Build a finite, look-ahead-labelled synthetic panel.

    Feature columns are deterministic functions of the *current* bar index only
    (no future leakage by construction).

    ``price_path``:
      - ``drift``: gentle random walk (default)
      - ``crash``: sharp drop mid-episode for drawdown tests
      - ``flat``: constant prices
    """
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    timestamps = np.array(
        [start + timedelta(days=i) for i in range(n_bars)], dtype=object
    )
    symbols = tuple(f"A{i}" for i in range(n_assets))

    if price_path == "flat":
        prices = np.full((n_bars, n_assets), 100.0, dtype=np.float64)
    elif price_path == "crash":
        prices = np.full((n_bars, n_assets), 100.0, dtype=np.float64)
        for t in range(1, n_bars):
            shock = -0.15 if t == n_bars // 3 else 0.001
            prices[t] = prices[t - 1] * (1.0 + shock)
    else:
        returns = rng.normal(0.001, 0.01, size=(n_bars, n_assets))
        prices = 100.0 * np.cumprod(1.0 + returns, axis=0)

    t_idx = np.arange(n_bars, dtype=np.float64)
    market = (
        np.column_stack([np.sin((t_idx + k) / 5.0) for k in range(n_market)])
        if n_market
        else np.zeros((n_bars, 0))
    )
    fundamental = (
        np.column_stack([0.1 * (k + 1) + 0.01 * t_idx for k in range(n_fundamental)])
        if n_fundamental
        else np.zeros((n_bars, 0))
    )
    regime = (
        np.column_stack([np.cos((t_idx + k) / 7.0) for k in range(n_regime)])
        if n_regime
        else np.zeros((n_bars, 0))
    )

    return EpisodeData(
        timestamps=timestamps,
        prices=prices,
        market_features=market,
        fundamental_features=fundamental,
        regime_features=regime,
        symbols=symbols,
    )
