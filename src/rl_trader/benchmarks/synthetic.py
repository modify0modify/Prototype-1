"""Offline synthetic price panels for benchmark tests / demos (not market data)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from rl_trader.benchmarks.panel import PricePanel


def make_synthetic_price_panel(
    *,
    n_bars: int = 120,
    symbols: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT"),
    seed: int = 0,
) -> PricePanel:
    """Build a finite random-walk panel including a SPY column when requested."""
    rng = np.random.default_rng(seed)
    start = date(2023, 1, 3)
    timestamps = np.array(
        [start + timedelta(days=i) for i in range(n_bars)], dtype=object
    )
    n = len(symbols)
    # Mildly correlated drifts; SPY slightly lower vol.
    vols = np.full(n, 0.012, dtype=np.float64)
    for i, sym in enumerate(symbols):
        if sym.upper() == "SPY":
            vols[i] = 0.008
    rets = rng.normal(0.0004, 1.0, size=(n_bars, n)) * vols
    prices = 100.0 * np.cumprod(1.0 + rets, axis=0)
    return PricePanel(timestamps=timestamps, prices=prices, symbols=symbols)
