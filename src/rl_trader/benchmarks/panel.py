"""Aligned price panel for shared benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class PricePanel:
    """Daily close panel shared by every baseline strategy.

    Parameters
    ----------
    timestamps:
        Length ``T`` decision dates.
    prices:
        Shape ``(T, n_assets)`` strictly positive closes.
    symbols:
        Column labels (e.g. ``SPY``, ``AAPL``).
    """

    timestamps: np.ndarray
    prices: np.ndarray
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        t = int(np.asarray(self.timestamps).shape[0])
        n = len(self.symbols)
        if t < 2:
            raise ValueError("price panel requires at least 2 bars")
        if n < 1:
            raise ValueError("price panel requires at least one symbol")
        prices = np.asarray(self.prices, dtype=np.float64)
        if prices.shape != (t, n):
            raise ValueError(f"prices shape {prices.shape} != {(t, n)}")
        if not np.isfinite(prices).all() or np.any(prices <= 0.0):
            raise ValueError("prices must be finite and strictly positive")
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "timestamps", np.asarray(self.timestamps))
        object.__setattr__(self, "symbols", tuple(str(s).upper() for s in self.symbols))

    @property
    def n_bars(self) -> int:
        """Number of bars."""
        return int(self.prices.shape[0])

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.symbols)
