"""Episode market / feature panels for the portfolio environment.

All arrays are aligned on the same daily index. Features at index ``t`` must
already be look-ahead-safe (causal) — the env never peeks at ``t+1`` when
building the observation for decision ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeData:
    """Aligned daily panel for one simulated trading episode.

    Parameters
    ----------
    timestamps:
        Length ``T`` decision timestamps (timezone-aware or naive).
    prices:
        Close / mid prices shaped ``(T, n_assets)`` used for marks and fills.
    market_features:
        Shape ``(T, n_market)``.
    fundamental_features:
        Shape ``(T, n_fundamental)``.
    regime_features:
        Shape ``(T, n_regime)``.
    symbols:
        Asset identifiers matching the price columns.
    """

    timestamps: np.ndarray
    prices: np.ndarray
    market_features: np.ndarray
    fundamental_features: np.ndarray
    regime_features: np.ndarray
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        t = int(self.timestamps.shape[0])
        if t < 2:
            raise ValueError("episode requires at least 2 bars (decide then mark)")
        n = len(self.symbols)
        if n < 1:
            raise ValueError("episode requires at least one asset")
        prices = np.asarray(self.prices, dtype=np.float64)
        if prices.shape != (t, n):
            raise ValueError(f"prices shape {prices.shape} != {(t, n)}")
        if not np.isfinite(prices).all() or np.any(prices <= 0.0):
            raise ValueError("prices must be finite and strictly positive")
        for name, arr in (
            ("market_features", self.market_features),
            ("fundamental_features", self.fundamental_features),
            ("regime_features", self.regime_features),
        ):
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 2 or a.shape[0] != t:
                raise ValueError(f"{name} must have shape (T, F); got {a.shape}")
        object.__setattr__(self, "prices", prices)
        object.__setattr__(
            self, "market_features", np.asarray(self.market_features, dtype=np.float64)
        )
        object.__setattr__(
            self,
            "fundamental_features",
            np.asarray(self.fundamental_features, dtype=np.float64),
        )
        object.__setattr__(
            self, "regime_features", np.asarray(self.regime_features, dtype=np.float64)
        )
        object.__setattr__(self, "timestamps", np.asarray(self.timestamps))

    @property
    def n_bars(self) -> int:
        """Number of daily bars."""
        return int(self.timestamps.shape[0])

    @property
    def n_assets(self) -> int:
        """Number of tradable assets."""
        return len(self.symbols)

    @property
    def n_market(self) -> int:
        """Market feature width."""
        return int(self.market_features.shape[1])

    @property
    def n_fundamental(self) -> int:
        """Fundamental feature width."""
        return int(self.fundamental_features.shape[1])

    @property
    def n_regime(self) -> int:
        """Regime feature width."""
        return int(self.regime_features.shape[1])

    def slice_bars(self, start: int, end: int) -> EpisodeData:
        """Return a contiguous bar slice ``[start, end)`` (end exclusive)."""
        if start < 0 or end > self.n_bars or end - start < 2:
            raise ValueError(
                f"invalid slice [{start}, {end}) for n_bars={self.n_bars} "
                "(need at least 2 bars)"
            )
        return EpisodeData(
            timestamps=self.timestamps[start:end],
            prices=self.prices[start:end],
            market_features=self.market_features[start:end],
            fundamental_features=self.fundamental_features[start:end],
            regime_features=self.regime_features[start:end],
            symbols=self.symbols,
        )
