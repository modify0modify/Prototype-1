"""Common strategy interface for deterministic baselines."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Strategy(Protocol):
    """Deterministic portfolio strategy (research baselines — not advice)."""

    name: str
    required_lookback: int

    def prepare(self, prices: np.ndarray, symbols: tuple[str, ...]) -> None:
        """Optional causal pre-computation over the full price panel."""
        ...

    def target_weights(self, t: int) -> np.ndarray:
        """Return long-only asset weights at decision bar ``t``.

        Implementations must use only information available at ``t``
        (prices up to and including ``t``). Cash is the residual ``1 - sum(w)``.
        """
        ...


class BaseStrategy:
    """Convenience base with shared prepare bookkeeping."""

    name: str = "base"
    required_lookback: int = 1
    # When False, harness will not re-apply max_weight caps (buy-and-hold drift).
    enforce_max_weight: bool = True

    def __init__(self) -> None:
        self._prices: np.ndarray | None = None
        self._symbols: tuple[str, ...] = ()
        self._n_assets: int = 0

    def prepare(self, prices: np.ndarray, symbols: tuple[str, ...]) -> None:
        """Store panel references (read-only use by subclasses)."""
        arr = np.asarray(prices, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(symbols):
            raise ValueError("prices must be (T, n_assets) matching symbols")
        self._prices = arr
        self._symbols = tuple(symbols)
        self._n_assets = len(symbols)

    def target_weights(self, t: int) -> np.ndarray:
        """Subclasses must implement causal target weights at bar ``t``."""
        raise NotImplementedError

    def _require_prepared(self) -> np.ndarray:
        if self._prices is None:
            raise RuntimeError(f"{self.name}: call prepare() before target_weights()")
        return self._prices
