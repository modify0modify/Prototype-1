"""Legacy benchmark shims — prefer :mod:`rl_trader.benchmarks`."""

from __future__ import annotations

import numpy as np

from rl_trader.benchmarks.strategies import (
    BuyAndHoldSpyStrategy,
    EqualWeightBuyAndHoldStrategy,
)
from rl_trader.portfolio.targets import TargetPortfolio


class EqualWeightBenchmark:
    """Always allocate equally across assets (legacy agent-style API)."""

    name = "equal_weight"

    def __init__(self, n_assets: int = 2) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = n_assets
        self._inner = EqualWeightBuyAndHoldStrategy()
        # prepare with flat prices so target_weights can run in agent-style mode
        flat = np.ones((2, n_assets), dtype=np.float64)
        symbols = tuple(f"A{i}" for i in range(n_assets))
        self._inner.prepare(flat, symbols)

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Uniform weights."""
        _ = observation
        weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)
        return TargetPortfolio(weights=weights)


class BuyAndHoldBenchmark:
    """Hold a static weight vector (defaults to first asset = 100%)."""

    name = "buy_and_hold"

    def __init__(self, weights: np.ndarray | None = None, n_assets: int = 2) -> None:
        if weights is None:
            w = np.zeros(n_assets, dtype=np.float64)
            w[0] = 1.0
            self.weights = w
        else:
            self.weights = np.asarray(weights, dtype=np.float64)
        self._spy = BuyAndHoldSpyStrategy(spy_symbol="A0")

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Return the static allocation."""
        _ = observation
        return TargetPortfolio(weights=self.weights.copy())
