"""Heuristic / stub agents — emit target portfolios only; never call brokers."""

from __future__ import annotations

import numpy as np

from rl_trader.agents.interface import PortfolioAgent
from rl_trader.portfolio.targets import TargetPortfolio

__all__ = ["PortfolioAgent", "StubAgent"]


class StubAgent:
    """Equal-weight placeholder agent — no learning implemented."""

    name = "stub_equal_weight"
    algorithm = "stub"

    def __init__(self, n_assets: int = 2) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = n_assets

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Emit a uniform target allocation."""
        _ = observation
        weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)
        return TargetPortfolio(weights=weights)
