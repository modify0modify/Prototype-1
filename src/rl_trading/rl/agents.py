"""RL agent stubs (Stable-Baselines3 / PyTorch integration points).

Agents output target portfolios only. Order routing is never performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    """Model-proposed portfolio weights (pre-risk)."""

    weights: np.ndarray
    as_of: str | None = None


class PortfolioAgent(Protocol):
    """Interface for trained or heuristic portfolio agents."""

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Return a target portfolio for the given observation."""
        ...

    def save(self, path: Path) -> None:
        """Persist agent parameters."""
        ...

    def load(self, path: Path) -> None:
        """Load agent parameters."""
        ...


class StubAgent:
    """Equal-weight placeholder agent — no learning implemented."""

    def __init__(self, n_assets: int = 2) -> None:
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = n_assets

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Emit a uniform target allocation."""
        _ = observation
        weights = np.full(self.n_assets, 1.0 / self.n_assets, dtype=np.float64)
        return TargetPortfolio(weights=weights)

    def save(self, path: Path) -> None:
        """Stub save — writes a marker file only."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub-agent\n", encoding="utf-8")

    def load(self, path: Path) -> None:
        """Stub load — verifies marker exists."""
        if not path.exists():
            raise FileNotFoundError(path)
