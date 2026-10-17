"""Gymnasium trading environment stubs (research).

Agents produce target portfolio weights; no order execution occurs here.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PortfolioEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal Gymnasium env placeholder for portfolio allocation."""

    metadata = {"render_modes": []}

    def __init__(self, n_assets: int = 2, seed: int | None = None) -> None:
        super().__init__()
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1")
        self.n_assets = n_assets
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_assets,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset episode; optional seed hooks for determinism."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        _ = options
        obs = np.zeros(self.n_assets, dtype=np.float32)
        return obs, {"target_portfolio_only": True}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one stub step; returns a zero reward placeholder."""
        _ = action
        self._step_count += 1
        obs = np.zeros(self.n_assets, dtype=np.float32)
        terminated = self._step_count >= 1
        return obs, 0.0, terminated, False, {"note": "env stub — no market dynamics"}
