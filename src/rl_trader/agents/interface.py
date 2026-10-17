"""Common agent interface wrapping Stable-Baselines3 algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_trader.agents.registry import AlgorithmSpec, get_algorithm_spec
from rl_trader.portfolio.targets import TargetPortfolio


class PortfolioAgent(Protocol):
    """Agents emit target portfolio weights only — never broker orders."""

    name: str
    algorithm: str

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Map an observation to long-only target weights."""
        ...


class SB3PortfolioAgent:
    """Thin wrapper around a trained SB3 model (PPO / SAC / TD3)."""

    def __init__(
        self,
        model: BaseAlgorithm,
        *,
        algorithm: str,
        name: str | None = None,
        deterministic: bool = True,
    ) -> None:
        self.model = model
        self.algorithm = algorithm.lower()
        self.name = name or f"sb3-{self.algorithm}"
        self.deterministic = deterministic
        shape = model.action_space.shape
        self.n_assets = int(np.prod(shape)) if shape is not None else 0

    def predict_target(self, observation: np.ndarray) -> TargetPortfolio:
        """Predict continuous target weights in the env action Box."""
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        weights = np.asarray(action, dtype=np.float64).reshape(-1)
        return TargetPortfolio(weights=weights)

    @classmethod
    def load(
        cls,
        directory: Path,
        algorithm: str,
        *,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> SB3PortfolioAgent:
        """Load weights from a checkpoint directory via the algorithm registry."""
        spec = get_algorithm_spec(algorithm)
        model_path = directory / spec.model_filename
        if not model_path.exists():
            stem = model_path.with_suffix("")
            if stem.with_suffix(".zip").exists():
                model_path = stem.with_suffix(".zip")
            else:
                raise FileNotFoundError(f"missing model weights: {model_path}")
        model = spec.sb3_cls.load(str(model_path), device=device)
        return cls(
            model,
            algorithm=spec.name,
            name=f"sb3-{spec.name}",
            deterministic=deterministic,
        )

    def save(self, directory: Path, spec: AlgorithmSpec | None = None) -> Path:
        """Persist SB3 weights using the registry filename."""
        directory.mkdir(parents=True, exist_ok=True)
        algo_spec = spec or get_algorithm_spec(self.algorithm)
        path = directory / algo_spec.model_filename
        self.model.save(str(path.with_suffix("")))
        return path


def describe_action_space(model: BaseAlgorithm) -> dict[str, Any]:
    """Summarize the continuous action Box for manifests / reports."""
    space = cast(spaces.Box, model.action_space)
    shape = space.shape if space.shape is not None else ()
    return {
        "type": type(space).__name__,
        "shape": list(shape),
        "low": np.asarray(space.low, dtype=np.float64).tolist(),
        "high": np.asarray(space.high, dtype=np.float64).tolist(),
    }
