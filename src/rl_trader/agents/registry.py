"""Algorithm registry for Stable-Baselines3 portfolio agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from rl_trader.config.models import TrainingSettings

AlgorithmName = Literal["ppo", "sac", "td3"]

SUPPORTED_ALGORITHMS: tuple[AlgorithmName, ...] = ("ppo", "sac", "td3")


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    """Factory + hyperparameter projection for one SB3 algorithm."""

    name: AlgorithmName
    sb3_cls: type[BaseAlgorithm]
    policy: str
    model_filename: str
    kwargs_builder: Callable[[TrainingSettings], dict[str, Any]]

    def build(self, env: VecEnv, training: TrainingSettings) -> BaseAlgorithm:
        """Instantiate the SB3 algorithm on ``env``."""
        return self.sb3_cls(
            self.policy,
            env,
            seed=training.seed,
            device=training.device,
            verbose=0,
            **self.kwargs_builder(training),
        )

    def hyperparameters(self, training: TrainingSettings) -> dict[str, Any]:
        """Algorithm-specific hyperparameters stored with checkpoints."""
        params = dict(self.kwargs_builder(training))
        params["algorithm"] = self.name
        params["policy"] = self.policy
        params["learning_rate"] = training.learning_rate
        params["gamma"] = training.gamma
        params["batch_size"] = training.batch_size
        params["seed"] = training.seed
        params["device"] = training.device
        params["total_timesteps"] = training.total_timesteps
        return params


def _ppo_kwargs(training: TrainingSettings) -> dict[str, Any]:
    return {
        "learning_rate": training.learning_rate,
        "n_steps": training.n_steps,
        "batch_size": training.batch_size,
        "n_epochs": training.n_epochs,
        "gamma": training.gamma,
        "gae_lambda": training.gae_lambda,
        "clip_range": training.clip_range,
        "ent_coef": training.ent_coef,
        "vf_coef": training.vf_coef,
        "max_grad_norm": training.max_grad_norm,
    }


def _sac_kwargs(training: TrainingSettings) -> dict[str, Any]:
    return {
        "learning_rate": training.learning_rate,
        "buffer_size": training.buffer_size,
        "learning_starts": training.learning_starts,
        "batch_size": training.batch_size,
        "tau": training.tau,
        "gamma": training.gamma,
        "train_freq": training.train_freq,
        "gradient_steps": training.gradient_steps,
        # Keep entropy coef numeric for reproducibility in manifests.
        "ent_coef": training.ent_coef if training.ent_coef > 0 else 0.01,
    }


def _td3_kwargs(training: TrainingSettings) -> dict[str, Any]:
    return {
        "learning_rate": training.learning_rate,
        "buffer_size": training.buffer_size,
        "learning_starts": training.learning_starts,
        "batch_size": training.batch_size,
        "tau": training.tau,
        "gamma": training.gamma,
        "train_freq": training.train_freq,
        "gradient_steps": training.gradient_steps,
        "policy_delay": training.policy_delay,
        "target_policy_noise": training.target_policy_noise,
        "target_noise_clip": training.target_noise_clip,
    }


_REGISTRY: dict[str, AlgorithmSpec] = {
    "ppo": AlgorithmSpec(
        name="ppo",
        sb3_cls=PPO,
        policy="MlpPolicy",
        model_filename="ppo_model.zip",
        kwargs_builder=_ppo_kwargs,
    ),
    "sac": AlgorithmSpec(
        name="sac",
        sb3_cls=SAC,
        policy="MlpPolicy",
        model_filename="sac_model.zip",
        kwargs_builder=_sac_kwargs,
    ),
    "td3": AlgorithmSpec(
        name="td3",
        sb3_cls=TD3,
        policy="MlpPolicy",
        model_filename="td3_model.zip",
        kwargs_builder=_td3_kwargs,
    ),
}


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    """Return the registered algorithm spec or raise ``KeyError``."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(
            f"unsupported algorithm {name!r}; supported={list(SUPPORTED_ALGORITHMS)}"
        )
    return _REGISTRY[key]


def list_algorithms() -> tuple[AlgorithmName, ...]:
    """Ordered supported algorithm names."""
    return SUPPORTED_ALGORITHMS
