"""Backward-compatible re-exports — prefer :mod:`rl_trader.agents.trainer`."""

from __future__ import annotations

from rl_trader.agents.trainer import (
    BEST_MODEL_DIRNAME,
    EVAL_LOG_DIRNAME,
    SplitAccessGuard,
    TrainResult,
    collect_library_versions,
    cost_assumptions_dict,
    make_vec_env,
    reward_params_dict,
    train_agent,
    train_ppo,
)

__all__ = [
    "BEST_MODEL_DIRNAME",
    "EVAL_LOG_DIRNAME",
    "SplitAccessGuard",
    "TrainResult",
    "collect_library_versions",
    "cost_assumptions_dict",
    "make_vec_env",
    "reward_params_dict",
    "train_agent",
    "train_ppo",
]
