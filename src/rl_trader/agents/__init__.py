"""RL agents, multi-algo SB3 training, and checkpoint metadata (research)."""

from rl_trader.agents.base import StubAgent
from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
)
from rl_trader.agents.compare import ComparisonReport, compare_algorithms
from rl_trader.agents.interface import PortfolioAgent, SB3PortfolioAgent
from rl_trader.agents.manifest import TrainingManifest, load_training_manifest
from rl_trader.agents.random_control import RandomControlResult, evaluate_random_policy
from rl_trader.agents.registry import SUPPORTED_ALGORITHMS, get_algorithm_spec, list_algorithms
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import SplitEpisodes, split_episode_by_fractions
from rl_trader.agents.trainer import TrainResult, train_agent, train_ppo

__all__ = [
    "SUPPORTED_ALGORITHMS",
    "CheckpointMetadata",
    "ComparisonReport",
    "FeatureScaler",
    "PortfolioAgent",
    "RandomControlResult",
    "SB3PortfolioAgent",
    "SplitEpisodes",
    "StubAgent",
    "TrainResult",
    "TrainingManifest",
    "TrainingPeriod",
    "compare_algorithms",
    "evaluate_random_policy",
    "get_algorithm_spec",
    "list_algorithms",
    "load_checkpoint_for_features",
    "load_checkpoint_metadata",
    "load_training_manifest",
    "save_checkpoint_metadata",
    "split_episode_by_fractions",
    "train_agent",
    "train_ppo",
]
