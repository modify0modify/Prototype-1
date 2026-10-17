"""RL agents, PPO training, and checkpoint metadata (research; no brokers)."""

from rl_trader.agents.base import StubAgent
from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
)
from rl_trader.agents.manifest import TrainingManifest, load_training_manifest
from rl_trader.agents.ppo_trainer import TrainResult, train_ppo
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import SplitEpisodes, split_episode_by_fractions

__all__ = [
    "CheckpointMetadata",
    "FeatureScaler",
    "SplitEpisodes",
    "StubAgent",
    "TrainResult",
    "TrainingManifest",
    "TrainingPeriod",
    "load_checkpoint_for_features",
    "load_checkpoint_metadata",
    "load_training_manifest",
    "save_checkpoint_metadata",
    "split_episode_by_fractions",
    "train_ppo",
]
