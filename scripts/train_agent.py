#!/usr/bin/env python3
"""Train an RL agent (research stub — no real training yet)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from rl_trader import __version__
from rl_trader.agents import CheckpointMetadata, TrainingPeriod, save_checkpoint_metadata
from rl_trader.config import get_settings
from rl_trader.features import FeatureSpec
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.utils.seeds import set_global_seeds


def main() -> int:
    """Persist example checkpoint metadata only."""
    settings = get_settings(config_file=Path("configs/research.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    set_global_seeds(settings.seed)
    logger = get_logger("rl_trader.scripts.train_agent")

    out = Path("models/checkpoints/stub-model")
    metadata = CheckpointMetadata(
        model_id="stub-model",
        algorithm="stub",
        feature_specs=(FeatureSpec(name="returns_1d", lookback=1),),
        training_period=TrainingPeriod(start=date(2020, 1, 1), end=date(2022, 12, 31)),
        seed=settings.seed,
        package_version=__version__,
        notes="train_agent stub — no weights trained",
    )
    path = save_checkpoint_metadata(out, metadata)
    logger.info("wrote checkpoint metadata to %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
