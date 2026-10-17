#!/usr/bin/env python3
"""Evaluate a checkpoint out-of-sample (research stub)."""

from __future__ import annotations

from pathlib import Path

from rl_trader.agents import load_checkpoint_for_features
from rl_trader.backtesting import evaluate_equity_curve
from rl_trader.config import get_settings
from rl_trader.features import FeatureSpec
from rl_trader.monitoring import configure_logging, get_logger
from rl_trader.validation import FeatureMismatchError


def main() -> int:
    """Load metadata with feature validation; metrics remain stubbed."""
    settings = get_settings(config_file=Path("configs/research.yaml"))
    configure_logging(settings.log_level, json_logs=settings.log_json)
    logger = get_logger("rl_trader.scripts.evaluate_agent")

    required = (FeatureSpec(name="returns_1d", lookback=1),)
    ckpt = Path("models/checkpoints/stub-model")
    try:
        metadata = load_checkpoint_for_features(ckpt, required)
    except (FileNotFoundError, FeatureMismatchError) as exc:
        logger.error("checkpoint load refused: %s", exc)
        return 1

    result = evaluate_equity_curve(None)
    logger.info(
        "evaluate_agent stub model_id=%s strategy=%s (no RL training)",
        metadata.model_id,
        result.strategy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
