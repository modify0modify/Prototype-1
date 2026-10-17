"""Unit tests for checkpoint metadata + feature mismatch refusal."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from rl_trader import __version__
from rl_trader.agents import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    save_checkpoint_metadata,
)
from rl_trader.features import FeatureSpec
from rl_trader.validation import FeatureMismatchError


def _metadata(*features: FeatureSpec) -> CheckpointMetadata:
    return CheckpointMetadata(
        model_id="unit-test-model",
        algorithm="stub",
        feature_specs=features,
        training_period=TrainingPeriod(start=date(2021, 1, 1), end=date(2021, 12, 31)),
        seed=7,
        package_version=__version__,
    )


def test_load_succeeds_when_features_match(tmp_path: Path) -> None:
    required = (
        FeatureSpec(name="returns_1d", lookback=1, version="1"),
        FeatureSpec(name="vol_20d", lookback=20, version="1"),
    )
    save_checkpoint_metadata(tmp_path, _metadata(*required))
    loaded = load_checkpoint_for_features(tmp_path, required)
    assert loaded.model_id == "unit-test-model"
    assert loaded.feature_specs == required
    assert loaded.training_period.start == date(2021, 1, 1)


def test_load_refuses_feature_mismatch(tmp_path: Path) -> None:
    saved = (FeatureSpec(name="returns_1d", lookback=1),)
    required = (FeatureSpec(name="returns_1d", lookback=5),)
    save_checkpoint_metadata(tmp_path, _metadata(*saved))
    with pytest.raises(FeatureMismatchError, match="do not match"):
        load_checkpoint_for_features(tmp_path, required)


def test_load_refuses_missing_feature(tmp_path: Path) -> None:
    saved = (FeatureSpec(name="returns_1d", lookback=1),)
    required = (
        FeatureSpec(name="returns_1d", lookback=1),
        FeatureSpec(name="vol_20d", lookback=20),
    )
    save_checkpoint_metadata(tmp_path, _metadata(*saved))
    with pytest.raises(FeatureMismatchError):
        load_checkpoint_for_features(tmp_path, required)
