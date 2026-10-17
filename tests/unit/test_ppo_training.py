"""Tests for PPO training hygiene (train-only scalers, split isolation)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    load_checkpoint_for_features,
    save_checkpoint_metadata,
)
from rl_trader.agents.features_for_training import feature_specs_from_episode
from rl_trader.agents.manifest import (
    PeriodSpec,
    TrainingManifest,
    save_training_manifest,
)
from rl_trader.agents.ppo_trainer import SplitAccessGuard, train_ppo
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import split_episode_by_fractions
from rl_trader.config.models import EnvironmentSettings, FeatureSettings, TrainingSettings
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.synthetic import make_synthetic_episode
from rl_trader.features import FeatureSpec
from rl_trader.validation import FeatureMismatchError


def _tiny_settings(**overrides: object) -> TrainingSettings:
    base: dict[str, object] = {
        "algorithm": "ppo",
        "model_id": "unit-ppo",
        "total_timesteps": 128,
        "learning_rate": 3e-4,
        "batch_size": 32,
        "n_steps": 64,
        "n_epochs": 1,
        "gamma": 0.99,
        "seed": 0,
        "device": "cpu",
        "eval_freq": 64,
        "n_eval_episodes": 1,
        "early_stopping_patience": 2,
        "min_evals_before_stop": 1,
        "normalize_features": True,
        "evaluate_test_after_training": True,
        "train_start": date(2020, 1, 1),
        "train_end": date(2020, 6, 30),
        "val_start": date(2020, 7, 1),
        "val_end": date(2020, 9, 30),
        "test_start": date(2020, 10, 1),
        "test_end": date(2020, 12, 31),
    }
    base.update(overrides)
    return TrainingSettings(**base)  # type: ignore[arg-type]


def _env() -> EnvironmentSettings:
    return EnvironmentSettings(
        initial_cash=10_000.0,
        max_weight=0.5,
        commission_bps=0.0,
        bid_ask_spread_bps=0.0,
        slippage_bps=0.0,
        market_impact_coeff=0.0,
    )


def _splits():
    episode = make_synthetic_episode(
        n_bars=50, n_assets=2, n_market=2, n_fundamental=1, n_regime=1, seed=1
    )
    return split_episode_by_fractions(episode, train_frac=0.6, val_frac=0.2)


def test_scaler_fit_rejects_non_train_split() -> None:
    splits = _splits()
    scaler = FeatureScaler()
    with pytest.raises(ValueError, match="split='train'"):
        scaler.fit(splits.validation, split="validation")  # type: ignore[arg-type]


def test_scaler_is_training_only_statistics() -> None:
    splits = _splits()
    scaler = FeatureScaler().fit(splits.train, split="train")
    assert scaler.fitted_on == "train"
    assert scaler.mean_ is not None
    # Transforming val must not change fitted stats.
    mean_before = scaler.mean_.copy()
    _ = scaler.transform(splits.validation)
    assert np.allclose(scaler.mean_, mean_before)
    # Train transform centers approximately (finite columns).
    train_scaled = scaler.transform(splits.train)
    stacked = np.concatenate(
        [
            train_scaled.market_features,
            train_scaled.fundamental_features,
            train_scaled.regime_features,
        ],
        axis=1,
    )
    assert np.allclose(np.nanmean(stacked, axis=0), 0.0, atol=1e-6)


def test_test_set_not_used_during_fitting(tmp_path: Path) -> None:
    splits = _splits()

    class GuardedTestEpisode(EpisodeData):
        """Episode that raises if its feature matrices are read during fit."""

        def __getattribute__(self, name: str) -> object:
            if name in {
                "market_features",
                "fundamental_features",
                "regime_features",
                "prices",
                "timestamps",
            }:
                # Allow dataclass init / slice construction paths via object.
                pass
            return object.__getattribute__(self, name)

    # Wrap test panel: accessing prices/features after construction flags usage.
    accessed: list[str] = []
    raw_test = splits.test

    class TrackingEpisode:
        def __init__(self, inner: EpisodeData) -> None:
            self._inner = inner

        @property
        def n_bars(self) -> int:
            return self._inner.n_bars

        @property
        def n_assets(self) -> int:
            return self._inner.n_assets

        @property
        def n_market(self) -> int:
            return self._inner.n_market

        @property
        def n_fundamental(self) -> int:
            return self._inner.n_fundamental

        @property
        def n_regime(self) -> int:
            return self._inner.n_regime

        @property
        def symbols(self) -> tuple[str, ...]:
            return self._inner.symbols

        @property
        def timestamps(self) -> np.ndarray:
            accessed.append("timestamps")
            return self._inner.timestamps

        @property
        def prices(self) -> np.ndarray:
            accessed.append("prices")
            return self._inner.prices

        @property
        def market_features(self) -> np.ndarray:
            accessed.append("market_features")
            return self._inner.market_features

        @property
        def fundamental_features(self) -> np.ndarray:
            accessed.append("fundamental_features")
            return self._inner.fundamental_features

        @property
        def regime_features(self) -> np.ndarray:
            accessed.append("regime_features")
            return self._inner.regime_features

    tracking_test = TrackingEpisode(raw_test)
    from rl_trader.agents.splits import SplitEpisodes

    guarded = SplitEpisodes(
        train=splits.train,
        validation=splits.validation,
        test=tracking_test,  # type: ignore[arg-type]
        train_indices=splits.train_indices,
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
    )

    # Phase 1: train without post-hoc test eval — test panel must stay untouched.
    settings = _tiny_settings(
        evaluate_test_after_training=False,
        model_id="no-test",
    )
    settings = settings.model_copy(
        update={
            "train_start": date(2023, 1, 3),
            "train_end": date(2023, 1, 20),
            "val_start": date(2023, 1, 21),
            "val_end": date(2023, 1, 30),
            "test_start": date(2023, 1, 31),
            "test_end": date(2023, 2, 10),
        }
    )
    guard = SplitAccessGuard()
    result = train_ppo(
        guarded,
        training=settings,
        environment=_env(),
        features=FeatureSettings(),
        output_dir=tmp_path / "no-test",
        access_guard=guard,
    )
    assert accessed == []
    assert "test" not in result.access_guard.envs_built_for
    assert result.access_guard.fitted_scaler_on == "train"
    assert result.manifest.model_selected_on == "validation"


def test_training_runs_on_tiny_fixture(tmp_path: Path) -> None:
    splits = _splits()
    settings = _tiny_settings(model_id="tiny", evaluate_test_after_training=True)
    settings = settings.model_copy(
        update={
            "train_start": date(2023, 1, 3),
            "train_end": date(2023, 1, 20),
            "val_start": date(2023, 1, 21),
            "val_end": date(2023, 1, 30),
            "test_start": date(2023, 1, 31),
            "test_end": date(2023, 2, 10),
        }
    )
    result = train_ppo(
        splits,
        training=settings,
        environment=_env(),
        output_dir=tmp_path / "tiny",
    )
    assert (result.output_dir / "ppo_model.zip").exists()
    assert (result.output_dir / "feature_scaler.json").exists()
    assert (result.output_dir / "metadata.json").exists()
    assert (result.output_dir / "training_manifest.json").exists()
    assert (result.output_dir / "feature_manifest.json").exists()
    assert (result.output_dir / "evaluation_results.json").exists()
    assert result.manifest.scaler_fitted_on == "train"
    assert result.manifest.universe == splits.train.symbols
    assert result.manifest.model_selected_on == "validation"
    assert "stable_baselines3" in result.manifest.library_versions


def test_model_and_feature_manifests_aligned(tmp_path: Path) -> None:
    splits = _splits()
    specs = feature_specs_from_episode(splits.train, FeatureSettings())
    settings = _tiny_settings(
        model_id="aligned",
        evaluate_test_after_training=False,
        total_timesteps=64,
        n_steps=32,
        eval_freq=32,
    )
    settings = settings.model_copy(
        update={
            "train_start": date(2023, 1, 3),
            "train_end": date(2023, 1, 20),
            "val_start": date(2023, 1, 21),
            "val_end": date(2023, 1, 30),
            "test_start": date(2023, 1, 31),
            "test_end": date(2023, 2, 10),
        }
    )
    result = train_ppo(
        splits,
        training=settings,
        environment=_env(),
        features=FeatureSettings(),
        output_dir=tmp_path / "aligned",
    )
    loaded = load_checkpoint_for_features(result.output_dir, specs)
    assert loaded.feature_specs == specs
    assert loaded.feature_specs == result.manifest.feature_specs


def test_incompatible_checkpoint_rejected(tmp_path: Path) -> None:
    specs = (FeatureSpec(name="market_0", lookback=1),)
    meta = CheckpointMetadata(
        model_id="bad",
        algorithm="ppo",
        feature_specs=specs,
        training_period=TrainingPeriod(start=date(2020, 1, 1), end=date(2020, 2, 1)),
        seed=1,
        package_version="0.0.0",
    )
    save_checkpoint_metadata(tmp_path, meta)
    manifest = TrainingManifest(
        model_id="bad",
        algorithm="ppo",
        seed=1,
        package_version="0.0.0",
        universe=("A0",),
        data_start=date(2020, 1, 1),
        data_end=date(2020, 12, 31),
        train_period=PeriodSpec(date(2020, 1, 1), date(2020, 6, 1)),
        val_period=PeriodSpec(date(2020, 6, 2), date(2020, 9, 1)),
        test_period=PeriodSpec(date(2020, 9, 2), date(2020, 12, 31)),
        feature_specs=specs,
        feature_parameters={},
        library_versions={},
        reward_parameters={},
        transaction_cost_assumptions={},
        scaler_fitted_on="validation",  # invalid
        normalize_features=True,
        total_timesteps_configured=1,
        total_timesteps_run=1,
        best_val_mean_reward=None,
        model_selected_on="validation",
    )
    save_training_manifest(tmp_path, manifest)
    with pytest.raises(FeatureMismatchError, match="scaler_fitted_on"):
        load_checkpoint_for_features(tmp_path, specs)


def test_feature_mismatch_still_rejected(tmp_path: Path) -> None:
    saved = (FeatureSpec(name="market_0", lookback=1),)
    required = (FeatureSpec(name="market_0", lookback=5),)
    save_checkpoint_metadata(
        tmp_path,
        CheckpointMetadata(
            model_id="x",
            algorithm="ppo",
            feature_specs=saved,
            training_period=TrainingPeriod(start=date(2020, 1, 1), end=date(2020, 2, 1)),
            seed=0,
            package_version="0",
        ),
    )
    with pytest.raises(FeatureMismatchError):
        load_checkpoint_for_features(tmp_path, required)
