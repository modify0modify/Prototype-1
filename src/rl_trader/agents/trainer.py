"""Shared SB3 training pipeline (PPO / SAC / TD3) with train/val/test hygiene."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_trader import __version__
from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    save_checkpoint_metadata,
)
from rl_trader.agents.features_for_training import feature_specs_from_episode
from rl_trader.agents.interface import describe_action_space
from rl_trader.agents.manifest import (
    PeriodSpec,
    TrainingManifest,
    save_training_manifest,
)
from rl_trader.agents.registry import get_algorithm_spec
from rl_trader.agents.scaling import FeatureScaler
from rl_trader.agents.splits import SplitEpisodes
from rl_trader.config.models import (
    EnvironmentSettings,
    FeatureSettings,
    TrainingSettings,
)
from rl_trader.environment.episode import EpisodeData
from rl_trader.environment.portfolio_env import PortfolioEnv
from rl_trader.utils.seeds import set_global_seeds

BEST_MODEL_DIRNAME = "best"
EVAL_LOG_DIRNAME = "eval_logs"


@dataclass
class SplitAccessGuard:
    """Records which split panels are materialized into envs / scalers."""

    fitted_scaler_on: str | None = None
    envs_built_for: set[str] = field(default_factory=set)
    model_selected_on: str = "validation"
    test_eval_completed: bool = False

    def note_scaler_fit(self, split: str) -> None:
        if split != "train":
            raise RuntimeError(f"scaler fit refused on split={split!r}")
        self.fitted_scaler_on = split

    def note_env(self, split: str) -> None:
        self.envs_built_for.add(split)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Artifacts produced by ``train_agent``."""

    output_dir: Path
    metadata: CheckpointMetadata
    manifest: TrainingManifest
    best_val_mean_reward: float | None
    test_mean_reward: float | None
    access_guard: SplitAccessGuard


def collect_library_versions() -> dict[str, str]:
    """Pin-relevant library versions for reproducibility metadata."""
    versions: dict[str, str] = {"rl_trader": __version__}
    for name in (
        "stable_baselines3",
        "gymnasium",
        "torch",
        "numpy",
        "sklearn",
        "pandas",
    ):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:  # pragma: no cover
            versions[name] = "missing"
    return versions


def reward_params_dict(env: EnvironmentSettings) -> dict[str, Any]:
    """Snapshot reward hyperparameters (identical across algorithms)."""
    return {
        "drawdown_penalty": env.drawdown_penalty,
        "turnover_penalty": env.turnover_penalty,
        "transaction_cost_penalty": env.transaction_cost_penalty,
        "concentration_penalty": env.concentration_penalty,
        "volatility_penalty": env.volatility_penalty,
    }


def cost_assumptions_dict(env: EnvironmentSettings) -> dict[str, Any]:
    """Snapshot transaction-cost assumptions (identical across algorithms)."""
    return {
        "commission_bps": env.commission_bps,
        "bid_ask_spread_bps": env.bid_ask_spread_bps,
        "slippage_bps": env.slippage_bps,
        "market_impact_coeff": env.market_impact_coeff,
        "execution_delay_bars": env.execution_delay_bars,
        "max_weight": env.max_weight,
        "initial_cash": env.initial_cash,
    }


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[no-any-return]
    return date.fromisoformat(str(value)[:10])


def make_vec_env(
    episode: EpisodeData,
    env_settings: EnvironmentSettings,
    *,
    seed: int,
) -> DummyVecEnv:
    """Build a Monitor-wrapped DummyVecEnv for training or evaluation."""

    def _thunk() -> Monitor:
        env = PortfolioEnv(episode, env_settings, seed=seed)
        return Monitor(env)

    return DummyVecEnv([_thunk])


def train_agent(
    splits: SplitEpisodes,
    *,
    training: TrainingSettings,
    environment: EnvironmentSettings,
    features: FeatureSettings | None = None,
    full_episode_dates: tuple[date, date] | None = None,
    output_dir: Path | None = None,
    access_guard: SplitAccessGuard | None = None,
    algorithm: str | None = None,
) -> TrainResult:
    """Train PPO/SAC/TD3 on train; early-stop / select on validation only.

    Identical data splits, cost assumptions, and reward definitions are used
    for every algorithm. The test split is never used for fitting or selection.
    """
    algo_name = (algorithm or training.algorithm).lower()
    spec = get_algorithm_spec(algo_name)
    training = training.model_copy(update={"algorithm": spec.name})

    guard = access_guard or SplitAccessGuard()
    set_global_seeds(training.seed)

    out = Path(output_dir or (training.checkpoint_dir / training.model_id))
    out.mkdir(parents=True, exist_ok=True)
    best_dir = out / BEST_MODEL_DIRNAME
    eval_log_dir = out / EVAL_LOG_DIRNAME
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    feature_settings = features or FeatureSettings()
    feature_specs = feature_specs_from_episode(splits.train, feature_settings)
    algo_hparams = spec.hyperparameters(training)

    # --- Scaler: TRAIN ONLY ---
    scaler = FeatureScaler()
    if training.normalize_features:
        guard.note_scaler_fit("train")
        train_episode = scaler.fit_transform(splits.train, split="train")
        val_episode = scaler.transform(splits.validation)
    else:
        train_episode = splits.train
        val_episode = splits.validation
        scaler.fit(splits.train, split="train")
        guard.note_scaler_fit("train")

    # --- Envs: train + validation only during fitting ---
    guard.note_env("train")
    guard.note_env("validation")
    train_env = make_vec_env(train_episode, environment, seed=training.seed)
    val_env = make_vec_env(val_episode, environment, seed=training.seed + 1)

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=training.early_stopping_patience,
        min_evals=training.min_evals_before_stop,
        verbose=0,
    )
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_log_dir),
        eval_freq=max(1, training.eval_freq),
        n_eval_episodes=training.n_eval_episodes,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=0,
    )

    model = spec.build(train_env, training)
    model.learn(
        total_timesteps=training.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )
    timesteps_run = int(model.num_timesteps)
    early_stopped = timesteps_run < training.total_timesteps

    model_path = out / spec.model_filename
    model.save(str(model_path.with_suffix("")))
    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        model = spec.sb3_cls.load(str(best_zip), device=training.device)
        model.save(str(model_path.with_suffix("")))

    scaler.save(out)
    action_space_info = describe_action_space(model)

    best_val = (
        float(eval_callback.best_mean_reward)
        if np.isfinite(eval_callback.best_mean_reward)
        else None
    )
    val_history: list[float] = []
    results_npz = eval_log_dir / "evaluations.npz"
    if results_npz.exists():
        loaded = np.load(results_npz)
        if "results" in loaded:
            val_history = [float(np.mean(row)) for row in loaded["results"]]

    guard.model_selected_on = "validation"

    test_mean: float | None = None
    if training.evaluate_test_after_training:
        if training.normalize_features:
            test_episode = scaler.transform(splits.test)
        else:
            test_episode = splits.test
        guard.note_env("test")
        test_env = make_vec_env(test_episode, environment, seed=training.seed + 2)
        mean_r, _std_r = evaluate_policy(
            model,
            test_env,
            n_eval_episodes=training.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=False,
        )
        test_mean = float(mean_r)  # type: ignore[arg-type]
        guard.test_eval_completed = True
        test_env.close()

    train_env.close()
    val_env.close()

    if full_episode_dates is not None:
        data_start, data_end = full_episode_dates
    else:
        data_start = _as_date(splits.train.timestamps[0])
        data_end = _as_date(splits.validation.timestamps[-1])
        if training.evaluate_test_after_training:
            data_end = _as_date(splits.test.timestamps[-1])

    metadata = CheckpointMetadata(
        model_id=training.model_id,
        algorithm=spec.name,
        feature_specs=feature_specs,
        training_period=TrainingPeriod(
            start=training.train_start, end=training.train_end
        ),
        seed=training.seed,
        package_version=__version__,
        notes=(
            f"{spec.name.upper()} Stable-Baselines3 checkpoint; "
            "selected on validation only; scaler fitted on train only"
        ),
    )
    save_checkpoint_metadata(out, metadata)

    feature_manifest_doc = {
        "feature_version": feature_settings.feature_version,
        "causal_only": feature_settings.causal_only,
        "parameters": feature_settings.model_dump(mode="json"),
        "features": [asdict(s) for s in feature_specs],
        "scaler_fitted_on": "train",
        "normalize_features": training.normalize_features,
    }
    (out / "feature_manifest.json").write_text(
        json.dumps(feature_manifest_doc, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    (out / "algorithm_hyperparameters.json").write_text(
        json.dumps(algo_hparams, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = TrainingManifest(
        model_id=training.model_id,
        algorithm=spec.name,
        seed=training.seed,
        package_version=__version__,
        universe=tuple(splits.train.symbols),
        data_start=data_start,
        data_end=data_end,
        train_period=PeriodSpec(start=training.train_start, end=training.train_end),
        val_period=PeriodSpec(start=training.val_start, end=training.val_end),
        test_period=PeriodSpec(start=training.test_start, end=training.test_end),
        feature_specs=feature_specs,
        feature_parameters=feature_settings.model_dump(mode="json"),
        library_versions=collect_library_versions(),
        reward_parameters=reward_params_dict(environment),
        transaction_cost_assumptions=cost_assumptions_dict(environment),
        scaler_fitted_on="train",
        normalize_features=training.normalize_features,
        total_timesteps_configured=training.total_timesteps,
        total_timesteps_run=timesteps_run,
        best_val_mean_reward=best_val,
        val_eval_history=tuple(val_history),
        test_eval_mean_reward=test_mean,
        early_stopped=early_stopped,
        model_selected_on="validation",
        algorithm_hyperparameters=algo_hparams,
        notes="Model selection uses validation only; test is reporting-only.",
        extra={
            "envs_built_for": sorted(guard.envs_built_for),
            "model_path": str(model_path),
            "best_model_dir": str(best_dir),
            "action_space": action_space_info,
        },
    )
    save_training_manifest(out, manifest)

    eval_summary = {
        "algorithm": spec.name,
        "best_val_mean_reward": best_val,
        "val_eval_history": val_history,
        "test_eval_mean_reward": test_mean,
        "model_selected_on": "validation",
        "test_used_for_selection": False,
        "algorithm_hyperparameters": algo_hparams,
    }
    (out / "evaluation_results.json").write_text(
        json.dumps(eval_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TrainResult(
        output_dir=out,
        metadata=metadata,
        manifest=manifest,
        best_val_mean_reward=best_val,
        test_mean_reward=test_mean,
        access_guard=guard,
    )


def train_ppo(
    splits: SplitEpisodes,
    *,
    training: TrainingSettings,
    environment: EnvironmentSettings,
    features: FeatureSettings | None = None,
    full_episode_dates: tuple[date, date] | None = None,
    output_dir: Path | None = None,
    access_guard: SplitAccessGuard | None = None,
) -> TrainResult:
    """Backward-compatible PPO entrypoint."""
    return train_agent(
        splits,
        training=training.model_copy(update={"algorithm": "ppo"}),
        environment=environment,
        features=features,
        full_episode_dates=full_episode_dates,
        output_dir=output_dir,
        access_guard=access_guard,
        algorithm="ppo",
    )
