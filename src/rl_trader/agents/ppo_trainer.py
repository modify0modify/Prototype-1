"""PPO training pipeline (Stable-Baselines3) with train/val/test hygiene."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
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
from rl_trader.agents.manifest import (
    PeriodSpec,
    TrainingManifest,
    save_training_manifest,
)
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

MODEL_FILENAME = "ppo_model.zip"
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
    """Artifacts produced by ``train_ppo``."""

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
    """Snapshot reward hyperparameters."""
    return {
        "drawdown_penalty": env.drawdown_penalty,
        "turnover_penalty": env.turnover_penalty,
        "transaction_cost_penalty": env.transaction_cost_penalty,
        "concentration_penalty": env.concentration_penalty,
        "volatility_penalty": env.volatility_penalty,
    }


def cost_assumptions_dict(env: EnvironmentSettings) -> dict[str, Any]:
    """Snapshot transaction-cost assumptions shared with the env."""
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


def _make_vec_env(
    episode: EpisodeData,
    env_settings: EnvironmentSettings,
    *,
    seed: int,
) -> DummyVecEnv:
    def _thunk() -> Monitor:
        env = PortfolioEnv(episode, env_settings, seed=seed)
        return Monitor(env)

    return DummyVecEnv([_thunk])


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
    """Train PPO on the training split; early-stop / select on validation only.

    The test split is never used for fitting, scaling, or model selection. When
    ``training.evaluate_test_after_training`` is True, a single post-hoc test
    evaluation is recorded for reporting only.
    """
    if training.algorithm != "ppo":
        raise ValueError(f"unsupported algorithm: {training.algorithm!r}")

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

    # --- Scaler: TRAIN ONLY ---
    scaler = FeatureScaler()
    if training.normalize_features:
        guard.note_scaler_fit("train")
        train_episode = scaler.fit_transform(splits.train, split="train")
        val_episode = scaler.transform(splits.validation)
    else:
        train_episode = splits.train
        val_episode = splits.validation
        scaler.fit(splits.train, split="train")  # still record train-only stats
        guard.note_scaler_fit("train")

    # --- Envs: train + validation only during fitting ---
    guard.note_env("train")
    guard.note_env("validation")
    train_env = _make_vec_env(train_episode, environment, seed=training.seed)
    val_env = _make_vec_env(val_episode, environment, seed=training.seed + 1)

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

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=training.learning_rate,
        n_steps=training.n_steps,
        batch_size=training.batch_size,
        n_epochs=training.n_epochs,
        gamma=training.gamma,
        gae_lambda=training.gae_lambda,
        clip_range=training.clip_range,
        ent_coef=training.ent_coef,
        vf_coef=training.vf_coef,
        max_grad_norm=training.max_grad_norm,
        seed=training.seed,
        device=training.device,
        verbose=0,
    )
    model.learn(
        total_timesteps=training.total_timesteps,
        callback=eval_callback,
        progress_bar=False,
    )
    timesteps_run = int(model.num_timesteps)
    early_stopped = timesteps_run < training.total_timesteps

    # Persist final + prefer best (validation-selected) weights when available.
    model_path = out / MODEL_FILENAME
    model.save(str(model_path.with_suffix("")))  # SB3 appends .zip
    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        model = PPO.load(str(best_zip), device=training.device)
        model.save(str(model_path.with_suffix("")))

    scaler.save(out)

    # Validation score from EvalCallback history (model selection signal).
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
            # shape (n_evals, n_episodes)
            val_history = [float(np.mean(row)) for row in loaded["results"]]

    guard.model_selected_on = "validation"

    test_mean: float | None = None
    if training.evaluate_test_after_training:
        # Post-hoc reporting only — never used for selection.
        if training.normalize_features:
            test_episode = scaler.transform(splits.test)
        else:
            test_episode = splits.test
        guard.note_env("test")
        test_env = _make_vec_env(test_episode, environment, seed=training.seed + 2)
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
        # Prefer validation end during fitting; only touch test dates after eval.
        data_end = _as_date(splits.validation.timestamps[-1])
        if training.evaluate_test_after_training:
            data_end = _as_date(splits.test.timestamps[-1])

    metadata = CheckpointMetadata(
        model_id=training.model_id,
        algorithm="ppo",
        feature_specs=feature_specs,
        training_period=TrainingPeriod(
            start=training.train_start, end=training.train_end
        ),
        seed=training.seed,
        package_version=__version__,
        notes=(
            "PPO Stable-Baselines3 checkpoint; selected on validation only; "
            "scaler fitted on train only"
        ),
    )
    save_checkpoint_metadata(out, metadata)

    # Full feature manifest document (parameters + specs).
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

    manifest = TrainingManifest(
        model_id=training.model_id,
        algorithm="ppo",
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
        notes="Model selection uses validation only; test is reporting-only.",
        extra={
            "envs_built_for": sorted(guard.envs_built_for),
            "model_path": str(model_path),
            "best_model_dir": str(best_dir),
        },
    )
    save_training_manifest(out, manifest)

    # Evaluation results summary
    eval_summary = {
        "best_val_mean_reward": best_val,
        "val_eval_history": val_history,
        "test_eval_mean_reward": test_mean,
        "model_selected_on": "validation",
        "test_used_for_selection": False,
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
