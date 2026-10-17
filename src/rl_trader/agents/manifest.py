"""Full training artifact manifest saved with every checkpoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from rl_trader.features.engineering import FeatureSpec

TRAINING_MANIFEST_FILENAME = "training_manifest.json"


@dataclass(frozen=True, slots=True)
class PeriodSpec:
    """Inclusive calendar window."""

    start: date
    end: date

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> PeriodSpec:
        return cls(start=date.fromisoformat(data["start"]), end=date.fromisoformat(data["end"]))


@dataclass(frozen=True, slots=True)
class TrainingManifest:
    """Everything needed to reproduce / validate a trained checkpoint."""

    model_id: str
    algorithm: str
    seed: int
    package_version: str
    universe: tuple[str, ...]
    data_start: date
    data_end: date
    train_period: PeriodSpec
    val_period: PeriodSpec
    test_period: PeriodSpec
    feature_specs: tuple[FeatureSpec, ...]
    feature_parameters: dict[str, Any]
    library_versions: dict[str, str]
    reward_parameters: dict[str, Any]
    transaction_cost_assumptions: dict[str, Any]
    scaler_fitted_on: str
    normalize_features: bool
    total_timesteps_configured: int
    total_timesteps_run: int
    best_val_mean_reward: float | None
    val_eval_history: tuple[float, ...] = ()
    test_eval_mean_reward: float | None = None
    early_stopped: bool = False
    model_selected_on: str = "validation"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "package_version": self.package_version,
            "universe": list(self.universe),
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "train_period": self.train_period.to_dict(),
            "val_period": self.val_period.to_dict(),
            "test_period": self.test_period.to_dict(),
            "feature_specs": [asdict(spec) for spec in self.feature_specs],
            "feature_parameters": self.feature_parameters,
            "library_versions": self.library_versions,
            "reward_parameters": self.reward_parameters,
            "transaction_cost_assumptions": self.transaction_cost_assumptions,
            "scaler_fitted_on": self.scaler_fitted_on,
            "normalize_features": self.normalize_features,
            "total_timesteps_configured": self.total_timesteps_configured,
            "total_timesteps_run": self.total_timesteps_run,
            "best_val_mean_reward": self.best_val_mean_reward,
            "val_eval_history": list(self.val_eval_history),
            "test_eval_mean_reward": self.test_eval_mean_reward,
            "early_stopped": self.early_stopped,
            "model_selected_on": self.model_selected_on,
            "notes": self.notes,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingManifest:
        """Restore from JSON object."""
        raw_specs = data["feature_specs"]
        if not isinstance(raw_specs, list):
            raise TypeError("feature_specs must be a list")
        specs = tuple(
            FeatureSpec(
                name=str(item["name"]),
                lookback=int(item["lookback"]),
                version=str(item.get("version", "1")),
                causal_only=bool(item.get("causal_only", True)),
            )
            for item in raw_specs
            if isinstance(item, dict)
        )
        return cls(
            model_id=str(data["model_id"]),
            algorithm=str(data["algorithm"]),
            seed=int(data["seed"]),
            package_version=str(data["package_version"]),
            universe=tuple(str(s) for s in data["universe"]),
            data_start=date.fromisoformat(str(data["data_start"])),
            data_end=date.fromisoformat(str(data["data_end"])),
            train_period=PeriodSpec.from_dict(data["train_period"]),
            val_period=PeriodSpec.from_dict(data["val_period"]),
            test_period=PeriodSpec.from_dict(data["test_period"]),
            feature_specs=specs,
            feature_parameters=dict(data.get("feature_parameters") or {}),
            library_versions=dict(data.get("library_versions") or {}),
            reward_parameters=dict(data.get("reward_parameters") or {}),
            transaction_cost_assumptions=dict(
                data.get("transaction_cost_assumptions") or {}
            ),
            scaler_fitted_on=str(data.get("scaler_fitted_on", "train")),
            normalize_features=bool(data.get("normalize_features", True)),
            total_timesteps_configured=int(data["total_timesteps_configured"]),
            total_timesteps_run=int(data["total_timesteps_run"]),
            best_val_mean_reward=(
                None
                if data.get("best_val_mean_reward") is None
                else float(data["best_val_mean_reward"])
            ),
            val_eval_history=tuple(float(x) for x in data.get("val_eval_history") or ()),
            test_eval_mean_reward=(
                None
                if data.get("test_eval_mean_reward") is None
                else float(data["test_eval_mean_reward"])
            ),
            early_stopped=bool(data.get("early_stopped", False)),
            model_selected_on=str(data.get("model_selected_on", "validation")),
            notes=str(data.get("notes", "")),
            extra=dict(data.get("extra") or {}),
        )


def save_training_manifest(directory: Path, manifest: TrainingManifest) -> Path:
    """Write ``training_manifest.json`` under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRAINING_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_training_manifest(directory: Path) -> TrainingManifest:
    """Load training manifest from ``directory``."""
    path = directory / TRAINING_MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing training manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training manifest root must be an object")
    return TrainingManifest.from_dict(data)
