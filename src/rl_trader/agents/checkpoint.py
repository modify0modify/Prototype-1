"""Checkpoint metadata and load/save stubs.

Every checkpoint must store model metadata, feature definitions, and training
periods. Loaders refuse models whose required features do not match.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from rl_trader.features.engineering import FeatureSpec
from rl_trader.validation.features import FeatureMismatchError, assert_features_match


@dataclass(frozen=True, slots=True)
class TrainingPeriod:
    """Inclusive training window recorded with a checkpoint."""

    start: date
    end: date

    def to_dict(self) -> dict[str, str]:
        """Serialize dates as ISO strings."""
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> TrainingPeriod:
        """Deserialize from ISO date strings."""
        return cls(start=date.fromisoformat(data["start"]), end=date.fromisoformat(data["end"]))


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Metadata persisted beside every model checkpoint."""

    model_id: str
    algorithm: str
    feature_specs: tuple[FeatureSpec, ...]
    training_period: TrainingPeriod
    seed: int
    package_version: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation."""
        return {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "feature_specs": [asdict(spec) for spec in self.feature_specs],
            "training_period": self.training_period.to_dict(),
            "seed": self.seed,
            "package_version": self.package_version,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointMetadata:
        """Build metadata from a decoded JSON object."""
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
        period_raw = data["training_period"]
        if not isinstance(period_raw, dict):
            raise TypeError("training_period must be a mapping")
        seed_raw = data["seed"]
        if not isinstance(seed_raw, int):
            raise TypeError("seed must be an int")
        return cls(
            model_id=str(data["model_id"]),
            algorithm=str(data["algorithm"]),
            feature_specs=specs,
            training_period=TrainingPeriod.from_dict(
                {str(k): str(v) for k, v in period_raw.items()}
            ),
            seed=seed_raw,
            package_version=str(data["package_version"]),
            notes=str(data.get("notes", "")),
        )


METADATA_FILENAME = "metadata.json"


def save_checkpoint_metadata(directory: Path, metadata: CheckpointMetadata) -> Path:
    """Write checkpoint metadata JSON under ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_checkpoint_metadata(directory: Path) -> CheckpointMetadata:
    """Load checkpoint metadata from ``directory``."""
    path = directory / METADATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("checkpoint metadata root must be an object")
    return CheckpointMetadata.from_dict(data)


def load_checkpoint_for_features(
    directory: Path,
    required_features: tuple[FeatureSpec, ...] | list[FeatureSpec],
) -> CheckpointMetadata:
    """Load metadata and refuse the model if required features do not match.

    When ``training_manifest.json`` is present, its feature specs must also
    match ``required_features`` (model / feature manifest alignment).

    Raises
    ------
    FeatureMismatchError
        When checkpoint feature definitions differ from ``required_features``.
    """
    metadata = load_checkpoint_metadata(directory)
    assert_features_match(metadata.feature_specs, required_features)
    manifest_path = directory / "training_manifest.json"
    if manifest_path.exists():
        from rl_trader.agents.manifest import load_training_manifest

        manifest = load_training_manifest(directory)
        assert_features_match(manifest.feature_specs, required_features)
        assert_features_match(manifest.feature_specs, metadata.feature_specs)
        if manifest.scaler_fitted_on != "train":
            raise FeatureMismatchError(
                "incompatible checkpoint: scaler_fitted_on must be 'train' "
                f"(got {manifest.scaler_fitted_on!r})"
            )
        if manifest.model_selected_on != "validation":
            raise FeatureMismatchError(
                "incompatible checkpoint: model_selected_on must be 'validation' "
                f"(got {manifest.model_selected_on!r})"
            )
    return metadata
