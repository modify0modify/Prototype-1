"""Feature manifest persistence (definitions + calculation parameters)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_trader.config.models import FeatureSettings
from rl_trader.features.engineering import FeatureSpec


@dataclass(frozen=True, slots=True)
class FeatureManifest:
    """Manifest stored with every feature matrix artifact."""

    symbol: str
    feature_version: str
    causal_only: bool
    generated_at: str
    parameters: dict[str, Any]
    features: tuple[FeatureSpec, ...]
    n_rows: int
    n_feature_columns: int
    row_policy: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable document."""
        return {
            "symbol": self.symbol,
            "feature_version": self.feature_version,
            "causal_only": self.causal_only,
            "generated_at": self.generated_at,
            "parameters": self.parameters,
            "features": [asdict(spec) for spec in self.features],
            "n_rows": self.n_rows,
            "n_feature_columns": self.n_feature_columns,
            "row_policy": self.row_policy,
            "notes": self.notes,
        }


def settings_to_parameters(settings: FeatureSettings) -> dict[str, Any]:
    """Flatten feature calculation parameters for the manifest."""
    data = settings.model_dump(mode="json")
    # Paths as strings already via mode=json
    return data


def build_manifest(
    *,
    symbol: str,
    settings: FeatureSettings,
    specs: list[FeatureSpec],
    n_rows: int,
    n_feature_columns: int,
) -> FeatureManifest:
    """Assemble a manifest document."""
    return FeatureManifest(
        symbol=symbol.upper(),
        feature_version=settings.feature_version,
        causal_only=settings.causal_only,
        generated_at=datetime.now(UTC).isoformat(),
        parameters=settings_to_parameters(settings),
        features=tuple(specs),
        n_rows=n_rows,
        n_feature_columns=n_feature_columns,
        row_policy=(
            "keep_all_rows_with_explicit_nans; never_drop_silently; "
            "warmup_and_missing_fundamentals_are_nan"
        ),
        notes="Features at t use only information with timestamps <= t.",
    )


def write_manifest(manifest: FeatureManifest, directory: Path) -> Path:
    """Write ``{symbol}_feature_manifest.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{manifest.symbol}_feature_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
