"""Provision a local stub approved model for ``--demo`` / fixture smoke runs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    TrainingPeriod,
    save_checkpoint_metadata,
)
from rl_trader.features.engineering import FeatureSpec


def provision_demo_approved_model(
    root: Path,
    *,
    model_id: str = "paper-demo-stub",
    n_assets: int = 2,
    evaluation_period_id: str = "paper-default",
) -> Path:
    """Write checkpoint + approved manifest; return manifest path."""
    root = Path(root)
    ckpt = root / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    specs = (
        FeatureSpec(name="returns_1d", lookback=1, version="1", causal_only=True),
        FeatureSpec(name="vol_20d", lookback=20, version="1", causal_only=True),
    )
    meta = CheckpointMetadata(
        model_id=model_id,
        algorithm="stub",
        feature_specs=specs,
        training_period=TrainingPeriod(start=date(2023, 1, 3), end=date(2023, 6, 30)),
        seed=42,
        package_version="0.1.0-demo",
        notes="Auto-provisioned stub for paper --demo; not a production approval.",
    )
    save_checkpoint_metadata(ckpt, meta)
    (ckpt / "feature_manifest.json").write_text(
        json.dumps(
            {
                "feature_specs": [
                    {
                        "name": s.name,
                        "lookback": s.lookback,
                        "version": s.version,
                        "causal_only": s.causal_only,
                    }
                    for s in specs
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "approved": True,
                "model_id": model_id,
                "model_version": "0.1.0-demo",
                "algorithm": "stub",
                "n_assets": n_assets,
                "checkpoint_dir": str(ckpt),
                "evaluation_period_id": evaluation_period_id,
                "notes": "Demo stub approval — replace with a signed-off model for real paper.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path
