"""Load only approved model candidates for paper trading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_trader.agents.base import StubAgent
from rl_trader.agents.checkpoint import (
    CheckpointMetadata,
    load_checkpoint_for_features,
    load_checkpoint_metadata,
)
from rl_trader.agents.interface import PortfolioAgent, SB3PortfolioAgent
from rl_trader.features.engineering import FeatureSpec
from rl_trader.validation.features import FeatureMismatchError


class ApprovedModelError(RuntimeError):
    """Raised when the approved-model gate fails."""


@dataclass(frozen=True, slots=True)
class ApprovedModelBundle:
    """Frozen model bundle for one evaluation period."""

    model_id: str
    model_version: str
    algorithm: str
    checkpoint_dir: Path
    feature_specs: tuple[FeatureSpec, ...]
    agent: PortfolioAgent
    evaluation_period_id: str
    metadata: CheckpointMetadata | None
    manifest_path: Path


def load_approved_manifest(path: Path) -> dict[str, Any]:
    """Load and validate an approved-model manifest document."""
    if not path.is_file():
        raise ApprovedModelError(f"approved model manifest missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovedModelError(f"approved model manifest unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ApprovedModelError("approved model manifest root must be an object")
    if data.get("approved") is not True:
        raise ApprovedModelError(
            f"model not approved in manifest model_id={data.get('model_id', '<unknown>')}"
        )
    if not data.get("model_id"):
        raise ApprovedModelError("approved manifest requires model_id")
    return data


def _feature_specs_from_manifest_or_checkpoint(
    manifest: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[FeatureSpec, ...]:
    raw = manifest.get("feature_specs")
    if isinstance(raw, list) and raw:
        specs = tuple(
            FeatureSpec(
                name=str(item["name"]),
                lookback=int(item["lookback"]),
                version=str(item.get("version", "1")),
                causal_only=bool(item.get("causal_only", True)),
            )
            for item in raw
            if isinstance(item, dict)
        )
        if specs:
            return specs
    # Fall back to checkpoint metadata / feature_manifest.json
    feature_manifest = checkpoint_dir / "feature_manifest.json"
    if feature_manifest.is_file():
        doc = json.loads(feature_manifest.read_text(encoding="utf-8"))
        items = doc.get("feature_specs") if isinstance(doc, dict) else None
        if isinstance(items, list) and items:
            return tuple(
                FeatureSpec(
                    name=str(item["name"]),
                    lookback=int(item["lookback"]),
                    version=str(item.get("version", "1")),
                    causal_only=bool(item.get("causal_only", True)),
                )
                for item in items
                if isinstance(item, dict)
            )
    meta = load_checkpoint_metadata(checkpoint_dir)
    return meta.feature_specs


def load_approved_model(
    manifest_path: Path,
    *,
    evaluation_period_id: str,
    device: str = "cpu",
) -> ApprovedModelBundle:
    """Resolve an approved manifest → verified checkpoint → frozen agent.

    Supports ``algorithm: stub`` (tests / smoke) and SB3 algorithms via
    :class:`SB3PortfolioAgent`. The returned agent is intended to stay frozen
    for the evaluation period — callers must not retrain it in-loop.
    """
    manifest = load_approved_manifest(manifest_path)
    model_id = str(manifest["model_id"])
    algorithm = str(manifest.get("algorithm") or "stub").lower()
    ckpt_raw = manifest.get("checkpoint_dir")
    if not ckpt_raw:
        raise ApprovedModelError("approved manifest requires checkpoint_dir")
    checkpoint_dir = Path(str(ckpt_raw))
    if not checkpoint_dir.is_dir():
        raise ApprovedModelError(f"checkpoint_dir missing: {checkpoint_dir}")

    required = _feature_specs_from_manifest_or_checkpoint(manifest, checkpoint_dir)
    try:
        metadata = load_checkpoint_for_features(checkpoint_dir, required)
    except (FileNotFoundError, FeatureMismatchError, TypeError, ValueError) as exc:
        raise ApprovedModelError(f"feature manifest verification failed: {exc}") from exc

    model_version = str(
        manifest.get("model_version")
        or manifest.get("version")
        or metadata.package_version
        or "1"
    )

    if algorithm == "stub":
        n_assets = int(manifest.get("n_assets") or 0)
        if n_assets < 1:
            # Infer from action space notes / feature dims is unavailable — require.
            raise ApprovedModelError("stub approved model requires n_assets >= 1")
        agent: PortfolioAgent = StubAgent(n_assets=n_assets)
    else:
        agent = SB3PortfolioAgent.load(
            checkpoint_dir,
            algorithm,
            device=device,
            deterministic=True,
        )

    return ApprovedModelBundle(
        model_id=model_id,
        model_version=model_version,
        algorithm=algorithm,
        checkpoint_dir=checkpoint_dir,
        feature_specs=metadata.feature_specs,
        agent=agent,
        evaluation_period_id=evaluation_period_id,
        metadata=metadata,
        manifest_path=manifest_path,
    )
