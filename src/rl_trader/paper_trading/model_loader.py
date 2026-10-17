"""Load only formally approved model candidates for paper trading."""

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
from rl_trader.approval.service import ApprovalRejectedError, verify_paper_approval
from rl_trader.config.models import ApprovalSettings
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
    content_hash: str = ""
    approval_status: str = "paper"


def load_approved_manifest(
    path: Path,
    *,
    approval_settings: ApprovalSettings | None = None,
) -> dict[str, Any]:
    """Load and formally verify an approved-model manifest (hash + paper gates)."""
    settings = approval_settings or ApprovalSettings()
    try:
        verified = verify_paper_approval(path, settings)
    except ApprovalRejectedError as exc:
        raise ApprovedModelError(str(exc)) from exc
    return verified.manifest


def _feature_specs_from_manifest_or_checkpoint(
    manifest: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[FeatureSpec, ...]:
    fm = manifest.get("feature_manifest")
    if isinstance(fm, dict):
        raw = fm.get("feature_specs")
        if isinstance(raw, list) and raw:
            return tuple(
                FeatureSpec(
                    name=str(item["name"]),
                    lookback=int(item["lookback"]),
                    version=str(item.get("version", "1")),
                    causal_only=bool(item.get("causal_only", True)),
                )
                for item in raw
                if isinstance(item, dict)
            )
    raw = manifest.get("feature_specs")
    if isinstance(raw, list) and raw:
        return tuple(
            FeatureSpec(
                name=str(item["name"]),
                lookback=int(item["lookback"]),
                version=str(item.get("version", "1")),
                causal_only=bool(item.get("causal_only", True)),
            )
            for item in raw
            if isinstance(item, dict)
        )
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
    approval_settings: ApprovalSettings | None = None,
) -> ApprovedModelBundle:
    """Resolve a formally approved manifest → verified checkpoint → frozen agent.

    Unapproved, tampered, incomplete, or gate-failing manifests are rejected.
    """
    settings = approval_settings or ApprovalSettings()
    try:
        verified = verify_paper_approval(manifest_path, settings)
    except ApprovalRejectedError as exc:
        raise ApprovedModelError(str(exc)) from exc
    manifest = verified.manifest

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
        content_hash=verified.content_hash,
        approval_status=str(manifest.get("approval_status") or "paper"),
    )
