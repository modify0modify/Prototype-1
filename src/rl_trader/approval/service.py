"""High-level model approval verification for paper and live paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_trader.approval.evidence import ApprovalEvidence
from rl_trader.approval.gates import ApprovalGateReport, evaluate_live_gates, evaluate_paper_gates
from rl_trader.approval.manifest import (
    ManifestError,
    load_manifest_file,
    verify_manifest,
)
from rl_trader.config.models import ApprovalSettings


class ApprovalRejectedError(RuntimeError):
    """Raised when a model fails formal approval verification."""


@dataclass(frozen=True, slots=True)
class VerifiedApproval:
    """Result of loading + verifying an approval manifest."""

    manifest: dict[str, Any]
    content_hash: str
    evidence: ApprovalEvidence
    gate_report: ApprovalGateReport
    tier: str


def _evidence_from_manifest(manifest: dict[str, Any]) -> ApprovalEvidence:
    raw = manifest.get("evaluation_evidence") or {}
    if not isinstance(raw, dict):
        raise ApprovalRejectedError("evaluation_evidence must be an object")
    return ApprovalEvidence.from_dict(raw)


def verify_paper_approval(
    manifest_path: Path,
    settings: ApprovalSettings,
) -> VerifiedApproval:
    """Require a sealed manifest with paper (or live) status that passes paper gates."""
    try:
        manifest = load_manifest_file(manifest_path)
        content_hash = verify_manifest(
            manifest,
            require_signature=settings.require_manifest_signature,
        )
    except ManifestError as exc:
        raise ApprovalRejectedError(str(exc)) from exc

    status = str(manifest.get("approval_status"))
    if status not in {"paper", "live"}:
        raise ApprovalRejectedError(
            f"manifest approval_status={status!r} is not paper/live — cannot enter paper trading"
        )

    evidence = _evidence_from_manifest(manifest)
    report = evaluate_paper_gates(evidence, settings)
    if not report.allowed:
        failed = "; ".join(f"{r.name}: {r.detail}" for r in report.failed)
        raise ApprovalRejectedError(f"paper approval gates failed: {failed}")

    return VerifiedApproval(
        manifest=manifest,
        content_hash=content_hash,
        evidence=evidence,
        gate_report=report,
        tier="paper",
    )


def verify_live_approval(
    manifest_path: Path,
    settings: ApprovalSettings,
) -> VerifiedApproval:
    """Require sealed live-tier approval (stricter than paper)."""
    try:
        manifest = load_manifest_file(manifest_path)
        content_hash = verify_manifest(
            manifest,
            require_signature=settings.require_manifest_signature,
        )
    except ManifestError as exc:
        raise ApprovalRejectedError(str(exc)) from exc

    status = str(manifest.get("approval_status"))
    if status != "live":
        raise ApprovalRejectedError(
            f"live activation requires approval_status='live' (got {status!r})"
        )

    evidence = _evidence_from_manifest(manifest)
    report = evaluate_live_gates(evidence, settings)
    if not report.allowed:
        failed = "; ".join(f"{r.name}: {r.detail}" for r in report.failed)
        raise ApprovalRejectedError(f"live approval gates failed: {failed}")

    return VerifiedApproval(
        manifest=manifest,
        content_hash=content_hash,
        evidence=evidence,
        gate_report=report,
        tier="live",
    )


def maybe_persist_verified(
    verified: VerifiedApproval,
    *,
    database_url: str | None,
    persist: bool,
) -> None:
    """Optionally dual-write the approval into SQLite."""
    if not persist or not database_url:
        return
    try:
        from rl_trader.approval.store import persist_approval
        from rl_trader.storage.db import create_db_engine
        from rl_trader.storage.migrate import upgrade_head

        upgrade_head(database_url)
        engine = create_db_engine(database_url, create_schema=False)
        persist_approval(
            engine,
            model_id=str(verified.manifest["model_id"]),
            approval_tier=verified.tier,
            approved=True,
            approved_by=str(verified.manifest.get("approver") or ""),
            content_hash=verified.content_hash,
            code_commit_id=str(verified.manifest.get("code_commit_id") or ""),
            manifest_path=str(verified.manifest.get("_path") or ""),
            evidence=verified.evidence.to_dict(),
            notes=str(verified.manifest.get("notes") or ""),
        )
    except Exception:
        # Persistence must not unblock trading if DB is unavailable — caller
        # already has a verified in-memory approval. Always log the failure.
        logging.getLogger("rl_trader.approval").exception(
            "failed to persist approval for model_id=%s (in-memory verification intact)",
            verified.manifest.get("model_id"),
        )
        return
