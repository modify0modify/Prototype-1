"""Hashed / optionally signed model approval manifests.

Hash scheme
-----------
1. Build a canonical JSON object of all payload fields **excluding**
   ``content_hash`` and ``signature`` (sorted keys, UTF-8, no whitespace gaps
   beyond ``separators=(",", ":")``, default=str).
2. ``content_hash = sha256(canonical_bytes).hexdigest()``.
3. Optional HMAC-SHA256 signature over ``content_hash`` using env
   ``RL_TRADER_MANIFEST_SIGNING_KEY`` (hex digest). Required only when
   ``approval.require_manifest_signature`` is True.

Tampering any hashed field invalidates ``content_hash``. Incomplete manifests
(missing required fields) are rejected before hashing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ApprovalStatus = Literal["none", "paper", "live", "rejected"]

REQUIRED_MANIFEST_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_version",
    "algorithm",
    "hyperparameters",
    "train_period",
    "val_period",
    "test_period",
    "feature_manifest",
    "benchmark_results",
    "transaction_cost_assumptions",
    "approval_status",
    "approval_timestamp",
    "approver",
    "code_commit_id",
    "checkpoint_dir",
    "evaluation_evidence",
)

SIGNING_KEY_ENV = "RL_TRADER_MANIFEST_SIGNING_KEY"


class ManifestError(RuntimeError):
    """Raised when a manifest is incomplete, tampered, or improperly signed."""


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    body = {k: payload[k] for k in sorted(payload) if k not in {"content_hash", "signature"}}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def compute_content_hash(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest over the canonical manifest payload."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def compute_signature(content_hash: str, *, key: bytes | None = None) -> str:
    """HMAC-SHA256 hex digest of ``content_hash``."""
    secret = key
    if secret is None:
        raw = os.environ.get(SIGNING_KEY_ENV, "")
        if not raw:
            raise ManifestError(
                f"signing key missing — set {SIGNING_KEY_ENV} to sign manifests"
            )
        secret = raw.encode("utf-8")
    return hmac.new(secret, content_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def assert_required_fields(payload: dict[str, Any]) -> None:
    """Reject incomplete manifests."""
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in payload or payload[f] in (None, "")]
    if missing:
        raise ManifestError(f"incomplete manifest — missing fields: {', '.join(missing)}")
    for period_key in ("train_period", "val_period", "test_period"):
        period = payload[period_key]
        if not isinstance(period, dict) or "start" not in period or "end" not in period:
            raise ManifestError(f"{period_key} must be an object with start/end")
    if not isinstance(payload.get("feature_manifest"), dict):
        raise ManifestError("feature_manifest must be an object")
    if "feature_specs" not in payload["feature_manifest"]:
        raise ManifestError("feature_manifest.feature_specs required")
    if not isinstance(payload.get("evaluation_evidence"), dict):
        raise ManifestError("evaluation_evidence must be an object")
    status = payload.get("approval_status")
    if status not in {"none", "paper", "live", "rejected"}:
        raise ManifestError(f"invalid approval_status={status!r}")


def seal_manifest(
    payload: dict[str, Any],
    *,
    sign: bool = False,
    signing_key: bytes | None = None,
) -> dict[str, Any]:
    """Validate required fields, attach ``content_hash`` (and optional signature)."""
    assert_required_fields(payload)
    sealed = deepcopy(payload)
    sealed.pop("content_hash", None)
    sealed.pop("signature", None)
    digest = compute_content_hash(sealed)
    sealed["content_hash"] = digest
    if sign:
        sealed["signature"] = compute_signature(digest, key=signing_key)
    return sealed


def verify_manifest(
    payload: dict[str, Any],
    *,
    require_signature: bool = False,
    signing_key: bytes | None = None,
) -> str:
    """Verify completeness + hash (+ optional signature). Return content_hash."""
    assert_required_fields(payload)
    expected = payload.get("content_hash")
    if not expected or not isinstance(expected, str):
        raise ManifestError("manifest missing content_hash")
    actual = compute_content_hash(payload)
    if not hmac.compare_digest(actual, expected):
        raise ManifestError(
            "manifest content_hash mismatch — modified or incomplete seal rejected"
        )
    signature = payload.get("signature")
    if require_signature:
        if not signature or not isinstance(signature, str):
            raise ManifestError("manifest signature required but missing")
        good = compute_signature(actual, key=signing_key)
        if not hmac.compare_digest(signature, good):
            raise ManifestError("manifest signature invalid — rejected")
    elif signature:
        # If a signature is present, verify it whenever a key is available.
        raw = os.environ.get(SIGNING_KEY_ENV, "")
        if signing_key is not None or raw:
            good = compute_signature(actual, key=signing_key)
            if not hmac.compare_digest(str(signature), good):
                raise ManifestError("manifest signature invalid — rejected")
    return actual


def load_manifest_file(path: Path) -> dict[str, Any]:
    """Load a JSON manifest document."""
    if not path.is_file():
        raise ManifestError(f"manifest missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"manifest unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("manifest root must be an object")
    return data


def write_manifest_file(path: Path, payload: dict[str, Any]) -> Path:
    """Write a sealed manifest JSON document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class Period:
    """Inclusive evaluation window."""

    start: str
    end: str

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end}


def build_manifest_payload(
    *,
    model_id: str,
    model_version: str,
    algorithm: str,
    hyperparameters: dict[str, Any],
    train_period: Period,
    val_period: Period,
    test_period: Period,
    feature_manifest: dict[str, Any],
    benchmark_results: dict[str, Any] | list[Any],
    transaction_cost_assumptions: dict[str, Any],
    approval_status: ApprovalStatus,
    approver: str,
    code_commit_id: str,
    checkpoint_dir: str | Path,
    evaluation_evidence: dict[str, Any],
    approval_timestamp: str | None = None,
    n_assets: int | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble an unsealed manifest payload (call :func:`seal_manifest` next)."""
    payload: dict[str, Any] = {
        "model_id": model_id,
        "model_version": model_version,
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "train_period": train_period.to_dict(),
        "val_period": val_period.to_dict(),
        "test_period": test_period.to_dict(),
        "feature_manifest": feature_manifest,
        "benchmark_results": benchmark_results,
        "transaction_cost_assumptions": transaction_cost_assumptions,
        "approval_status": approval_status,
        "approval_timestamp": approval_timestamp
        or datetime.now(UTC).isoformat(),
        "approver": approver,
        "code_commit_id": code_commit_id,
        "checkpoint_dir": str(checkpoint_dir),
        "evaluation_evidence": evaluation_evidence,
        "notes": notes,
    }
    if n_assets is not None:
        payload["n_assets"] = n_assets
    if extra:
        payload.update(extra)
    return payload
