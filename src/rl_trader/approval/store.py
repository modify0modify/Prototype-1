"""Persist formal approvals into ``model_approvals``."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from rl_trader.storage.db import session_scope
from rl_trader.storage.models import ModelApprovalRow, new_uuid
from rl_trader.storage.secrets import sanitize_config_for_storage


def persist_approval(
    engine: Engine,
    *,
    model_id: str,
    approval_tier: str,
    approved: bool,
    approved_by: str,
    content_hash: str,
    code_commit_id: str,
    manifest_path: str | Path,
    evidence: dict[str, Any],
    notes: str = "",
    checkpoint_id: uuid.UUID | None = None,
    approved_at: datetime | None = None,
) -> uuid.UUID:
    """Insert one approval row; returns its UUID."""
    with session_scope(engine) as session:
        row = ModelApprovalRow(
            id=new_uuid(),
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            approved=approved,
            approved_at=approved_at or datetime.now(UTC),
            approved_by=approved_by,
            manifest_path=str(manifest_path),
            notes=notes,
            approval_tier=approval_tier,
            content_hash=content_hash,
            code_commit_id=code_commit_id,
            evidence_json=json.dumps(sanitize_config_for_storage(evidence), default=str),
        )
        session.add(row)
        session.flush()
        return row.id
