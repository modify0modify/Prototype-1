"""Scheduled retrain hook — separate from the paper decision loop.

The prospective paper service keeps the loaded model frozen for the
evaluation period. Callers may invoke :func:`maybe_schedule_retrain` from an
external cron / orchestrator; this module never mutates model weights during
``run_cycle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrainHookResult:
    """Outcome of the retrain scheduling stub."""

    scheduled: bool
    reason: str
    schedule: str | None
    details: dict[str, Any]


def maybe_schedule_retrain(
    *,
    enabled: bool,
    schedule: str | None,
    evaluation_period_id: str,
    model_id: str,
    now: datetime,
) -> RetrainHookResult:
    """Stub for an external retrain scheduler.

    When ``enabled`` is False (default), returns a no-op. When enabled, records
    intent only — does **not** train or swap the in-process model.
    """
    if not enabled:
        return RetrainHookResult(
            scheduled=False,
            reason="scheduled_retrain_disabled",
            schedule=schedule,
            details={
                "evaluation_period_id": evaluation_period_id,
                "model_id": model_id,
                "now": now.isoformat(),
            },
        )
    return RetrainHookResult(
        scheduled=True,
        reason="retrain_hook_recorded_for_external_scheduler",
        schedule=schedule,
        details={
            "evaluation_period_id": evaluation_period_id,
            "model_id": model_id,
            "now": now.isoformat(),
            "note": "model remains frozen inside the paper loop",
        },
    )
