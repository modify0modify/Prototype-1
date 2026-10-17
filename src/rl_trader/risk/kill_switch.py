"""Kill-switch state — trips on loss-limit breaches; requires manual reset."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class KillSwitchEvent:
    """Immutable kill-switch audit event."""

    event_type: str
    reason: str
    timestamp: datetime
    details: dict[str, str] = field(default_factory=dict)


class KillSwitch:
    """Fail-closed trading halt requiring an explicit manual reset phrase."""

    def __init__(self, *, reset_phrase: str = "MANUAL_KILL_SWITCH_RESET") -> None:
        self._reset_phrase = reset_phrase
        self._active = False
        self._reason = ""
        self._tripped_at: datetime | None = None
        self._events: list[KillSwitchEvent] = []

    @property
    def is_active(self) -> bool:
        """Whether new orders must be rejected."""
        return self._active

    @property
    def reason(self) -> str:
        """Last trip reason (empty when inactive)."""
        return self._reason

    @property
    def events(self) -> tuple[KillSwitchEvent, ...]:
        """Audit trail of trip / reset events."""
        return tuple(self._events)

    def trip(self, reason: str, *, details: dict[str, str] | None = None) -> None:
        """Activate the kill switch (idempotent while already active)."""
        now = datetime.now(UTC)
        if not self._active:
            self._active = True
            self._reason = reason
            self._tripped_at = now
        self._events.append(
            KillSwitchEvent(
                event_type="trip",
                reason=reason,
                timestamp=now,
                details=dict(details or {}),
            )
        )

    def manual_reset(self, acknowledgement: str) -> None:
        """Clear the kill switch only when ``acknowledgement`` matches the phrase.

        Automatic / empty resets are refused — operator intent is required.
        """
        if acknowledgement != self._reset_phrase:
            raise PermissionError(
                "kill-switch reset refused: acknowledgement phrase mismatch "
                "(manual reset required)"
            )
        if not self._active:
            return
        now = datetime.now(UTC)
        self._events.append(
            KillSwitchEvent(
                event_type="manual_reset",
                reason=self._reason,
                timestamp=now,
                details={"acknowledgement": "accepted"},
            )
        )
        self._active = False
        self._reason = ""
        self._tripped_at = None
