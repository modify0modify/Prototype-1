"""Capital ceilings — human/config only; never auto-increased by the system."""

from __future__ import annotations


class CapitalAutoIncreaseError(RuntimeError):
    """Raised when any code path attempts to raise live capital programmatically."""


class CapitalLimitGuard:
    """Immutable-at-runtime capital ceiling loaded from human config.

    Reloading a new config with a higher limit is a deliberate human action
    (new process / settings object). In-process ``propose_increase`` always fails.
    """

    def __init__(self, configured_limit: float | None) -> None:
        if configured_limit is not None and configured_limit < 0:
            raise ValueError("configured_limit must be >= 0")
        self._limit = configured_limit

    @property
    def limit(self) -> float | None:
        """Current configured ceiling (None = unset / blocked)."""
        return self._limit

    def propose_increase(self, new_limit: float) -> None:
        """Always refuse — the system must never raise its own capital allocation."""
        raise CapitalAutoIncreaseError(
            f"capital auto-increase forbidden "
            f"(configured={self._limit!r}, proposed={new_limit!r}); "
            "raise limits only via human-reviewed config reload"
        )

    def propose_decrease(self, new_limit: float) -> float:
        """Allow in-process tightening (never loosening)."""
        if new_limit < 0:
            raise ValueError("new_limit must be >= 0")
        if self._limit is None:
            self._limit = new_limit
            return self._limit
        if new_limit > self._limit:
            self.propose_increase(new_limit)
        self._limit = new_limit
        return self._limit
