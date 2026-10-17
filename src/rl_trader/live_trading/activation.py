"""Manual live activation / deactivation (human-operated)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rl_trader.config.constants import LIVE_TRADING_CONFIRMATION_PHRASE
from rl_trader.config.live_gates import (
    LiveTradingBlockedError,
    assert_live_trading_allowed,
)
from rl_trader.config.models import LiveTradingSettings
from rl_trader.config.settings import AppSettings


@dataclass(slots=True)
class ActivationState:
    """Runtime arming state (not persisted secrets)."""

    activated: bool = False
    activated_at: datetime | None = None
    deactivated_at: datetime | None = None
    last_actor: str = ""


class ManualLiveController:
    """Explicit manual arming on top of complete safety configuration.

    ``activate`` requires the confirmation phrase **and** passing
    :func:`assert_live_trading_allowed`. ``deactivate`` is always allowed and
    immediately blocks live submits.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._state = ActivationState(activated=bool(settings.live.manual_activated))

    @property
    def is_activated(self) -> bool:
        """True when manually armed for live path (still subject to stage)."""
        return self._state.activated

    @property
    def state(self) -> ActivationState:
        """Copy-ish view of arming state."""
        return ActivationState(
            activated=self._state.activated,
            activated_at=self._state.activated_at,
            deactivated_at=self._state.deactivated_at,
            last_actor=self._state.last_actor,
        )

    def activate(self, acknowledgement: str, *, actor: str = "operator") -> AppSettings:
        """Arm live path after gates pass; returns settings with manual_activated=True."""
        if acknowledgement != LIVE_TRADING_CONFIRMATION_PHRASE:
            raise LiveTradingBlockedError(
                "manual activation refused — acknowledgement phrase incorrect"
            )
        assert_live_trading_allowed(self._settings)
        self._state.activated = True
        self._state.activated_at = datetime.now(UTC)
        self._state.deactivated_at = None
        self._state.last_actor = actor
        live = self._settings.live.model_copy(update={"manual_activated": True})
        self._settings = self._settings.model_copy(update={"live": live})
        return self._settings

    def deactivate(self, *, actor: str = "operator") -> AppSettings:
        """Disarm immediately — live submits must stop."""
        self._state.activated = False
        self._state.deactivated_at = datetime.now(UTC)
        self._state.last_actor = actor
        live = self._settings.live.model_copy(update={"manual_activated": False})
        self._settings = self._settings.model_copy(update={"live": live})
        return self._settings

    def require_activated(self) -> None:
        """Raise if not manually armed."""
        if not self._state.activated:
            raise LiveTradingBlockedError(
                "live path disarmed — call ManualLiveController.activate() first"
            )


def with_manual_flag(settings: AppSettings, activated: bool) -> AppSettings:
    """Return a copy of settings with ``live.manual_activated`` set."""
    live: LiveTradingSettings = settings.live.model_copy(
        update={"manual_activated": activated}
    )
    return settings.model_copy(update={"live": live})
