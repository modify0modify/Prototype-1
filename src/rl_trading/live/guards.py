"""Live-trading activation guards.

Live order routing is blocked by default. Activation requires multiple
independent checks so a single mis-set env var cannot enable live trading.
RL agents still never send orders; only risk-approved targets may reach a
future live adapter — and only when every guard passes.
"""

from __future__ import annotations

from dataclasses import dataclass

from rl_trading.config.settings import RuntimeMode, Settings


@dataclass(frozen=True, slots=True)
class GuardCheck:
    """Single activation check result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ActivationReport:
    """Aggregate live-activation decision."""

    allowed: bool
    checks: tuple[GuardCheck, ...]

    @property
    def failed(self) -> tuple[GuardCheck, ...]:
        """Checks that did not pass."""
        return tuple(c for c in self.checks if not c.passed)


class LiveTradingBlockedError(RuntimeError):
    """Raised when live routing is attempted without full activation."""


def evaluate_live_activation(settings: Settings) -> ActivationReport:
    """Run multi-check live activation evaluation.

    All of the following must pass:
    1. ``mode == live``
    2. ``live_trading_enabled`` is True
    3. ``live_activation_token`` is non-empty
    4. Broker API key and secret are present
    """
    token = settings.live_activation_token
    token_ok = token is not None and bool(token.get_secret_value().strip())
    key_ok = (
        settings.broker_api_key is not None
        and bool(settings.broker_api_key.get_secret_value().strip())
    )
    secret_ok = (
        settings.broker_api_secret is not None
        and bool(settings.broker_api_secret.get_secret_value().strip())
    )

    checks = (
        GuardCheck(
            name="mode_is_live",
            passed=settings.mode is RuntimeMode.LIVE,
            detail=f"mode={settings.mode.value}",
        ),
        GuardCheck(
            name="live_trading_enabled",
            passed=settings.live_trading_enabled,
            detail=f"live_trading_enabled={settings.live_trading_enabled}",
        ),
        GuardCheck(
            name="activation_token_present",
            passed=token_ok,
            detail="token set" if token_ok else "token missing",
        ),
        GuardCheck(
            name="broker_credentials_present",
            passed=key_ok and secret_ok,
            detail="broker credentials present"
            if key_ok and secret_ok
            else "broker credentials incomplete",
        ),
    )
    allowed = all(c.passed for c in checks)
    return ActivationReport(allowed=allowed, checks=checks)


def assert_live_trading_allowed(settings: Settings) -> ActivationReport:
    """Raise :class:`LiveTradingBlockedError` unless all guards pass."""
    report = evaluate_live_activation(settings)
    if not report.allowed:
        failed = ", ".join(f"{c.name}: {c.detail}" for c in report.failed)
        raise LiveTradingBlockedError(
            f"live trading blocked by default / failed guards: {failed}"
        )
    return report


def is_live_trading_blocked(settings: Settings) -> bool:
    """Return True when live routing must not proceed (safe default)."""
    return not evaluate_live_activation(settings).allowed
