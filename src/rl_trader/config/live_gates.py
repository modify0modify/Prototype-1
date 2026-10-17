"""Fail-fast live-trading activation gates (before any live broker connect).

All of the following must pass:

1. ``APP_MODE=live``
2. ``LIVE_TRADING_ENABLED=true``
3. ``BROKER_ENVIRONMENT=live``
4. ``LIVE_TRADING_ACKNOWLEDGEMENT`` equals
   :data:`~rl_trader.config.constants.LIVE_TRADING_CONFIRMATION_PHRASE`
5. Approved model manifest exists and marks the model approved
6. Risk-engine startup checks pass
7. Non-empty live account identifier
8. Explicit maximum live capital (> 0)
9. Explicit maximum order value (> 0)
10. Explicit daily loss limit in risk settings
11. Point-in-time universe **or** explicit survivorship-bias acknowledgement
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rl_trader.config.constants import LIVE_TRADING_CONFIRMATION_PHRASE
from rl_trader.config.masking import mask_account_id

if TYPE_CHECKING:
    from rl_trader.config.settings import AppSettings


@dataclass(frozen=True, slots=True)
class GuardCheck:
    """Single activation check result (safe for logs — no secrets)."""

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


def _live_manifest_approved(settings: AppSettings) -> tuple[bool, str]:
    """Require sealed live-tier formal approval (stricter than paper)."""
    path = settings.live.approved_model_manifest
    if not path.is_file():
        return False, f"manifest missing: {path}"
    try:
        from rl_trader.approval.service import ApprovalRejectedError, verify_live_approval

        verified = verify_live_approval(path, settings.approval)
    except ApprovalRejectedError as exc:
        return False, f"live approval rejected: {exc}"
    except Exception as exc:  # noqa: BLE001 — surface any verify failure as blocked
        return False, f"live approval error: {exc}"
    return (
        True,
        f"live-approved model_id={verified.manifest.get('model_id')} "
        f"hash={verified.content_hash[:12]}…",
    )


def evaluate_live_activation(settings: AppSettings) -> ActivationReport:
    """Evaluate every live gate without connecting to a broker."""
    # Lazy import avoids config ↔ risk circular imports at package import time.
    from rl_trader.config.settings import RuntimeMode
    from rl_trader.risk.startup import run_risk_startup_checks

    ack = settings.live.acknowledgement
    ack_ok = ack == LIVE_TRADING_CONFIRMATION_PHRASE
    account_ok = bool(settings.live.account_id.strip())
    capital_ok = settings.live.max_live_capital is not None and settings.live.max_live_capital > 0
    order_ok = settings.live.max_order_value is not None and settings.live.max_order_value > 0
    daily_ok = settings.risk.daily_loss_limit is not None
    manifest_ok, manifest_detail = _live_manifest_approved(settings)
    risk_report = run_risk_startup_checks(settings.risk)

    checks = (
        GuardCheck(
            name="app_mode_live",
            passed=settings.app_mode is RuntimeMode.LIVE,
            detail=f"APP_MODE={settings.app_mode.value}",
        ),
        GuardCheck(
            name="live_trading_enabled",
            passed=settings.live_trading_enabled is True,
            detail=f"LIVE_TRADING_ENABLED={settings.live_trading_enabled}",
        ),
        GuardCheck(
            name="broker_environment_live",
            passed=settings.broker.environment == "live",
            detail=f"BROKER_ENVIRONMENT={settings.broker.environment}",
        ),
        GuardCheck(
            name="live_trading_acknowledgement",
            passed=ack_ok,
            detail=(
                "acknowledgement phrase matches"
                if ack_ok
                else "acknowledgement phrase missing or incorrect"
            ),
        ),
        GuardCheck(
            name="approved_model_manifest",
            passed=manifest_ok,
            detail=manifest_detail,
        ),
        GuardCheck(
            name="live_tier_formal_approval",
            passed=manifest_ok,
            detail=manifest_detail,
        ),
        GuardCheck(
            name="risk_engine_startup",
            passed=risk_report.ok,
            detail=(
                "risk startup checks passed"
                if risk_report.ok
                else "failed: "
                + ", ".join(f"{c.name}" for c in risk_report.failed)
            ),
        ),
        GuardCheck(
            name="live_account_id",
            passed=account_ok,
            detail=(
                f"account_id={mask_account_id(settings.live.account_id)}"
                if account_ok
                else "live account identifier empty"
            ),
        ),
        GuardCheck(
            name="max_live_capital",
            passed=capital_ok,
            detail=(
                f"max_live_capital={settings.live.max_live_capital}"
                if capital_ok
                else "max_live_capital missing"
            ),
        ),
        GuardCheck(
            name="max_order_value",
            passed=order_ok,
            detail=(
                f"max_order_value={settings.live.max_order_value}"
                if order_ok
                else "max_order_value missing"
            ),
        ),
        GuardCheck(
            name="daily_loss_limit",
            passed=daily_ok,
            detail=(
                f"daily_loss_limit={settings.risk.daily_loss_limit}"
                if daily_ok
                else "daily_loss_limit missing"
            ),
        ),
        GuardCheck(
            name="survivorship_universe_policy",
            passed=(
                settings.data.point_in_time_universe
                or settings.data.survivorship_bias_acknowledged
            ),
            detail=(
                "PIT universe or survivorship acknowledgement present"
                if (
                    settings.data.point_in_time_universe
                    or settings.data.survivorship_bias_acknowledged
                )
                else "static universe without survivorship acknowledgement"
            ),
        ),
    )
    return ActivationReport(allowed=all(c.passed for c in checks), checks=checks)


def assert_live_trading_allowed(settings: AppSettings) -> ActivationReport:
    """Raise :class:`LiveTradingBlockedError` unless all gates pass.

    Call this **before** constructing or connecting a live broker adapter.
    """
    report = evaluate_live_activation(settings)
    if not report.allowed:
        failed = ", ".join(f"{c.name}: {c.detail}" for c in report.failed)
        raise LiveTradingBlockedError(
            f"live trading blocked — fail-fast before broker connect: {failed}"
        )
    return report


def is_live_trading_blocked(settings: AppSettings) -> bool:
    """Return True when live routing must not proceed (safe default)."""
    return not evaluate_live_activation(settings).allowed
