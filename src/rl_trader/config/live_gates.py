"""Fail-fast live-trading activation gates (before any live broker connect).

All of the following must pass:

1. ``APP_MODE=live``
2. ``LIVE_TRADING_ENABLED=true``
3. ``BROKER_ENVIRONMENT=live`` (separate from paper)
4. ``LIVE_TRADING_ACKNOWLEDGEMENT`` equals confirmation phrase
5. Approved model manifest (live tier)
6. Risk-engine startup checks pass
7. Non-empty live account identifier on approved allowlist
8. Explicit maximum live capital (> 0)
9. Explicit maximum order value (> 0)
10. Explicit daily loss limit + drawdown limit
11. Point-in-time universe **or** survivorship-bias acknowledgement
12. Strict position caps configured
13. Rollout stage in 0..3 (Stage 0 = shadow only; still valid for gated sandbox)
14. Live credential slots distinct from paper (keys may be unset in stub mode)

Passing these gates does **not** place live orders. Manual activation and
stage policy still gate submits; Stage 0 never submits live orders.
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
    """Evaluate every live config gate without connecting to a broker."""
    from rl_trader.config.settings import RuntimeMode
    from rl_trader.risk.startup import run_risk_startup_checks

    ack = settings.live.acknowledgement
    ack_ok = ack == LIVE_TRADING_CONFIRMATION_PHRASE
    account_ok = bool(settings.live.account_id.strip())
    allowlist = {a.strip() for a in settings.live.approved_account_allowlist if a.strip()}
    allowlist_ok = account_ok and settings.live.account_id.strip() in allowlist
    capital_ok = settings.live.max_live_capital is not None and settings.live.max_live_capital > 0
    order_ok = settings.live.max_order_value is not None and settings.live.max_order_value > 0
    daily_ok = settings.risk.daily_loss_limit is not None
    drawdown_ok = 0.0 < settings.risk.max_drawdown_limit <= 1.0
    position_cap_ok = 0.0 < settings.live.max_position_fraction <= 1.0
    stage_ok = 0 <= int(settings.live.rollout_stage) <= 3
    # Separate live credentials: paper keys must not be required/used for live env.
    # Presence of live key slots is optional for the sandbox stub, but paper env is forbidden.
    creds_separated = settings.broker.environment == "live"
    # Stage 2/3 review flags: required only when stage demands them for *orders*;
    # activation config may still load at stage 0.
    stage = int(settings.live.rollout_stage)
    stage_review_ok = True
    stage_review_detail = "n/a for stage"
    if stage >= 2 and not settings.live.stage2_review_acknowledged:
        stage_review_ok = False
        stage_review_detail = "stage2_review_acknowledged required for stage>=2"
    elif stage >= 3 and not settings.live.stage3_review_acknowledged:
        stage_review_ok = False
        stage_review_detail = "stage3_review_acknowledged required for stage>=3"
    elif stage >= 3 and not settings.live.stage3_expanded_universe:
        stage_review_ok = False
        stage_review_detail = "stage3_expanded_universe required for stage 3"
    else:
        stage_review_detail = f"stage={stage} reviews ok"

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
            name="separate_live_credentials",
            passed=creds_separated,
            detail=(
                "live env selected — paper credential fields ignored by LiveBroker"
                if creds_separated
                else "broker.environment must be live (paper credentials must not route here)"
            ),
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
            name="approved_account_allowlist",
            passed=allowlist_ok,
            detail=(
                "account on allowlist"
                if allowlist_ok
                else "account_id missing from approved_account_allowlist"
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
            name="strict_position_caps",
            passed=position_cap_ok,
            detail=f"max_position_fraction={settings.live.max_position_fraction}",
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
            name="drawdown_limit",
            passed=drawdown_ok,
            detail=f"max_drawdown_limit={settings.risk.max_drawdown_limit}",
        ),
        GuardCheck(
            name="rollout_stage",
            passed=stage_ok,
            detail=f"rollout_stage={settings.live.rollout_stage}",
        ),
        GuardCheck(
            name="stage_review_acknowledgements",
            passed=stage_review_ok,
            detail=stage_review_detail,
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
        GuardCheck(
            name="stale_data_threshold_configured",
            passed=settings.live.max_data_age_seconds > 0,
            detail=f"max_data_age_seconds={settings.live.max_data_age_seconds}",
        ),
        GuardCheck(
            name="market_hours_check_configured",
            passed=settings.live.require_market_hours is True
            or settings.live.rollout_stage == 0,
            detail=f"require_market_hours={settings.live.require_market_hours}",
        ),
        GuardCheck(
            name="account_reconciliation_configured",
            passed=settings.live.require_account_reconciliation is True
            or settings.live.rollout_stage == 0,
            detail=(
                f"require_account_reconciliation="
                f"{settings.live.require_account_reconciliation}"
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
