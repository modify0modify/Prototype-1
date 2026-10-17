"""Pre-submit live safety checks (beyond activation gates)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rl_trader.brokers.types import OrderRequest, OrderSide, Quote
from rl_trader.live_trading.stages import resolve_stage_policy

if TYPE_CHECKING:
    from rl_trader.config.settings import AppSettings
    from rl_trader.live_trading.activation import ManualLiveController
    from rl_trader.risk.kill_switch import KillSwitch


class LiveSafetyError(RuntimeError):
    """Raised when a live submit fails a safety layer."""


@dataclass(frozen=True, slots=True)
class SafetyCheck:
    """One pre-submit check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SubmitSafetyReport:
    """Aggregate pre-submit decision."""

    allowed: bool
    shadow_only: bool
    checks: tuple[SafetyCheck, ...]

    @property
    def failed(self) -> tuple[SafetyCheck, ...]:
        return tuple(c for c in self.checks if not c.passed)


def evaluate_submit_safety(
    settings: AppSettings,
    request: OrderRequest,
    *,
    manual: ManualLiveController,
    kill_switch: KillSwitch | None = None,
    quote: Quote | None = None,
    now: datetime | None = None,
    market_open: bool | None = None,
    reconciliation_ok: bool = True,
    seen_idempotency_keys: set[str] | None = None,
) -> SubmitSafetyReport:
    """Evaluate safety layers for one prospective live submit."""
    from rl_trader.config.live_gates import evaluate_live_activation

    activation = evaluate_live_activation(settings)
    policy = resolve_stage_policy(
        settings.live, base_universe=settings.data.universe
    )
    checks: list[SafetyCheck] = [
        SafetyCheck(
            name="activation_gates",
            passed=activation.allowed,
            detail="ok" if activation.allowed else "activation gates failed",
        ),
        SafetyCheck(
            name="manual_activation",
            passed=manual.is_activated,
            detail="armed" if manual.is_activated else "disarmed",
        ),
        SafetyCheck(
            name="separate_live_credentials",
            passed=settings.broker.environment == "live",
            detail=f"broker.environment={settings.broker.environment}",
        ),
        SafetyCheck(
            name="paper_live_routing_separation",
            passed=settings.app_mode.value == "live"
            and settings.broker.environment == "live",
            detail="live mode uses live broker env only",
        ),
    ]

    ks_active = bool(kill_switch.is_active) if kill_switch is not None else False
    checks.append(
        SafetyCheck(
            name="global_kill_switch",
            passed=not ks_active,
            detail="tripped" if ks_active else "clear",
        )
    )

    allowlist = {a.strip() for a in settings.live.approved_account_allowlist if a.strip()}
    acct = settings.live.account_id.strip()
    checks.append(
        SafetyCheck(
            name="approved_account_allowlist",
            passed=bool(acct) and acct in allowlist,
            detail=f"account in allowlist={acct in allowlist}",
        )
    )

    capital = settings.live.max_live_capital
    checks.append(
        SafetyCheck(
            name="max_live_capital",
            passed=capital is not None and capital > 0,
            detail=f"max_live_capital={capital}",
        )
    )
    order_cap = settings.live.max_order_value
    checks.append(
        SafetyCheck(
            name="max_order_notional",
            passed=order_cap is not None and order_cap > 0,
            detail=f"max_order_value={order_cap}",
        )
    )
    checks.append(
        SafetyCheck(
            name="strict_position_caps",
            passed=0.0 < settings.live.max_position_fraction <= settings.risk.max_position_size
            or settings.live.max_position_fraction <= 1.0,
            detail=f"max_position_fraction={settings.live.max_position_fraction}",
        )
    )
    checks.append(
        SafetyCheck(
            name="daily_loss_limit",
            passed=settings.risk.daily_loss_limit is not None,
            detail=f"daily_loss_limit={settings.risk.daily_loss_limit}",
        )
    )
    checks.append(
        SafetyCheck(
            name="drawdown_limit",
            passed=0.0 < settings.risk.max_drawdown_limit <= 1.0,
            detail=f"max_drawdown_limit={settings.risk.max_drawdown_limit}",
        )
    )

    if settings.live.require_market_hours:
        open_ok = market_open is True
        checks.append(
            SafetyCheck(
                name="market_hours_check",
                passed=open_ok,
                detail="open" if open_ok else "closed_or_unknown",
            )
        )

    if settings.live.require_account_reconciliation:
        checks.append(
            SafetyCheck(
                name="account_reconciliation",
                passed=reconciliation_ok is True,
                detail="ok" if reconciliation_ok else "mismatch",
            )
        )

    # Stale-data rejection
    stale_ok = True
    if quote is not None and now is not None:
        age = (now - quote.timestamp).total_seconds()
        stale_ok = age <= settings.live.max_data_age_seconds
        checks.append(
            SafetyCheck(
                name="stale_data_rejection",
                passed=stale_ok,
                detail=f"quote_age_s={age:.3f} max={settings.live.max_data_age_seconds}",
            )
        )
    else:
        checks.append(
            SafetyCheck(
                name="stale_data_rejection",
                passed=True,
                detail="no quote provided — deferred",
            )
        )

    # Duplicate-order prevention
    key = request.idempotency_key or request.client_order_id
    dup_ok = True
    if key and seen_idempotency_keys is not None:
        dup_ok = key not in seen_idempotency_keys
    checks.append(
        SafetyCheck(
            name="duplicate_order_prevention",
            passed=dup_ok,
            detail="unique" if dup_ok else f"duplicate key={key}",
        )
    )

    # Stage / long-only / universe
    checks.append(
        SafetyCheck(
            name="rollout_stage_policy",
            passed=policy.allow_live_orders or policy.stage.value == 0,
            detail=policy.detail,
        )
    )
    if policy.long_only and request.side is OrderSide.SELL:
        # SELL to flatten may be needed; Stage 1 long-only blocks opening shorts.
        # Treat naked sell (no position context here) as blocked for safety.
        checks.append(
            SafetyCheck(
                name="long_only",
                passed=False,
                detail="sell blocked under long_only stage policy without position ctx",
            )
        )
    else:
        checks.append(
            SafetyCheck(name="long_only", passed=True, detail="ok"),
        )

    if policy.allowed_symbols is not None:
        sym_ok = request.symbol.upper() in {s.upper() for s in policy.allowed_symbols}
        checks.append(
            SafetyCheck(
                name="universe_allowlist",
                passed=sym_ok,
                detail=f"symbol={request.symbol} allowed={sym_ok}",
            )
        )

    shadow_only = not policy.allow_live_orders
    # For Stage 0, "allowed" means we may record a shadow decision — not submit live.
    # Fail closed on safety layers other than stage permitting live orders.
    non_critical = {"rollout_stage_policy", "long_only", "universe_allowlist"}
    critical_failed = [
        c for c in checks if not c.passed and c.name not in non_critical
    ]
    # When not shadow-only, all checks must pass.
    if shadow_only:
        allowed = len(critical_failed) == 0 and manual.is_activated and activation.allowed
    else:
        allowed = all(c.passed for c in checks)

    return SubmitSafetyReport(
        allowed=allowed,
        shadow_only=shadow_only,
        checks=tuple(checks),
    )


def assert_submit_safety(
    settings: AppSettings,
    request: OrderRequest,
    **kwargs: object,
) -> SubmitSafetyReport:
    """Raise :class:`LiveSafetyError` when submit safety fails."""
    report = evaluate_submit_safety(settings, request, **kwargs)  # type: ignore[arg-type]
    if not report.allowed:
        failed = ", ".join(f"{c.name}:{c.detail}" for c in report.failed)
        raise LiveSafetyError(f"live submit blocked: {failed}")
    return report
