"""Risk-engine startup checks (fail-fast before live broker connect)."""

from __future__ import annotations

from dataclasses import dataclass

from rl_trader.config.models import RiskSettings


@dataclass(frozen=True, slots=True)
class RiskStartupCheck:
    """Single risk startup check result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class RiskStartupReport:
    """Aggregate risk startup validation."""

    ok: bool
    checks: tuple[RiskStartupCheck, ...]

    @property
    def failed(self) -> tuple[RiskStartupCheck, ...]:
        """Checks that did not pass."""
        return tuple(c for c in self.checks if not c.passed)


def run_risk_startup_checks(risk: RiskSettings) -> RiskStartupReport:
    """Validate risk settings are safe to use before any broker connection."""
    daily_ok = risk.daily_loss_limit is not None
    position_ok = risk.max_position_size <= risk.max_portfolio_exposure
    exposure_ok = 0.0 < risk.max_portfolio_exposure <= 1.0
    position_range_ok = 0.0 < risk.max_position_size <= 1.0
    drawdown_ok = 0.0 < risk.max_drawdown_limit <= 1.0
    daily_range_ok = (
        risk.daily_loss_limit is not None and 0.0 < risk.daily_loss_limit <= 1.0
    )
    weekly_ok = 0.0 < risk.weekly_loss_limit <= 1.0
    cash_ok = 0.0 <= risk.min_cash_reserve < 1.0
    sector_ok = 0.0 < risk.max_sector_exposure <= 1.0
    phrase_ok = bool(risk.kill_switch_reset_phrase.strip())

    checks = (
        RiskStartupCheck(
            name="daily_loss_limit_present",
            passed=daily_ok,
            detail=(
                f"daily_loss_limit={risk.daily_loss_limit}"
                if daily_ok
                else "daily_loss_limit missing"
            ),
        ),
        RiskStartupCheck(
            name="daily_loss_limit_range",
            passed=daily_range_ok,
            detail=f"daily_loss_limit={risk.daily_loss_limit}",
        ),
        RiskStartupCheck(
            name="weekly_loss_limit_range",
            passed=weekly_ok,
            detail=f"weekly_loss_limit={risk.weekly_loss_limit}",
        ),
        RiskStartupCheck(
            name="position_within_exposure",
            passed=position_ok,
            detail=(
                f"max_position_size={risk.max_position_size} "
                f"max_portfolio_exposure={risk.max_portfolio_exposure}"
            ),
        ),
        RiskStartupCheck(
            name="exposure_range",
            passed=exposure_ok,
            detail=f"max_portfolio_exposure={risk.max_portfolio_exposure}",
        ),
        RiskStartupCheck(
            name="position_size_range",
            passed=position_range_ok,
            detail=f"max_position_size={risk.max_position_size}",
        ),
        RiskStartupCheck(
            name="max_drawdown_range",
            passed=drawdown_ok,
            detail=f"max_drawdown_limit={risk.max_drawdown_limit}",
        ),
        RiskStartupCheck(
            name="min_cash_reserve_range",
            passed=cash_ok,
            detail=f"min_cash_reserve={risk.min_cash_reserve}",
        ),
        RiskStartupCheck(
            name="max_sector_exposure_range",
            passed=sector_ok,
            detail=f"max_sector_exposure={risk.max_sector_exposure}",
        ),
        RiskStartupCheck(
            name="kill_switch_reset_phrase",
            passed=phrase_ok,
            detail="kill_switch_reset_phrase configured",
        ),
    )
    return RiskStartupReport(ok=all(c.passed for c in checks), checks=checks)
