"""Individual deterministic risk rules (pure functions where practical)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from rl_trader.config.models import RiskSettings
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.context import RiskReviewContext


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """Outcome of a single rule evaluation."""

    rule: str
    ok: bool
    action: str  # "pass" | "reject" | "resize" | "kill"
    message: str
    resized_weights: np.ndarray | None = None


def order_fingerprint(weights: np.ndarray) -> str:
    """Stable fingerprint for duplicate-order detection."""
    rounded = np.round(np.asarray(weights, dtype=np.float64).reshape(-1), 8)
    raw = ",".join(f"{x:.8f}" for x in rounded)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def check_kill_switch_active(active: bool, reason: str) -> RuleVerdict:
    if active:
        return RuleVerdict(
            rule="kill_switch",
            ok=False,
            action="reject",
            message=f"kill switch active: {reason or 'halted'}; manual reset required",
        )
    return RuleVerdict(rule="kill_switch", ok=True, action="pass", message="inactive")


def check_market_data(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    """Missing / non-finite / non-positive prices → reject."""
    if ctx.prices.size == 0 or not np.isfinite(ctx.prices).all():
        return RuleVerdict(
            rule="stale_or_invalid_market_data",
            ok=False,
            action="reject",
            message="missing or non-finite market prices",
        )
    if np.any(ctx.prices <= 0.0):
        return RuleVerdict(
            rule="stale_or_invalid_market_data",
            ok=False,
            action="reject",
            message="non-positive market prices",
        )
    # Stale timestamps vs now
    for ts in ctx.price_timestamps:
        age = (ctx.now - ts).total_seconds()
        if age > settings.max_data_age_seconds:
            return RuleVerdict(
                rule="stale_data_rejection",
                ok=False,
                action="reject",
                message=(
                    f"stale market data age={age:.1f}s > "
                    f"max_data_age_seconds={settings.max_data_age_seconds}"
                ),
            )
        if age < -1.0:
            return RuleVerdict(
                rule="stale_or_invalid_market_data",
                ok=False,
                action="reject",
                message="price timestamp in the future relative to review clock",
            )
    return RuleVerdict(
        rule="stale_or_invalid_market_data",
        ok=True,
        action="pass",
        message="market data present and fresh",
    )


def check_live_database(ctx: RiskReviewContext) -> RuleVerdict:
    if ctx.trading_mode != "live":
        return RuleVerdict(rule="database_available", ok=True, action="pass", message="n/a")
    if ctx.database_available is not True:
        return RuleVerdict(
            rule="database_available",
            ok=False,
            action="reject",
            message="database unavailable — live trading rejected (fail-closed)",
        )
    return RuleVerdict(rule="database_available", ok=True, action="pass", message="ok")


def check_live_account(ctx: RiskReviewContext) -> RuleVerdict:
    if ctx.trading_mode != "live":
        return RuleVerdict(rule="account_confirmed", ok=True, action="pass", message="n/a")
    if ctx.account_confirmed is not True:
        return RuleVerdict(
            rule="account_confirmed",
            ok=False,
            action="reject",
            message="account state not confirmed — live trading rejected (fail-closed)",
        )
    return RuleVerdict(rule="account_confirmed", ok=True, action="pass", message="ok")


def check_daily_loss(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    limit = settings.daily_loss_limit
    if limit is None:
        return RuleVerdict(
            rule="maximum_daily_loss",
            ok=False,
            action="kill",
            message="daily_loss_limit not configured — fail-closed",
        )
    if ctx.day_pnl_frac <= -float(limit):
        return RuleVerdict(
            rule="maximum_daily_loss",
            ok=False,
            action="kill",
            message=(
                f"daily loss {ctx.day_pnl_frac:.4%} breached limit -{float(limit):.4%}"
            ),
        )
    return RuleVerdict(rule="maximum_daily_loss", ok=True, action="pass", message="ok")


def check_weekly_loss(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.week_pnl_frac <= -float(settings.weekly_loss_limit):
        return RuleVerdict(
            rule="maximum_weekly_loss",
            ok=False,
            action="kill",
            message=(
                f"weekly loss {ctx.week_pnl_frac:.4%} breached "
                f"limit -{settings.weekly_loss_limit:.4%}"
            ),
        )
    return RuleVerdict(rule="maximum_weekly_loss", ok=True, action="pass", message="ok")


def check_drawdown(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.peak_nav <= 0:
        return RuleVerdict(
            rule="maximum_portfolio_drawdown",
            ok=False,
            action="kill",
            message="invalid peak NAV for drawdown check",
        )
    dd = max(0.0, (ctx.peak_nav - ctx.current_nav) / ctx.peak_nav)
    if dd > settings.max_drawdown_limit:
        return RuleVerdict(
            rule="maximum_portfolio_drawdown",
            ok=False,
            action="kill",
            message=f"drawdown {dd:.4%} exceeded limit {settings.max_drawdown_limit:.4%}",
        )
    return RuleVerdict(rule="maximum_portfolio_drawdown", ok=True, action="pass", message="ok")


def check_abnormal_price_jumps(
    prices: np.ndarray,
    prior_prices: np.ndarray | None,
    settings: RiskSettings,
) -> RuleVerdict:
    if prior_prices is None:
        return RuleVerdict(
            rule="abnormal_price_rejection",
            ok=True,
            action="pass",
            message="no prior prices supplied",
        )
    prior = np.asarray(prior_prices, dtype=np.float64).reshape(-1)
    px = np.asarray(prices, dtype=np.float64).reshape(-1)
    if prior.shape != px.shape or np.any(prior <= 0):
        return RuleVerdict(
            rule="abnormal_price_rejection",
            ok=False,
            action="reject",
            message="invalid prior prices for jump check",
        )
    jumps = np.abs(px / prior - 1.0)
    if float(np.max(jumps)) > settings.abnormal_price_jump:
        return RuleVerdict(
            rule="abnormal_price_rejection",
            ok=False,
            action="reject",
            message=(
                f"abnormal price jump {float(np.max(jumps)):.2%} > "
                f"{settings.abnormal_price_jump:.2%}"
            ),
        )
    return RuleVerdict(rule="abnormal_price_rejection", ok=True, action="pass", message="ok")


def check_spread(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.bid_ask_spread_bps is None:
        return RuleVerdict(
            rule="spread_threshold",
            ok=True,
            action="pass",
            message="spreads not provided",
        )
    worst = float(np.max(ctx.bid_ask_spread_bps))
    if worst > settings.max_spread_bps:
        return RuleVerdict(
            rule="spread_threshold",
            ok=False,
            action="reject",
            message=f"spread {worst:.1f} bps exceeds max_spread_bps={settings.max_spread_bps}",
        )
    return RuleVerdict(rule="spread_threshold", ok=True, action="pass", message="ok")


def check_market_hours(ctx: RiskReviewContext) -> RuleVerdict:
    if not ctx.market_is_open:
        return RuleVerdict(
            rule="market_hours_check",
            ok=False,
            action="reject",
            message="market closed — trading rejected",
        )
    return RuleVerdict(rule="market_hours_check", ok=True, action="pass", message="open")


def check_duplicate_order(
    proposed: np.ndarray, last_fingerprint: str | None
) -> RuleVerdict:
    fp = order_fingerprint(proposed)
    if last_fingerprint is not None and fp == last_fingerprint:
        return RuleVerdict(
            rule="duplicate_order_prevention",
            ok=False,
            action="reject",
            message="duplicate order fingerprint matches last submission",
        )
    return RuleVerdict(
        rule="duplicate_order_prevention",
        ok=True,
        action="pass",
        message=f"fingerprint={fp}",
    )


def check_max_orders_per_day(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.orders_submitted_today >= settings.max_orders_per_day:
        return RuleVerdict(
            rule="maximum_orders_per_day",
            ok=False,
            action="reject",
            message=(
                f"orders today {ctx.orders_submitted_today} >= "
                f"max_orders_per_day={settings.max_orders_per_day}"
            ),
        )
    return RuleVerdict(rule="maximum_orders_per_day", ok=True, action="pass", message="ok")


def check_consecutive_losses(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.consecutive_losses >= settings.max_consecutive_losses:
        return RuleVerdict(
            rule="maximum_consecutive_loss_safeguard",
            ok=False,
            action="reject",
            message=(
                f"consecutive losses {ctx.consecutive_losses} >= "
                f"max_consecutive_losses={settings.max_consecutive_losses}"
            ),
        )
    return RuleVerdict(
        rule="maximum_consecutive_loss_safeguard", ok=True, action="pass", message="ok"
    )


def check_volatility_target(ctx: RiskReviewContext, settings: RiskSettings) -> RuleVerdict:
    if ctx.realized_volatility > settings.max_volatility_target:
        return RuleVerdict(
            rule="maximum_volatility_target",
            ok=False,
            action="reject",
            message=(
                f"realized volatility {ctx.realized_volatility:.4f} > "
                f"max_volatility_target={settings.max_volatility_target}"
            ),
        )
    return RuleVerdict(rule="maximum_volatility_target", ok=True, action="pass", message="ok")


def _resize_caps(
    weights: np.ndarray,
    *,
    max_position: float,
    max_exposure: float,
    min_cash: float,
) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, max_position)
    max_invested = max(0.0, 1.0 - min_cash)
    exposure_cap = min(max_exposure, max_invested)
    gross = float(np.sum(w))
    if gross > exposure_cap and gross > 0:
        w = w * (exposure_cap / gross)
    return w


def check_position_and_exposure(
    weights: np.ndarray, settings: RiskSettings
) -> RuleVerdict:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if float(np.max(w)) > settings.max_position_size + 1e-12:
        resized = _resize_caps(
            w,
            max_position=settings.max_position_size,
            max_exposure=settings.max_portfolio_exposure,
            min_cash=settings.min_cash_reserve,
        )
        return RuleVerdict(
            rule="maximum_position_weight",
            ok=False,
            action="resize",
            message="position weight exceeded max_position_size; resized",
            resized_weights=resized,
        )
    return RuleVerdict(rule="maximum_position_weight", ok=True, action="pass", message="ok")


def check_portfolio_exposure(weights: np.ndarray, settings: RiskSettings) -> RuleVerdict:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    gross = float(np.sum(np.abs(w)))
    max_invested = max(0.0, 1.0 - settings.min_cash_reserve)
    cap = min(settings.max_portfolio_exposure, max_invested)
    if gross > cap + 1e-12:
        resized = _resize_caps(
            w,
            max_position=settings.max_position_size,
            max_exposure=settings.max_portfolio_exposure,
            min_cash=settings.min_cash_reserve,
        )
        return RuleVerdict(
            rule="maximum_portfolio_exposure",
            ok=False,
            action="resize",
            message="portfolio exposure exceeded; resized",
            resized_weights=resized,
        )
    return RuleVerdict(rule="maximum_portfolio_exposure", ok=True, action="pass", message="ok")


def check_min_cash_reserve(weights: np.ndarray, settings: RiskSettings) -> RuleVerdict:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    cash_frac = 1.0 - float(np.sum(np.clip(w, 0.0, None)))
    if cash_frac + 1e-12 < settings.min_cash_reserve:
        resized = _resize_caps(
            w,
            max_position=settings.max_position_size,
            max_exposure=settings.max_portfolio_exposure,
            min_cash=settings.min_cash_reserve,
        )
        return RuleVerdict(
            rule="minimum_cash_reserve",
            ok=False,
            action="resize",
            message="cash reserve below minimum; resized",
            resized_weights=resized,
        )
    return RuleVerdict(rule="minimum_cash_reserve", ok=True, action="pass", message="ok")


def check_sector_exposure(
    weights: np.ndarray,
    symbols: tuple[str, ...],
    sector_by_symbol: dict[str, str],
    settings: RiskSettings,
) -> RuleVerdict:
    if not sector_by_symbol:
        return RuleVerdict(
            rule="maximum_sector_exposure",
            ok=True,
            action="pass",
            message="no sector classification provided",
        )
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    sector_sums: dict[str, float] = {}
    for i, sym in enumerate(symbols):
        sector = sector_by_symbol.get(sym, sector_by_symbol.get(sym.upper(), ""))
        if not sector:
            continue
        sector_sums[sector] = sector_sums.get(sector, 0.0) + float(max(w[i], 0.0))
    worst_sector = ""
    worst = 0.0
    for sector, total in sector_sums.items():
        if total > worst:
            worst = total
            worst_sector = sector
    if worst > settings.max_sector_exposure + 1e-12:
        # Scale down names in the offending sector proportionally.
        resized = w.copy()
        scale = settings.max_sector_exposure / worst
        for i, sym in enumerate(symbols):
            sector = sector_by_symbol.get(sym, sector_by_symbol.get(sym.upper(), ""))
            if sector == worst_sector:
                resized[i] = float(resized[i]) * scale
        return RuleVerdict(
            rule="maximum_sector_exposure",
            ok=False,
            action="resize",
            message=(
                f"sector {worst_sector!r} exposure {worst:.4f} > "
                f"{settings.max_sector_exposure}; resized"
            ),
            resized_weights=resized,
        )
    return RuleVerdict(rule="maximum_sector_exposure", ok=True, action="pass", message="ok")


def check_order_notional(
    weights: np.ndarray,
    current_weights: np.ndarray,
    portfolio_value: float,
    settings: RiskSettings,
) -> RuleVerdict:
    delta = np.abs(
        np.asarray(weights, dtype=np.float64) - np.asarray(current_weights, dtype=np.float64)
    )
    notionals = delta * float(portfolio_value)
    worst = float(np.max(notionals)) if notionals.size else 0.0
    if worst > settings.max_order_notional + 1e-9:
        # Scale trade deltas so max leg notional fits.
        scale = settings.max_order_notional / worst
        cur = np.asarray(current_weights, dtype=np.float64)
        tgt = np.asarray(weights, dtype=np.float64)
        resized = cur + (tgt - cur) * scale
        resized = _resize_caps(
            resized,
            max_position=settings.max_position_size,
            max_exposure=settings.max_portfolio_exposure,
            min_cash=settings.min_cash_reserve,
        )
        return RuleVerdict(
            rule="maximum_order_notional",
            ok=False,
            action="resize",
            message=(
                f"order notional {worst:.2f} exceeds "
                f"max_order_notional={settings.max_order_notional}; resized"
            ),
            resized_weights=resized,
        )
    return RuleVerdict(rule="maximum_order_notional", ok=True, action="pass", message="ok")


def check_daily_turnover(
    weights: np.ndarray,
    current_weights: np.ndarray,
    settings: RiskSettings,
) -> RuleVerdict:
    turnover = 0.5 * float(
        np.sum(
            np.abs(
                np.asarray(weights, dtype=np.float64)
                - np.asarray(current_weights, dtype=np.float64)
            )
        )
    )
    if turnover > settings.max_daily_turnover + 1e-12:
        cur = np.asarray(current_weights, dtype=np.float64)
        tgt = np.asarray(weights, dtype=np.float64)
        scale = settings.max_daily_turnover / turnover
        resized = cur + (tgt - cur) * scale
        return RuleVerdict(
            rule="maximum_daily_turnover",
            ok=False,
            action="resize",
            message=(
                f"turnover {turnover:.4f} exceeds "
                f"max_daily_turnover={settings.max_daily_turnover}; resized"
            ),
            resized_weights=resized,
        )
    return RuleVerdict(rule="maximum_daily_turnover", ok=True, action="pass", message="ok")


def validate_proposed_weights(target: TargetPortfolio) -> RuleVerdict:
    w = np.asarray(target.weights, dtype=np.float64).reshape(-1)
    if w.size == 0 or not np.isfinite(w).all():
        return RuleVerdict(
            rule="valid_weights",
            ok=False,
            action="reject",
            message="non-finite or empty proposed weights",
        )
    if np.any(w < -1e-12):
        return RuleVerdict(
            rule="valid_weights",
            ok=False,
            action="reject",
            message="negative weights not permitted (long-only risk book)",
        )
    return RuleVerdict(rule="valid_weights", ok=True, action="pass", message="ok")


# Silence unused import if datetime only used in annotations elsewhere
_ = datetime
