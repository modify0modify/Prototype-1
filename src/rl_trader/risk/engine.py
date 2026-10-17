"""Deterministic risk engine — independent from the RL agent.

The agent proposes desired target weights only. This engine returns
``approved`` | ``resized`` | ``rejected`` with a full decision record.

This module must never import ``rl_trader.agents``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from rl_trader.config.models import RiskSettings
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk import rules
from rl_trader.risk.context import RiskReviewContext
from rl_trader.risk.decisions import DecisionRecord, RiskDecision, RiskResult, weights_tuple
from rl_trader.risk.kill_switch import KillSwitch
from rl_trader.risk.rules import RuleVerdict

# Backward-compatible alias used by older call sites.
RiskLimits = RiskSettings

__all__ = ["RiskDecision", "RiskEngine", "RiskLimits", "RiskResult"]


@dataclass
class _WorkingOrder:
    weights: np.ndarray
    resized: bool = False
    violated: list[str] | None = None

    def __post_init__(self) -> None:
        if self.violated is None:
            self.violated = []


class RiskEngine:
    """Deterministic target-portfolio gate with fail-closed live checks."""

    def __init__(
        self,
        settings: RiskSettings | None = None,
        *,
        limits: RiskSettings | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        # ``limits`` kept as keyword alias for RiskSettings (legacy RiskLimits).
        self.settings = settings or limits or RiskSettings()
        self.kill_switch = kill_switch or KillSwitch(
            reset_phrase=self.settings.kill_switch_reset_phrase
        )
        self.decision_log: list[DecisionRecord] = []

    def review(
        self,
        target: TargetPortfolio,
        context: RiskReviewContext | None = None,
        *,
        model_id: str | None = None,
        prior_prices: np.ndarray | None = None,
    ) -> RiskResult:
        """Approve, resize, or reject a proposed target portfolio.

        When ``context`` is omitted, the review fails closed (missing market
        data) — callers must supply a snapshot for any real trading path.
        """
        now = datetime.now(UTC)
        cfg_ver = self.settings.configuration_version
        mid = model_id or (context.model_id if context is not None else "unspecified")
        if context is not None and context.config_version:
            cfg_ver = context.config_version

        original = weights_tuple(target) or ()

        def _reject(
            reason: str,
            violated: list[str],
            *,
            cancel_pending: bool = False,
            details: dict[str, Any] | None = None,
        ) -> RiskResult:
            pending = context.pending_order_ids if context is not None else ()
            record = DecisionRecord(
                decision_status=RiskDecision.REJECTED,
                original_order=original,
                approved_order=None,
                violated_rules=tuple(violated),
                reason=reason,
                timestamp=now,
                configuration_version=cfg_ver,
                model_identifier=mid,
                cancel_pending_orders=cancel_pending,
                pending_order_ids_cancelled=pending if cancel_pending else (),
                kill_switch_active=self.kill_switch.is_active,
                details=dict(details or {}),
            )
            self.decision_log.append(record)
            return RiskResult(
                decision=RiskDecision.REJECTED,
                portfolio=None,
                reason=reason,
                record=record,
                details=record.details,
            )

        def _finish(
            status: RiskDecision,
            weights: np.ndarray,
            reason: str,
            violated: list[str],
        ) -> RiskResult:
            portfolio = TargetPortfolio(weights=weights, as_of=target.as_of)
            record = DecisionRecord(
                decision_status=status,
                original_order=original,
                approved_order=weights_tuple(weights),
                violated_rules=tuple(violated),
                reason=reason,
                timestamp=now,
                configuration_version=cfg_ver,
                model_identifier=mid,
                kill_switch_active=self.kill_switch.is_active,
            )
            self.decision_log.append(record)
            return RiskResult(
                decision=status,
                portfolio=portfolio,
                reason=reason,
                record=record,
            )

        # --- Hard stops ---
        v = rules.validate_proposed_weights(target)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_kill_switch_active(
            self.kill_switch.is_active, self.kill_switch.reason
        )
        if not v.ok:
            return _reject(v.message, [v.rule])

        if context is None:
            return _reject(
                "missing market / account context — fail-closed",
                ["stale_or_invalid_market_data"],
            )

        # Live fail-closed infrastructure checks
        v = rules.check_live_database(context)
        if not v.ok:
            return _reject(v.message, [v.rule])
        v = rules.check_live_account(context)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_market_data(context, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        prior = prior_prices if prior_prices is not None else context.prior_prices
        v = rules.check_abnormal_price_jumps(context.prices, prior, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_spread(context, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_market_hours(context)
        if not v.ok:
            return _reject(v.message, [v.rule])

        # Loss / drawdown → kill switch
        loss_checks: list[RuleVerdict] = [
            rules.check_daily_loss(context, self.settings),
            rules.check_weekly_loss(context, self.settings),
            rules.check_drawdown(context, self.settings),
        ]
        for v in loss_checks:
            if v.action == "kill":
                self.kill_switch.trip(
                    v.message,
                    details={"rule": v.rule, "mode": context.trading_mode},
                )
                return _reject(
                    (
                        f"{v.message}; kill switch engaged; pending orders cancelled; "
                        "manual reset required"
                    ),
                    [v.rule, "kill_switch"],
                    cancel_pending=True,
                    details={"kill_switch_events": len(self.kill_switch.events)},
                )

        proposed = np.asarray(target.weights, dtype=np.float64).reshape(-1)
        v = rules.check_duplicate_order(proposed, context.last_order_fingerprint)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_max_orders_per_day(context, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_consecutive_losses(context, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        v = rules.check_volatility_target(context, self.settings)
        if not v.ok:
            return _reject(v.message, [v.rule])

        # --- Soft / resize path ---
        work = _WorkingOrder(weights=proposed.copy())

        def _apply_resize(verdict: RuleVerdict) -> RiskResult | None:
            if verdict.action == "resize" and verdict.resized_weights is not None:
                work.weights = np.asarray(verdict.resized_weights, dtype=np.float64)
                work.resized = True
                assert work.violated is not None
                work.violated.append(verdict.rule)
                return None
            if verdict.action == "reject":
                return _reject(verdict.message, [verdict.rule])
            return None

        resize_steps: tuple[Any, ...] = (
            lambda: rules.check_position_and_exposure(work.weights, self.settings),
            lambda: rules.check_portfolio_exposure(work.weights, self.settings),
            lambda: rules.check_min_cash_reserve(work.weights, self.settings),
            lambda: rules.check_sector_exposure(
                work.weights, context.symbols, context.sector_by_symbol, self.settings
            ),
            lambda: rules.check_order_notional(
                work.weights,
                context.current_weights,
                context.portfolio_value,
                self.settings,
            ),
            lambda: rules.check_daily_turnover(
                work.weights, context.current_weights, self.settings
            ),
        )
        for step in resize_steps:
            verdict = step()
            assert isinstance(verdict, RuleVerdict)
            early = _apply_resize(verdict)
            if early is not None:
                return early

        if work.resized:
            return _finish(
                RiskDecision.RESIZED,
                work.weights,
                "resized to satisfy risk limits: " + ", ".join(work.violated or []),
                work.violated or [],
            )
        return _finish(
            RiskDecision.APPROVED,
            work.weights,
            "within limits",
            [],
        )

    def manual_reset_kill_switch(self, acknowledgement: str) -> None:
        """Operator-only kill-switch clear."""
        self.kill_switch.manual_reset(acknowledgement)
