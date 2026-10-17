"""Live execution orchestration — gated sandbox, Stage 0 shadow-only.

Research modules must never import this module.
"""

from __future__ import annotations

import inspect
from typing import Any

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.guards import assert_live_trading_allowed
from rl_trader.brokers.live import LiveBroker
from rl_trader.config.settings import Settings
from rl_trader.live_trading.activation import ManualLiveController
from rl_trader.live_trading.isolation import (
    ProtectedLiveControls,
    ignore_agent_control_attempt,
)
from rl_trader.portfolio.targets import TargetPortfolio
from rl_trader.risk.decisions import RiskDecision
from rl_trader.risk.engine import RiskEngine
from rl_trader.utils.boundaries import guard_live_import


def _enforce_caller_boundary() -> None:
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            module_name = caller.f_globals.get("__name__", "")
            if isinstance(module_name, str) and module_name.startswith("rl_trader."):
                guard_live_import(module_name)
            caller = caller.f_back
    finally:
        del frame


class LiveExecutor:
    """Risk → live broker path (gated; sandbox/stub; Stage 0 shadow-only)."""

    def __init__(
        self,
        settings: Settings,
        risk_engine: RiskEngine | None = None,
        *,
        manual: ManualLiveController | None = None,
    ) -> None:
        _enforce_caller_boundary()
        assert_live_trading_allowed(settings)
        self.settings = settings
        self.risk_engine = risk_engine or RiskEngine(settings=settings.risk)
        self.manual = manual or ManualLiveController(settings)
        self.broker = LiveBroker(settings, manual=self.manual)
        self.controls = ProtectedLiveControls(
            app_mode=settings.app_mode.value,
            live_trading_enabled=settings.live_trading_enabled,
            broker_environment=settings.broker.environment,
            max_live_capital=settings.live.max_live_capital,
            max_order_value=settings.live.max_order_value,
            daily_loss_limit=settings.risk.daily_loss_limit,
            max_drawdown_limit=settings.risk.max_drawdown_limit,
            approval_status_required="live",
            kill_switch_active=self.risk_engine.kill_switch.is_active,
            manual_activated=self.manual.is_activated,
            rollout_stage=int(settings.live.rollout_stage),
            account_id=settings.live.account_id,
        )

    def execute(
        self,
        target: TargetPortfolio,
        *,
        agent_control_overrides: dict[str, Any] | None = None,
    ) -> BrokerFill:
        """Review with risk engine and submit to live broker (shadow at Stage 0).

        Any agent-supplied control overrides are ignored.
        """
        _enforce_caller_boundary()
        assert_live_trading_allowed(self.settings)
        self.manual.require_activated()
        # Agent isolation: discard attempted control mutations.
        self.controls = ignore_agent_control_attempt(
            self.controls, agent_control_overrides
        )
        result = self.risk_engine.review(target)
        if result.decision is RiskDecision.REJECTED or result.portfolio is None:
            raise ValueError(
                f"risk rejected target — live broker not called: {result.reason}"
            )
        return self.broker.submit_target(result)
