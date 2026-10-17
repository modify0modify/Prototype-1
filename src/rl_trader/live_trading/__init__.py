"""Live-trading orchestration — disabled by default, Stage 0 shadow-only.

This package is **not** part of the research surface. RL agents must never
import it. No real broker API calls are made; the live adapter is a gated
sandbox/stub.
"""

from rl_trader.live_trading.activation import ManualLiveController
from rl_trader.live_trading.capital import (
    CapitalAutoIncreaseError,
    CapitalLimitGuard,
)
from rl_trader.live_trading.isolation import (
    AGENT_FORBIDDEN_CONTROL_ATTRS,
    ProtectedLiveControls,
    assert_agents_cannot_write_controls,
    ignore_agent_control_attempt,
)
from rl_trader.live_trading.routing import (
    LiveRoutingError,
    OrderRouter,
    RoutingMode,
)
from rl_trader.live_trading.safety import LiveSafetyError, evaluate_submit_safety
from rl_trader.live_trading.stages import (
    RolloutStage,
    StagePolicy,
    effective_capital_limit,
    live_orders_permitted,
)

__all__ = [
    "AGENT_FORBIDDEN_CONTROL_ATTRS",
    "CapitalAutoIncreaseError",
    "CapitalLimitGuard",
    "LiveRoutingError",
    "LiveSafetyError",
    "ManualLiveController",
    "OrderRouter",
    "ProtectedLiveControls",
    "RolloutStage",
    "RoutingMode",
    "StagePolicy",
    "assert_agents_cannot_write_controls",
    "effective_capital_limit",
    "evaluate_submit_safety",
    "ignore_agent_control_attempt",
    "live_orders_permitted",
]
