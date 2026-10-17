"""Required live-trading integration tests (live remains inactive by default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_trader.brokers.types import Order, OrderRequest, OrderSide, OrderStatus, OrderType
from rl_trader.config import (
    LIVE_TRADING_CONFIRMATION_PHRASE,
    AppSettings,
    LiveTradingBlockedError,
    RuntimeMode,
    assert_live_trading_allowed,
    evaluate_live_activation,
    is_live_trading_blocked,
)
from rl_trader.config.models import (
    BrokerSettings,
    DataSettings,
    LiveTradingSettings,
    RiskSettings,
)
from rl_trader.live_trading.activation import ManualLiveController
from rl_trader.live_trading.capital import CapitalAutoIncreaseError, CapitalLimitGuard
from rl_trader.live_trading.isolation import (
    ProtectedLiveControls,
    assert_agents_cannot_write_controls,
    ignore_agent_control_attempt,
)
from rl_trader.live_trading.routing import OrderRouter, RoutingMode
from rl_trader.live_trading.stages import RolloutStage, live_orders_permitted


def _write_live_manifest(tmp_path: Path) -> Path:
    from rl_trader.approval.testing import write_sealed_approved_manifest

    sealed = write_sealed_approved_manifest(
        tmp_path / "approved", tier="live", model_id="live-integ"
    )
    path = tmp_path / "manifest.json"
    path.write_text(sealed.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _complete_live_settings(tmp_path: Path, **live_overrides: object) -> AppSettings:
    """Settings with every safety variable present (still Stage 0 by default)."""
    manifest = _write_live_manifest(tmp_path)
    live_kwargs: dict[str, object] = {
        "acknowledgement": LIVE_TRADING_CONFIRMATION_PHRASE,
        "account_id": "LIVEACCT999",
        "max_live_capital": 10_000.0,
        "max_order_value": 500.0,
        "approved_model_manifest": manifest,
        "approved_account_allowlist": ("LIVEACCT999",),
        "rollout_stage": 0,
        "manual_activated": False,
        "shadow_state_dir": tmp_path / "shadow",
    }
    live_kwargs.update(live_overrides)
    return AppSettings(
        _env_file=None,
        app_mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        data=DataSettings(survivorship_bias_acknowledged=True),
        broker=BrokerSettings(environment="live"),
        risk=RiskSettings(
            max_position_size=0.1,
            max_portfolio_exposure=0.5,
            daily_loss_limit=0.02,
            max_drawdown_limit=0.1,
        ),
        live=LiveTradingSettings(**live_kwargs),  # type: ignore[arg-type]
    )


class _SpyBroker:
    """Minimal submit spy for routing tests."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[OrderRequest] = []

    def submit_order(self, request: OrderRequest) -> Order:
        self.calls.append(request)
        return Order(
            order_id=f"{self.mode}-1",
            symbol=request.symbol,
            side=request.side,
            qty=request.qty,
            order_type=request.order_type,
            status=OrderStatus.ACCEPTED,
        )


# ---------------------------------------------------------------------------
# 1. Paper orders cannot reach the live adapter
# ---------------------------------------------------------------------------


def test_paper_orders_cannot_reach_live_adapter() -> None:
    paper = _SpyBroker("paper")
    live = _SpyBroker("live")
    router = OrderRouter(paper=paper, live=live)
    req = OrderRequest(symbol="SPY", side=OrderSide.BUY, qty=1.0, order_type=OrderType.MARKET)
    order = router.submit(RoutingMode.PAPER, req)
    assert order.order_id.startswith("paper")
    assert len(paper.calls) == 1
    assert len(live.calls) == 0
    router.assert_paper_never_reached_live()
    assert router.live_submit_calls == 0


# ---------------------------------------------------------------------------
# 2. Live mode cannot activate with incomplete safety variables
# ---------------------------------------------------------------------------


def test_live_cannot_activate_with_incomplete_safety_variables(tmp_path: Path) -> None:
    # Default AppSettings — live disabled / incomplete.
    defaults = AppSettings(_env_file=None)
    assert defaults.live_trading_enabled is False
    assert is_live_trading_blocked(defaults) is True
    with pytest.raises(LiveTradingBlockedError):
        assert_live_trading_allowed(defaults)

    # Missing allowlist
    incomplete = _complete_live_settings(
        tmp_path, approved_account_allowlist=()
    )
    report = evaluate_live_activation(incomplete)
    assert report.allowed is False
    assert "approved_account_allowlist" in {c.name for c in report.failed}

    # Missing capital
    incomplete2 = _complete_live_settings(tmp_path / "c2")
    incomplete2 = incomplete2.model_copy(
        update={
            "live": incomplete2.live.model_copy(update={"max_live_capital": None}),
        }
    )
    assert "max_live_capital" in {
        c.name for c in evaluate_live_activation(incomplete2).failed
    }


# ---------------------------------------------------------------------------
# 3. Stage 0 never submits live orders
# ---------------------------------------------------------------------------


def test_stage_0_never_submits_live_orders(tmp_path: Path) -> None:
    settings = _complete_live_settings(tmp_path, rollout_stage=0)
    assert live_orders_permitted(settings.live) is False
    assert settings.live.rollout_stage == RolloutStage.SHADOW

    ctrl = ManualLiveController(settings)
    settings = ctrl.activate(LIVE_TRADING_CONFIRMATION_PHRASE)

    from rl_trader.brokers.live import LiveBroker

    broker = LiveBroker(settings, manual=ctrl)
    req = OrderRequest(
        symbol="SPY",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.MARKET,
        idempotency_key="stage0-1",
    )
    order = broker.submit_order(req)
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is not None
    assert "shadow" in order.reject_reason
    assert order.metadata.get("shadow") is True
    assert order.filled_qty == 0.0
    assert len(broker.shadow_log) == 1
    assert broker.shadow_log[0]["shadow"] is True
    # No sandbox "accepted" live order at stage 0.
    assert all(o.status is not OrderStatus.FILLED for o in broker._orders.values())


# ---------------------------------------------------------------------------
# 4. Capital auto-increase impossible
# ---------------------------------------------------------------------------


def test_capital_auto_increase_impossible() -> None:
    guard = CapitalLimitGuard(1_000.0)
    with pytest.raises(CapitalAutoIncreaseError, match="auto-increase forbidden"):
        guard.propose_increase(10_000.0)
    assert guard.limit == 1_000.0
    # Tightening is allowed.
    assert guard.propose_decrease(500.0) == 500.0
    with pytest.raises(CapitalAutoIncreaseError):
        guard.propose_decrease(800.0)  # decrease helper refuses loosening via increase path


# ---------------------------------------------------------------------------
# 5. Manual activate / deactivate
# ---------------------------------------------------------------------------


def test_manual_activate_deactivate(tmp_path: Path) -> None:
    settings = _complete_live_settings(tmp_path)
    ctrl = ManualLiveController(settings)
    assert ctrl.is_activated is False

    with pytest.raises(LiveTradingBlockedError, match="acknowledgement"):
        ctrl.activate("WRONG_PHRASE")

    armed = ctrl.activate(LIVE_TRADING_CONFIRMATION_PHRASE, actor="tester")
    assert ctrl.is_activated is True
    assert armed.live.manual_activated is True

    disarmed = ctrl.deactivate(actor="tester")
    assert ctrl.is_activated is False
    assert disarmed.live.manual_activated is False
    with pytest.raises(LiveTradingBlockedError, match="disarmed"):
        ctrl.require_activated()


def test_live_remains_inactive_by_default() -> None:
    settings = AppSettings(_env_file=None)
    assert settings.app_mode is RuntimeMode.PAPER
    assert settings.live_trading_enabled is False
    assert settings.live.rollout_stage == 0
    assert settings.live.manual_activated is False
    assert is_live_trading_blocked(settings) is True


def test_agent_control_overrides_ignored() -> None:
    controls = ProtectedLiveControls(
        app_mode="live",
        live_trading_enabled=True,
        broker_environment="live",
        max_live_capital=1_000.0,
        max_order_value=100.0,
        daily_loss_limit=0.02,
        max_drawdown_limit=0.1,
        approval_status_required="live",
        kill_switch_active=False,
        manual_activated=True,
        rollout_stage=0,
        account_id="X",
    )
    out = ignore_agent_control_attempt(
        controls,
        {
            "max_live_capital": 1_000_000.0,
            "kill_switch_active": False,
            "rollout_stage": 3,
            "live_trading_enabled": True,
        },
    )
    assert out is controls
    assert out.max_live_capital == 1_000.0
    assert out.rollout_stage == 0


def test_agents_cannot_write_protected_controls() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "rl_trader"
    assert_agents_cannot_write_controls(package_root)


def test_live_yaml_template_disabled() -> None:
    from rl_trader.config.settings import get_settings, load_yaml_config

    path = Path("configs/live.yaml")
    data = load_yaml_config(path)
    assert data["live_trading_enabled"] is False
    assert data["live"]["rollout_stage"] == 0
    assert data["live"]["manual_activated"] is False
    settings = get_settings(config_file=path)
    assert settings.live_trading_enabled is False
    assert is_live_trading_blocked(settings) is True
