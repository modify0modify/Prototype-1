"""Live broker adapter (sandbox/stub) — disabled by default, Stage 0 shadow-only.

Implements :class:`~rl_trader.brokers.base.BrokerAdapter`. No real broker HTTP
calls are made. Paper credentials are never read.

Safety
------
Construction requires activation gates. Order submission additionally requires
manual arming and stage policy. Stage 0 records shadow decisions only and never
marks an order as a live fill.
"""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.guards import assert_live_trading_allowed
from rl_trader.brokers.types import (
    Account,
    MarketClock,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)
from rl_trader.config.settings import Settings
from rl_trader.live_trading.activation import ManualLiveController
from rl_trader.live_trading.capital import CapitalLimitGuard
from rl_trader.live_trading.safety import LiveSafetyError, evaluate_submit_safety
from rl_trader.live_trading.stages import (
    RolloutStage,
    effective_capital_limit,
    resolve_stage_policy,
)
from rl_trader.risk.engine import RiskResult
from rl_trader.risk.kill_switch import KillSwitch
from rl_trader.utils.boundaries import guard_live_import


class LiveOrderBlockedError(RuntimeError):
    """Raised when a live (or stage-gated) submit is refused."""


def _enforce_caller_boundary() -> None:
    """Reject imports/calls originating from research packages."""
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


def _utcnow() -> datetime:
    return datetime.now(UTC)


class LiveBroker:
    """Gated live broker sandbox implementing :class:`BrokerAdapter`.

    - Uses **live** credential fields only (never paper keys).
    - Stage 0: ``submit_order`` / ``submit_target`` → shadow log, no live fill.
    - Stage ≥ 1: sandbox acceptance only when manual-armed + all safety layers;
      still no external API calls.
    """

    mode = "live"

    def __init__(
        self,
        settings: Settings,
        *,
        manual: ManualLiveController | None = None,
        kill_switch: KillSwitch | None = None,
        now_fn: Callable[[], datetime] | None = None,
        market_open: bool = True,
        reconciliation_ok: bool = True,
    ) -> None:
        _enforce_caller_boundary()
        assert_live_trading_allowed(settings)
        # Refuse if paper credentials were somehow bound into the live env.
        if settings.broker.environment != "live":
            raise LiveOrderBlockedError("LiveBroker requires broker.environment=live")
        self._settings = settings
        self._manual = manual or ManualLiveController(settings)
        self._kill_switch = kill_switch or KillSwitch(
            reset_phrase=settings.risk.kill_switch_reset_phrase
        )
        self._now_fn = now_fn or _utcnow
        self._market_open = market_open
        self._reconciliation_ok = reconciliation_ok
        self._connected = False
        self._orders: dict[str, Order] = {}
        self._idempotency: dict[str, str] = {}
        self._shadow_log: list[dict[str, Any]] = []
        self._quotes: dict[str, Quote] = {}
        self._capital_guard = CapitalLimitGuard(effective_capital_limit(settings.live))
        # Live credentials are referenced only as opaque presence checks — never logged.
        self._live_key_present = settings.broker.live_api_key is not None
        self._live_secret_present = settings.broker.live_api_secret is not None
        # Explicitly ignore paper credentials.
        _ = settings.broker.paper_api_key
        _ = settings.broker.paper_api_secret

    # ------------------------------------------------------------------
    # BrokerAdapter
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish a sandbox session (no external network)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        self._manual.require_activated()
        self._connected = True

    def disconnect(self) -> None:
        """Tear down the sandbox session."""
        _enforce_caller_boundary()
        self._connected = False

    def get_account(self) -> Account:
        """Return sandbox account snapshot capped by configured capital."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        cap = self._capital_guard.limit or 0.0
        return Account(
            cash=float(cap),
            equity=float(cap),
            buying_power=float(cap),
            currency="USD",
            account_id=self._settings.live.account_id or "live-sandbox",
        )

    def get_positions(self) -> list[Position]:
        """Sandbox starts flat — no real positions."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        return []

    def get_open_orders(self) -> list[Order]:
        """Return non-terminal sandbox orders."""
        _enforce_caller_boundary()
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    def get_latest_quote(self, symbol: str) -> Quote:
        """Return an injected/sandbox quote (tests set quotes via ``set_quote``)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        key = symbol.upper()
        if key not in self._quotes:
            raise LiveOrderBlockedError(f"no sandbox quote for {key}")
        return self._quotes[key]

    def set_quote(self, quote: Quote) -> None:
        """Inject a sandbox quote (tests / shadow loop)."""
        self._quotes[quote.symbol.upper()] = quote

    def submit_order(self, request: OrderRequest) -> Order:
        """Submit with full safety stack; Stage 0 → shadow only (never live fill)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        self._manual.require_activated()

        now = self._now_fn()
        quote = self._quotes.get(request.symbol.upper())
        report = evaluate_submit_safety(
            self._settings,
            request,
            manual=self._manual,
            kill_switch=self._kill_switch,
            quote=quote,
            now=now,
            market_open=self._market_open,
            reconciliation_ok=self._reconciliation_ok,
            seen_idempotency_keys=set(self._idempotency),
        )
        if not report.allowed:
            failed = "; ".join(f"{c.name}:{c.detail}" for c in report.failed)
            raise LiveSafetyError(f"live submit blocked: {failed}")

        # Duplicate idempotency → return prior order.
        key = request.idempotency_key or request.client_order_id
        if key and key in self._idempotency:
            return self._orders[self._idempotency[key]]

        policy = resolve_stage_policy(
            self._settings.live,
            base_universe=self._settings.data.universe,
        )
        order_id = f"live-sandbox-{uuid.uuid4().hex[:12]}"

        # Stage 0 — shadow only: never a live fill.
        if policy.stage is RolloutStage.SHADOW or report.shadow_only:
            shadow = {
                "order_id": order_id,
                "symbol": request.symbol,
                "side": str(request.side),
                "qty": request.qty,
                "stage": int(policy.stage),
                "shadow": True,
                "timestamp": now.isoformat(),
                "detail": policy.detail,
            }
            self._shadow_log.append(shadow)
            self._persist_shadow(shadow)
            order = Order(
                order_id=order_id,
                symbol=request.symbol.upper(),
                side=request.side,
                qty=float(request.qty),
                order_type=request.order_type,
                status=OrderStatus.REJECTED,
                filled_qty=0.0,
                idempotency_key=request.idempotency_key,
                client_order_id=request.client_order_id,
                created_at=now,
                updated_at=now,
                reject_reason="stage_0_shadow_only — no live orders",
                metadata={"shadow": True, "stage": int(policy.stage)},
            )
            self._orders[order_id] = order
            if key:
                self._idempotency[key] = order_id
            return order

        # Stage ≥ 1 sandbox accept (still no external API).
        notional_price = float(quote.last) if quote is not None else 1.0
        notional = abs(float(request.qty) * notional_price)
        max_order = self._settings.live.max_order_value
        if max_order is not None and notional > max_order:
            raise LiveOrderBlockedError(
                f"order notional {notional} exceeds max_order_value {max_order}"
            )
        if policy.long_only and request.side is OrderSide.SELL:
            raise LiveOrderBlockedError("long_only stage policy blocks sell submits")

        order = Order(
            order_id=order_id,
            symbol=request.symbol.upper(),
            side=request.side,
            qty=float(request.qty),
            order_type=request.order_type or OrderType.MARKET,
            status=OrderStatus.ACCEPTED,
            filled_qty=0.0,
            idempotency_key=request.idempotency_key,
            client_order_id=request.client_order_id,
            created_at=now,
            updated_at=now,
            metadata={
                "sandbox": True,
                "stage": int(policy.stage),
                "live_key_configured": self._live_key_present,
                "external_api_called": False,
            },
        )
        self._orders[order_id] = order
        if key:
            self._idempotency[key] = order_id
        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a sandbox order."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        order = self._orders.get(order_id)
        if order is None:
            raise LiveOrderBlockedError(f"unknown order_id={order_id}")
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            return order
        updated = Order(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type=order.order_type,
            status=OrderStatus.CANCELLED,
            filled_qty=order.filled_qty,
            avg_fill_price=order.avg_fill_price,
            limit_price=order.limit_price,
            idempotency_key=order.idempotency_key,
            client_order_id=order.client_order_id,
            created_at=order.created_at,
            updated_at=self._now_fn(),
            reject_reason=order.reject_reason,
            commission=order.commission,
            metadata=dict(order.metadata),
        )
        self._orders[order_id] = updated
        return updated

    def cancel_all_orders(self) -> list[Order]:
        """Cancel all open sandbox orders."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        return [self.cancel_order(o.order_id) for o in self.get_open_orders()]

    def get_order_status(self, order_id: str) -> Order:
        """Return sandbox order status."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        order = self._orders.get(order_id)
        if order is None:
            raise LiveOrderBlockedError(f"unknown order_id={order_id}")
        return order

    def get_market_clock(self) -> MarketClock:
        """Return sandbox market clock."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        now = self._now_fn()
        return MarketClock(is_open=self._market_open, timestamp=now, timezone="UTC")

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Map risk-approved weights to shadow/sandbox orders (Stage 0 → shadow)."""
        _enforce_caller_boundary()
        assert_live_trading_allowed(self._settings)
        self._manual.require_activated()
        if not risk_result.approved:
            raise LiveOrderBlockedError("risk result not approved — refuse live submit")

        policy = resolve_stage_policy(
            self._settings.live,
            base_universe=self._settings.data.universe,
        )
        now = self._now_fn()
        if policy.stage is RolloutStage.SHADOW or not policy.allow_live_orders:
            shadow = {
                "type": "submit_target",
                "stage": int(policy.stage),
                "shadow": True,
                "weights": list(risk_result.record.approved_order or ()),
                "timestamp": now.isoformat(),
                "detail": policy.detail,
            }
            self._shadow_log.append(shadow)
            self._persist_shadow(shadow)
            return BrokerFill(
                portfolio=risk_result.portfolio,
                filled_at=now,
                mode="live-shadow",
                metadata={"shadow": True, "stage": int(policy.stage), "external_api_called": False},
            )

        # Stage ≥ 1: still sandbox — no external API.
        return BrokerFill(
            portfolio=risk_result.portfolio,
            filled_at=now,
            mode="live-sandbox",
            metadata={
                "sandbox": True,
                "stage": int(policy.stage),
                "external_api_called": False,
            },
        )

    # ------------------------------------------------------------------
    # Introspection (tests / ops)
    # ------------------------------------------------------------------

    @property
    def shadow_log(self) -> tuple[dict[str, Any], ...]:
        """Recorded Stage-0 shadow decisions."""
        return tuple(self._shadow_log)

    @property
    def capital_guard(self) -> CapitalLimitGuard:
        """Capital ceiling guard (auto-increase impossible)."""
        return self._capital_guard

    @property
    def manual(self) -> ManualLiveController:
        """Manual arming controller."""
        return self._manual

    def _persist_shadow(self, record: dict[str, Any]) -> None:
        path = Path(self._settings.live.shadow_state_dir)
        path.mkdir(parents=True, exist_ok=True)
        log_path = path / "shadow_decisions.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
