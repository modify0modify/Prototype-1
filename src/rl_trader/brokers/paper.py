"""Persistent simulated paper broker — no external credentials.

Applies configurable commission, spread, slippage, partial fills, random
rejects, execution delay, and optional market-hours enforcement. State lives
under ``data/interim/paper_broker`` by default.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from rl_trader.brokers.base import BrokerFill
from rl_trader.brokers.ledger import LedgerState, PaperLedger
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
from rl_trader.risk.engine import RiskResult

# Terminal statuses — cannot cancel / re-fill.
_TERMINAL = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    }
)
_OPEN = frozenset(
    {
        OrderStatus.NEW,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
    }
)


class PaperTradingDisabledError(RuntimeError):
    """Legacy gate error — kept for import compatibility.

    The paper broker now applies a real cost model; construction no longer
    raises this by default. Prefer constructing :class:`PaperBroker` directly.
    """


@dataclass(slots=True)
class PaperBrokerConfig:
    """Tunable paper-broker simulation knobs."""

    initial_cash: float = 100_000.0
    commission_bps: float = 1.0
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    # Fraction of remaining qty filled per attempt (1.0 = full fill).
    partial_fill_fraction: float = 1.0
    # Probability of rejecting a newly submitted order (0–1).
    reject_probability: float = 0.0
    # Delay before an accepted order is eligible to fill (seconds).
    execution_delay_seconds: float = 0.0
    enforce_market_hours: bool = False
    allow_fractional_shares: bool = True
    max_quote_age_seconds: float = 60.0
    data_dir: Path = field(default_factory=lambda: Path("data/interim/paper_broker"))
    account_id: str = "paper-sim"
    currency: str = "USD"
    # Simple UTC session windows used when enforce_market_hours is True.
    market_open_hour_utc: int = 14  # ~09:30 ET
    market_close_hour_utc: int = 21  # ~16:00 ET
    rng_seed: int | None = 42


class PaperBroker:
    """Simulated broker with persistent cash, positions, and order ledger."""

    mode = "paper"

    def __init__(
        self,
        config: PaperBrokerConfig | None = None,
        *,
        now_fn: Callable[[], datetime] | None = None,
        allow_incomplete: bool | None = None,
    ) -> None:
        # ``allow_incomplete`` retained for call-site compatibility; ignored.
        _ = allow_incomplete
        self.config = config or PaperBrokerConfig()
        self._now_fn = now_fn or (lambda: datetime.now(UTC))
        self._ledger = PaperLedger(self.config.data_dir)
        self._state: LedgerState | None = None
        self._quotes: dict[str, Quote] = {}
        self._connected = False
        self._rng = np.random.default_rng(self.config.rng_seed)
        self.fills: list[BrokerFill] = []
        self.incomplete = False

    # ------------------------------------------------------------------ #
    # Session
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Load persistent state and reconcile the ledger."""
        self._state = self._ledger.load(
            default_cash=self.config.initial_cash,
            account_id=self.config.account_id,
        )
        self._connected = True
        self.reconcile()
        self._persist()

    def disconnect(self) -> None:
        """Flush state and mark the session closed."""
        if self._connected and self._state is not None:
            self._persist()
        self._connected = False

    @property
    def connected(self) -> bool:
        """Whether ``connect`` has been called."""
        return self._connected

    # ------------------------------------------------------------------ #
    # Quotes / clock helpers (test + orchestration injection)
    # ------------------------------------------------------------------ #

    def set_quote(self, quote: Quote) -> None:
        """Inject or refresh a quote used by the simulator."""
        self._quotes[quote.symbol.upper()] = quote

    def set_quotes(self, quotes: Mapping[str, Quote] | Sequence[Quote]) -> None:
        """Bulk-inject quotes."""
        if isinstance(quotes, Mapping):
            for sym, q in quotes.items():
                self._quotes[str(sym).upper()] = q
        else:
            for q in quotes:
                self._quotes[q.symbol.upper()] = q

    def advance_time(self, seconds: float = 0.0) -> list[Order]:
        """Process due orders after an optional wall-clock advance.

        When ``now_fn`` is a controllable clock, callers should advance that
        clock themselves; ``seconds`` is only used when the default wall clock
        is active (no-op hint). Returns orders that transitioned.
        """
        _ = seconds
        self._require_connected()
        return self._process_due_orders()

    # ------------------------------------------------------------------ #
    # BrokerAdapter API
    # ------------------------------------------------------------------ #

    def get_account(self) -> Account:
        """Cash, mark-to-market equity, and buying power."""
        state = self._require_connected()
        self._process_due_orders()
        equity = self._equity(state)
        return Account(
            cash=state.cash,
            equity=equity,
            buying_power=max(0.0, state.cash),
            currency=state.currency,
            account_id=state.account_id,
        )

    def get_positions(self) -> list[Position]:
        """Open positions marked at latest quotes when available."""
        state = self._require_connected()
        self._process_due_orders()
        out: list[Position] = []
        for sym, pos in sorted(state.positions.items()):
            qty = float(pos["qty"])
            if abs(qty) < 1e-12:
                continue
            avg = float(pos["avg_price"])
            px = self._mark_price(sym, avg)
            mv = qty * px
            out.append(
                Position(
                    symbol=sym,
                    qty=qty,
                    avg_price=avg,
                    market_price=px,
                    market_value=mv,
                    unrealized_pnl=mv - qty * avg,
                )
            )
        return out

    def get_open_orders(self) -> list[Order]:
        """Non-terminal orders."""
        state = self._require_connected()
        self._process_due_orders()
        return [o for o in state.orders.values() if o.status in _OPEN]

    def get_latest_quote(self, symbol: str) -> Quote:
        """Return injected quote or raise if missing."""
        self._require_connected()
        key = symbol.upper()
        if key not in self._quotes:
            raise KeyError(f"no quote for {key}; call set_quote first")
        return self._quotes[key]

    def submit_order(self, request: OrderRequest) -> Order:
        """Validate, accept (or reject), and optionally fill an order."""
        state = self._require_connected()
        now = self._now()

        if request.idempotency_key:
            existing_id = state.idempotency.get(request.idempotency_key)
            if existing_id is not None and existing_id in state.orders:
                return state.orders[existing_id]

        order = Order(
            order_id=self._new_order_id(),
            symbol=request.symbol.upper(),
            side=request.side,
            qty=float(request.qty),
            order_type=request.order_type,
            status=OrderStatus.NEW,
            limit_price=request.limit_price,
            idempotency_key=request.idempotency_key,
            client_order_id=request.client_order_id,
            created_at=now,
            updated_at=now,
            metadata={"time_in_force": request.time_in_force},
        )

        reject = self._validate_new_order(order, now)
        if reject is not None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = reject
            order.updated_at = now
            self._store_order(order)
            self._persist()
            return order

        if self.config.reject_probability > 0.0:
            if float(self._rng.random()) < self.config.reject_probability:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "simulated_random_reject"
                order.updated_at = now
                self._store_order(order)
                self._persist()
                return order

        order.status = OrderStatus.ACCEPTED
        order.updated_at = now
        order.metadata["accepted_at"] = now.isoformat()
        order.metadata["eligible_at"] = (
            now + timedelta(seconds=self.config.execution_delay_seconds)
        ).isoformat()
        self._store_order(order)
        self._persist()

        if self.config.execution_delay_seconds <= 0.0:
            self._try_fill(order.order_id)
            self._persist()
            return state.orders[order.order_id]

        return order

    def cancel_order(self, order_id: str) -> Order:
        """Cancel a working order."""
        state = self._require_connected()
        self._process_due_orders()
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown order_id={order_id}")
        if order.status in _TERMINAL:
            return order
        now = self._now()
        order.status = OrderStatus.CANCELLED
        order.updated_at = now
        order.metadata["cancelled_at"] = now.isoformat()
        self._persist()
        return order

    def cancel_all_orders(self) -> list[Order]:
        """Cancel every open order."""
        self._require_connected()
        cancelled: list[Order] = []
        for order in list(self.get_open_orders()):
            cancelled.append(self.cancel_order(order.order_id))
        return cancelled

    def get_order_status(self, order_id: str) -> Order:
        """Return one order after processing due fills."""
        state = self._require_connected()
        self._process_due_orders()
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"unknown order_id={order_id}")
        return order

    def get_market_clock(self) -> MarketClock:
        """UTC session clock (simplified)."""
        now = self._now()
        is_open = self._is_market_open(now)
        return MarketClock(
            is_open=is_open,
            timestamp=now,
            next_open=None,
            next_close=None,
            timezone="UTC",
        )

    def submit_target(self, risk_result: RiskResult) -> BrokerFill:
        """Translate a risk-approved weight vector into market orders.

        Requires symbols to be provided via ``risk_result.details['symbols']``
        (or ``record.details['symbols']``) matching weight order, and quotes
        for each symbol. Applies the paper cost model on every fill.
        """
        self._require_connected()
        if risk_result.portfolio is None:
            raise ValueError(f"cannot paper-trade rejected target: {risk_result.reason}")
        weights = np.asarray(risk_result.portfolio.weights, dtype=np.float64).reshape(-1)
        if not np.isfinite(weights).all():
            raise ValueError("paper broker refuses non-finite weights")

        symbols = self._symbols_from_risk(risk_result, len(weights))
        account = self.get_account()
        equity = account.equity
        submitted: list[str] = []
        for sym, w in zip(symbols, weights, strict=True):
            quote = self.get_latest_quote(sym)
            target_value = float(w) * equity
            target_qty = target_value / max(quote.last, 1e-12)
            current_qty = float(self._require_connected().positions.get(sym, {}).get("qty", 0.0))
            delta = target_qty - current_qty
            if abs(delta) < 1e-8:
                continue
            if not self.config.allow_fractional_shares:
                delta = math.trunc(delta)
                if abs(delta) < 1:
                    continue
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            order = self.submit_order(
                OrderRequest(
                    symbol=sym,
                    side=side,
                    qty=abs(delta),
                    order_type=OrderType.MARKET,
                    idempotency_key=f"target:{risk_result.record.timestamp.isoformat()}:{sym}",
                )
            )
            submitted.append(order.order_id)

        fill = BrokerFill(
            portfolio=risk_result.portfolio,
            filled_at=self._now(),
            mode=self.mode,
            metadata={
                "decision": risk_result.decision.value,
                "credential_profile": "paper",
                "costs_applied": True,
                "incomplete_paper_broker": False,
                "order_ids": submitted,
                "commission_bps": self.config.commission_bps,
                "spread_bps": self.config.spread_bps,
                "slippage_bps": self.config.slippage_bps,
            },
        )
        self.fills.append(fill)
        self._persist()
        return fill

    # ------------------------------------------------------------------ #
    # Reconciliation
    # ------------------------------------------------------------------ #

    def reconcile(self) -> dict[str, Any]:
        """Check ledger invariants; drop zero positions; return a report."""
        state = self._require_connected()
        issues: list[str] = []
        if not math.isfinite(state.cash):
            issues.append("non_finite_cash")
            state.cash = self.config.initial_cash
        if state.cash < -1e-6:
            issues.append("negative_cash")

        cleaned: dict[str, dict[str, float]] = {}
        for sym, pos in state.positions.items():
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_price", 0.0))
            if abs(qty) < 1e-12:
                continue
            if not math.isfinite(qty) or not math.isfinite(avg):
                issues.append(f"bad_position:{sym}")
                continue
            cleaned[sym.upper()] = {"qty": qty, "avg_price": avg}
        state.positions = cleaned

        # Idempotency map must point at known orders.
        stale_keys = [k for k, oid in state.idempotency.items() if oid not in state.orders]
        for k in stale_keys:
            del state.idempotency[k]
            issues.append(f"stale_idempotency:{k}")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "cash": state.cash,
            "n_positions": len(state.positions),
            "n_orders": len(state.orders),
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_connected(self) -> LedgerState:
        if not self._connected or self._state is None:
            raise RuntimeError("PaperBroker is not connected; call connect() first")
        return self._state

    def _now(self) -> datetime:
        ts = self._now_fn()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)

    def _persist(self) -> None:
        state = self._require_connected()
        self._ledger.save(state, now=self._now())

    def _store_order(self, order: Order) -> None:
        state = self._require_connected()
        state.orders[order.order_id] = order
        if order.idempotency_key:
            state.idempotency[order.idempotency_key] = order.order_id

    def _new_order_id(self) -> str:
        return f"paper-{uuid.uuid4().hex[:16]}"

    def _is_market_open(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        hour = now.hour
        return self.config.market_open_hour_utc <= hour < self.config.market_close_hour_utc

    def _validate_new_order(self, order: Order, now: datetime) -> str | None:
        if order.qty <= 0 or not math.isfinite(order.qty):
            return "invalid_qty"
        if not self.config.allow_fractional_shares and abs(order.qty - round(order.qty)) > 1e-9:
            return "fractional_shares_disabled"
        if order.order_type == OrderType.LIMIT and (
            order.limit_price is None or order.limit_price <= 0
        ):
            return "limit_price_required"
        if self.config.enforce_market_hours and not self._is_market_open(now):
            return "market_closed"
        try:
            quote = self.get_latest_quote(order.symbol)
        except KeyError:
            return "missing_quote"
        age = (now - quote.timestamp.astimezone(UTC)).total_seconds()
        if age > self.config.max_quote_age_seconds:
            return "stale_quote"
        if quote.bid <= 0 or quote.ask <= 0 or quote.ask < quote.bid:
            return "invalid_quote"
        # Buying-power / inventory checks at submit time (conservative).
        if order.side == OrderSide.BUY:
            est_px = self._execution_price(order.side, quote)
            notional = est_px * order.qty
            commission = self._commission(notional)
            state = self._require_connected()
            if state.cash + 1e-9 < notional + commission:
                return "insufficient_buying_power"
        else:
            held = float(
                self._require_connected().positions.get(order.symbol, {}).get("qty", 0.0)
            )
            if held + 1e-9 < order.qty:
                return "insufficient_position"
        return None

    def _process_due_orders(self) -> list[Order]:
        state = self._require_connected()
        now = self._now()
        changed: list[Order] = []
        for order_id, order in list(state.orders.items()):
            if order.status not in (OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED):
                continue
            eligible_raw = order.metadata.get("eligible_at")
            if eligible_raw:
                eligible = datetime.fromisoformat(str(eligible_raw))
                if eligible.tzinfo is None:
                    eligible = eligible.replace(tzinfo=UTC)
                if now < eligible:
                    continue
            before = order.status
            self._try_fill(order_id)
            after = state.orders[order_id]
            if after.status != before or after.filled_qty != order.filled_qty:
                changed.append(after)
        if changed:
            self._persist()
        return changed

    def _try_fill(self, order_id: str) -> None:
        state = self._require_connected()
        order = state.orders[order_id]
        if order.status in _TERMINAL:
            return
        now = self._now()
        try:
            quote = self.get_latest_quote(order.symbol)
        except KeyError:
            return

        age = (now - quote.timestamp.astimezone(UTC)).total_seconds()
        if age > self.config.max_quote_age_seconds:
            # Leave working; do not reject after accept unless hours enforced.
            return

        if order.order_type == OrderType.LIMIT:
            assert order.limit_price is not None
            if order.side == OrderSide.BUY and quote.ask > order.limit_price + 1e-12:
                return
            if order.side == OrderSide.SELL and quote.bid < order.limit_price - 1e-12:
                return

        remaining = order.qty - order.filled_qty
        if remaining <= 1e-12:
            order.status = OrderStatus.FILLED
            order.updated_at = now
            return

        frac = min(1.0, max(0.0, float(self.config.partial_fill_fraction)))
        fill_qty = remaining * frac
        if not self.config.allow_fractional_shares:
            fill_qty = math.floor(fill_qty)
        if fill_qty <= 1e-12:
            return

        # Re-check buying power / inventory at fill time.
        px = self._execution_price(order.side, quote)
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if order.side == OrderSide.BUY:
                px = min(px, order.limit_price)
            else:
                px = max(px, order.limit_price)

        notional = px * fill_qty
        commission = self._commission(notional)
        if order.side == OrderSide.BUY:
            cost = notional + commission
            if state.cash + 1e-9 < cost:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "insufficient_buying_power"
                order.updated_at = now
                return
            state.cash -= cost
            self._apply_buy(order.symbol, fill_qty, px)
        else:
            held = float(state.positions.get(order.symbol, {}).get("qty", 0.0))
            if held + 1e-9 < fill_qty:
                order.status = OrderStatus.REJECTED
                order.reject_reason = "insufficient_position"
                order.updated_at = now
                return
            proceeds = notional - commission
            state.cash += proceeds
            self._apply_sell(order.symbol, fill_qty)

        prev_filled = order.filled_qty
        prev_avg = order.avg_fill_price or 0.0
        new_filled = prev_filled + fill_qty
        if new_filled > 0:
            order.avg_fill_price = (
                (prev_avg * prev_filled) + (px * fill_qty)
            ) / new_filled
        order.filled_qty = new_filled
        order.commission += commission
        order.updated_at = now
        if order.filled_qty + 1e-9 >= order.qty:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        self._ledger.append_fill(
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": str(order.side),
                "qty": fill_qty,
                "price": px,
                "commission": commission,
                "timestamp": now.isoformat(),
                "cash_after": state.cash,
            }
        )

    def _execution_price(self, side: OrderSide, quote: Quote) -> float:
        """Mid ± half-spread ± slippage (buys pay up, sells receive down)."""
        mid = (quote.bid + quote.ask) / 2.0
        half_spread = mid * (self.config.spread_bps / 10_000.0) / 2.0
        slip = mid * (self.config.slippage_bps / 10_000.0)
        if side == OrderSide.BUY:
            # Prefer ask when available; worsen by configured slip.
            base = quote.ask if quote.ask > 0 else mid + half_spread
            return base + slip
        base = quote.bid if quote.bid > 0 else mid - half_spread
        return max(1e-8, base - slip)

    def _commission(self, notional: float) -> float:
        return abs(notional) * (self.config.commission_bps / 10_000.0)

    def _apply_buy(self, symbol: str, qty: float, price: float) -> None:
        state = self._require_connected()
        pos = state.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        old_qty = float(pos["qty"])
        old_avg = float(pos["avg_price"])
        new_qty = old_qty + qty
        if new_qty > 0:
            new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty
        else:
            new_avg = 0.0
        state.positions[symbol] = {"qty": new_qty, "avg_price": new_avg}

    def _apply_sell(self, symbol: str, qty: float) -> None:
        state = self._require_connected()
        pos = state.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        new_qty = float(pos["qty"]) - qty
        if abs(new_qty) < 1e-12:
            state.positions.pop(symbol, None)
        else:
            state.positions[symbol] = {
                "qty": new_qty,
                "avg_price": float(pos["avg_price"]),
            }

    def _mark_price(self, symbol: str, fallback: float) -> float:
        q = self._quotes.get(symbol)
        if q is None:
            return fallback
        return float(q.last)

    def _equity(self, state: LedgerState) -> float:
        total = state.cash
        for sym, pos in state.positions.items():
            qty = float(pos["qty"])
            px = self._mark_price(sym, float(pos["avg_price"]))
            total += qty * px
        return total

    @staticmethod
    def _symbols_from_risk(risk_result: RiskResult, n: int) -> list[str]:
        details = dict(risk_result.details or {})
        details.update(risk_result.record.details or {})
        symbols = details.get("symbols")
        if symbols is None:
            raise ValueError(
                "submit_target requires details['symbols'] matching weight order"
            )
        out = [str(s).upper() for s in symbols]
        if len(out) != n:
            raise ValueError(f"symbols length {len(out)} != weights length {n}")
        return out
