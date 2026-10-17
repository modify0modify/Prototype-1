"""Persistent JSON ledger for the paper broker.

Paths (under ``data_dir``, default ``data/interim/paper_broker``):

* ``account.json`` — cash + positions
* ``orders.json`` — full order ledger
* ``idempotency.json`` — idempotency_key → order_id
* ``fills.jsonl`` — append-only fill events
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rl_trader.brokers.types import Order


@dataclass
class LedgerState:
    """In-memory snapshot of the paper ledger."""

    cash: float
    positions: dict[str, dict[str, float]] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)
    idempotency: dict[str, str] = field(default_factory=dict)
    currency: str = "USD"
    account_id: str = "paper-sim"
    updated_at: str | None = None


class PaperLedger:
    """Load/save paper-broker state under a local directory."""

    ACCOUNT_FILE = "account.json"
    ORDERS_FILE = "orders.json"
    IDEMPOTENCY_FILE = "idempotency.json"
    FILLS_FILE = "fills.jsonl"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def ensure_dir(self) -> None:
        """Create the ledger directory if missing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, *, default_cash: float, account_id: str) -> LedgerState:
        """Load ledger files or return a fresh state."""
        self.ensure_dir()
        account_path = self.data_dir / self.ACCOUNT_FILE
        if account_path.is_file():
            raw = json.loads(account_path.read_text(encoding="utf-8"))
            cash = float(raw.get("cash", default_cash))
            positions = {
                str(sym).upper(): {
                    "qty": float(pos.get("qty", 0.0)),
                    "avg_price": float(pos.get("avg_price", 0.0)),
                }
                for sym, pos in dict(raw.get("positions") or {}).items()
            }
            currency = str(raw.get("currency", "USD"))
            acct_id = str(raw.get("account_id", account_id))
            updated_at = raw.get("updated_at")
        else:
            cash = float(default_cash)
            positions = {}
            currency = "USD"
            acct_id = account_id
            updated_at = None

        orders: dict[str, Order] = {}
        orders_path = self.data_dir / self.ORDERS_FILE
        if orders_path.is_file():
            payload = json.loads(orders_path.read_text(encoding="utf-8"))
            for item in payload.get("orders", []):
                order = Order.from_dict(item)
                orders[order.order_id] = order

        idem: dict[str, str] = {}
        idem_path = self.data_dir / self.IDEMPOTENCY_FILE
        if idem_path.is_file():
            raw_idem = json.loads(idem_path.read_text(encoding="utf-8"))
            idem = {str(k): str(v) for k, v in dict(raw_idem).items()}

        return LedgerState(
            cash=cash,
            positions=positions,
            orders=orders,
            idempotency=idem,
            currency=currency,
            account_id=acct_id,
            updated_at=updated_at,
        )

    def save(self, state: LedgerState, *, now: datetime) -> None:
        """Persist account, orders, and idempotency map atomically-ish."""
        self.ensure_dir()
        stamp = now.isoformat()
        account_doc: dict[str, Any] = {
            "cash": state.cash,
            "positions": state.positions,
            "currency": state.currency,
            "account_id": state.account_id,
            "updated_at": stamp,
        }
        self._write_json(self.data_dir / self.ACCOUNT_FILE, account_doc)
        orders_doc = {
            "orders": [o.to_dict() for o in state.orders.values()],
            "updated_at": stamp,
        }
        self._write_json(self.data_dir / self.ORDERS_FILE, orders_doc)
        self._write_json(self.data_dir / self.IDEMPOTENCY_FILE, state.idempotency)
        state.updated_at = stamp

    def append_fill(self, event: dict[str, Any]) -> None:
        """Append one fill event to the JSONL fill log."""
        self.ensure_dir()
        path = self.data_dir / self.FILLS_FILE
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
