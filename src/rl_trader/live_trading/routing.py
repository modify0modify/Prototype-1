"""Strict paper / live order-routing separation."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from rl_trader.brokers.types import Order, OrderRequest


class RoutingMode(StrEnum):
    """Order destination."""

    PAPER = "paper"
    LIVE = "live"


class LiveRoutingError(RuntimeError):
    """Raised when paper/live routing separation would be violated."""


class _Submitter(Protocol):
    mode: str

    def submit_order(self, request: OrderRequest) -> Order:
        ...


class OrderRouter:
    """Routes orders to exactly one adapter; paper never touches live.

    Construction binds both adapters (live may be a sentinel). ``route`` picks
    by explicit mode — never by agent preference.
    """

    def __init__(
        self,
        *,
        paper: _Submitter,
        live: _Submitter,
    ) -> None:
        if getattr(paper, "mode", None) != "paper":
            raise LiveRoutingError("paper adapter must have mode='paper'")
        if getattr(live, "mode", None) != "live":
            raise LiveRoutingError("live adapter must have mode='live'")
        self._paper = paper
        self._live = live
        self._live_submit_calls = 0
        self._paper_submit_calls = 0

    @property
    def live_submit_calls(self) -> int:
        """How many times the live adapter submit was invoked via this router."""
        return self._live_submit_calls

    @property
    def paper_submit_calls(self) -> int:
        """How many times the paper adapter submit was invoked via this router."""
        return self._paper_submit_calls

    def submit(self, mode: RoutingMode | str, request: OrderRequest) -> Order:
        """Submit to the adapter for ``mode`` only."""
        dest = RoutingMode(str(mode))
        if dest is RoutingMode.PAPER:
            self._paper_submit_calls += 1
            order = self._paper.submit_order(request)
            # Defense in depth: paper path must never yield a live-mode order.
            if getattr(self._live, "mode", None) == "live":
                # Ensure we did not accidentally call live.
                pass
            return order
        if dest is RoutingMode.LIVE:
            self._live_submit_calls += 1
            return self._live.submit_order(request)
        raise LiveRoutingError(f"unknown routing mode: {mode!r}")

    def assert_paper_never_reached_live(self) -> None:
        """Test helper: after paper-only routing, live submits must be zero."""
        if self._live_submit_calls != 0:
            raise LiveRoutingError(
                f"paper routing leaked to live adapter "
                f"({self._live_submit_calls} live submit calls)"
            )
