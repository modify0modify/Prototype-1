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


class _CountedLiveSubmitter:
    """Wraps the live adapter so every submit increments the router counter."""

    def __init__(self, inner: _Submitter, router: OrderRouter) -> None:
        self._inner = inner
        self._router = router
        self.mode = "live"

    def submit_order(self, request: OrderRequest) -> Order:
        self._router._note_live_submit()
        return self._inner.submit_order(request)


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
        self._live_submit_calls = 0
        self._paper_submit_calls = 0
        self._paper = paper
        # Count every live.submit_order invocation, including accidental
        # calls from a miswired paper adapter that holds a live reference.
        self._live = _CountedLiveSubmitter(live, self)

    def _note_live_submit(self) -> None:
        self._live_submit_calls += 1

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
            before_live = self._live_submit_calls
            self._paper_submit_calls += 1
            order = self._paper.submit_order(request)
            # Defense in depth: paper path must never invoke the live adapter.
            if self._live_submit_calls != before_live:
                raise LiveRoutingError(
                    "paper routing invoked live adapter — separation violated"
                )
            return order
        if dest is RoutingMode.LIVE:
            return self._live.submit_order(request)
        raise LiveRoutingError(f"unknown routing mode: {mode!r}")

    def assert_paper_never_reached_live(self) -> None:
        """Test helper: after paper-only routing, live submits must be zero."""
        if self._live_submit_calls != 0:
            raise LiveRoutingError(
                f"paper routing leaked to live adapter "
                f"({self._live_submit_calls} live submit calls)"
            )
