"""Config-driven live rollout stages.

Stage 0: No live orders — shadow decisions only (default).
Stage 1: Tiny fixed capital, long-only.
Stage 2: Gradual capital increase **only after completed human/config review**.
Stage 3: Expanded ticker universe only after additional review.

The system never advances stages or raises capital on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_trader.config.models import LiveTradingSettings


class RolloutStage(IntEnum):
    """Live rollout stage (config-driven)."""

    SHADOW = 0
    TINY_LONG_ONLY = 1
    REVIEWED_CAPITAL = 2
    EXPANDED_UNIVERSE = 3


@dataclass(frozen=True, slots=True)
class StagePolicy:
    """Resolved policy for the configured rollout stage."""

    stage: RolloutStage
    allow_live_orders: bool
    long_only: bool
    capital_limit: float | None
    allowed_symbols: tuple[str, ...] | None  # None = use data.universe as-is
    detail: str


def _stage(settings: LiveTradingSettings) -> RolloutStage:
    return RolloutStage(int(settings.rollout_stage))


def effective_capital_limit(settings: LiveTradingSettings) -> float | None:
    """Human-configured capital ceiling for the active stage (never auto-raised)."""
    stage = _stage(settings)
    hard_cap = settings.max_live_capital
    if stage is RolloutStage.SHADOW:
        return 0.0
    if stage is RolloutStage.TINY_LONG_ONLY:
        stage_cap = settings.stage1_capital_limit
    elif stage is RolloutStage.REVIEWED_CAPITAL:
        stage_cap = settings.stage2_capital_limit
    else:
        stage_cap = settings.stage3_capital_limit
    candidates = [c for c in (hard_cap, stage_cap) if c is not None]
    if not candidates:
        return None
    return min(candidates)


def live_orders_permitted(settings: LiveTradingSettings) -> bool:
    """True only when stage > 0 (Stage 0 never submits live orders)."""
    return _stage(settings) is not RolloutStage.SHADOW


def resolve_stage_policy(
    settings: LiveTradingSettings,
    *,
    base_universe: tuple[str, ...] = (),
) -> StagePolicy:
    """Resolve stage policy from config (no side effects, no auto-advance)."""
    stage = _stage(settings)
    capital = effective_capital_limit(settings)

    if stage is RolloutStage.SHADOW:
        return StagePolicy(
            stage=stage,
            allow_live_orders=False,
            long_only=True,
            capital_limit=0.0,
            allowed_symbols=None,
            detail="stage_0_shadow_only — no live orders",
        )

    if stage is RolloutStage.TINY_LONG_ONLY:
        return StagePolicy(
            stage=stage,
            allow_live_orders=True,
            long_only=bool(settings.long_only),
            capital_limit=capital,
            allowed_symbols=tuple(base_universe) if base_universe else None,
            detail="stage_1_tiny_long_only",
        )

    if stage is RolloutStage.REVIEWED_CAPITAL:
        if not settings.stage2_review_acknowledged:
            return StagePolicy(
                stage=stage,
                allow_live_orders=False,
                long_only=bool(settings.long_only),
                capital_limit=capital,
                allowed_symbols=tuple(base_universe) if base_universe else None,
                detail="stage_2_blocked — stage2_review_acknowledged required",
            )
        return StagePolicy(
            stage=stage,
            allow_live_orders=True,
            long_only=bool(settings.long_only),
            capital_limit=capital,
            allowed_symbols=tuple(base_universe) if base_universe else None,
            detail="stage_2_reviewed_capital",
        )

    # Stage 3
    if not settings.stage3_review_acknowledged:
        return StagePolicy(
            stage=stage,
            allow_live_orders=False,
            long_only=bool(settings.long_only),
            capital_limit=capital,
            allowed_symbols=None,
            detail="stage_3_blocked — stage3_review_acknowledged required",
        )
    expanded = tuple(s.upper() for s in settings.stage3_expanded_universe)
    if not expanded:
        return StagePolicy(
            stage=stage,
            allow_live_orders=False,
            long_only=bool(settings.long_only),
            capital_limit=capital,
            allowed_symbols=(),
            detail="stage_3_blocked — stage3_expanded_universe empty",
        )
    return StagePolicy(
        stage=stage,
        allow_live_orders=True,
        long_only=bool(settings.long_only),
        capital_limit=capital,
        allowed_symbols=expanded,
        detail="stage_3_expanded_universe",
    )
