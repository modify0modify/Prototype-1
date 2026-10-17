"""Walk-forward window geometry with strict no-leak train/val/test bounds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from rl_trader.config.models import BacktestSettings
from rl_trader.environment.episode import EpisodeData


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):
        return value.date()  # type: ignore[no-any-return]
    return date.fromisoformat(str(value)[:10])


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One non-overlapping train / validation / test fold (bar indices half-open)."""

    fold_id: int
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: date

    def assert_no_leak(self) -> None:
        """Raise if train/val/test index ranges overlap or are out of order."""
        if not (
            0
            <= self.train_start_idx
            < self.train_end_idx
            <= self.val_start_idx
            < self.val_end_idx
            <= self.test_start_idx
            < self.test_end_idx
        ):
            raise ValueError(
                f"fold {self.fold_id}: invalid ordering or overlap "
                f"train=[{self.train_start_idx},{self.train_end_idx}) "
                f"val=[{self.val_start_idx},{self.val_end_idx}) "
                f"test=[{self.test_start_idx},{self.test_end_idx})"
            )
        # Contiguous allowed (end == next start); intersections empty.
        train = set(range(self.train_start_idx, self.train_end_idx))
        val = set(range(self.val_start_idx, self.val_end_idx))
        test = set(range(self.test_start_idx, self.test_end_idx))
        if train & val or train & test or val & test:
            raise ValueError(f"fold {self.fold_id}: index sets overlap (leak)")


@dataclass(frozen=True, slots=True)
class WalkForwardPlan:
    """Ordered walk-forward folds with non-overlapping test windows."""

    folds: tuple[WalkForwardFold, ...]
    window_mode: str
    train_bars: int
    val_bars: int
    test_bars: int
    step_bars: int
    n_bars: int

    def assert_non_overlapping_tests(self) -> None:
        """Ensure successive test windows do not overlap (safe aggregation)."""
        prev_end = -1
        for fold in self.folds:
            fold.assert_no_leak()
            if fold.test_start_idx < prev_end:
                raise ValueError(
                    f"fold {fold.fold_id}: test overlaps previous "
                    f"[{fold.test_start_idx},{fold.test_end_idx}) vs end={prev_end}"
                )
            prev_end = fold.test_end_idx


def build_walk_forward_plan(
    episode: EpisodeData,
    settings: BacktestSettings,
) -> WalkForwardPlan:
    """Build rolling or expanding walk-forward folds from bar geometry.

    Each fold has train → val → test with no index overlap. Successive test
    windows are non-overlapping because ``settings.step_bars >= test_bars``.
    """
    n = episode.n_bars
    train_b = settings.train_bars
    val_b = settings.val_bars
    test_b = settings.test_bars
    step = settings.step_bars
    min_needed = train_b + val_b + test_b
    if n < min_needed:
        raise ValueError(
            f"episode too short for walk-forward: n_bars={n} < {min_needed}"
        )

    folds: list[WalkForwardFold] = []
    fold_id = 0
    origin = 0
    while True:
        if settings.window_mode == "rolling":
            train_start = origin
            train_end = train_start + train_b
        else:  # expanding
            train_start = 0
            train_end = train_b + origin
        val_start = train_end
        val_end = val_start + val_b
        test_start = val_end
        test_end = test_start + test_b
        if test_end > n:
            break
        fold = WalkForwardFold(
            fold_id=fold_id,
            train_start_idx=train_start,
            train_end_idx=train_end,
            val_start_idx=val_start,
            val_end_idx=val_end,
            test_start_idx=test_start,
            test_end_idx=test_end,
            train_start=_as_date(episode.timestamps[train_start]),
            train_end=_as_date(episode.timestamps[train_end - 1]),
            val_start=_as_date(episode.timestamps[val_start]),
            val_end=_as_date(episode.timestamps[val_end - 1]),
            test_start=_as_date(episode.timestamps[test_start]),
            test_end=_as_date(episode.timestamps[test_end - 1]),
        )
        fold.assert_no_leak()
        folds.append(fold)
        fold_id += 1
        if fold_id >= settings.walk_forward_folds:
            break
        origin += step

    if not folds:
        raise ValueError("walk-forward produced zero folds")

    plan = WalkForwardPlan(
        folds=tuple(folds),
        window_mode=settings.window_mode,
        train_bars=train_b,
        val_bars=val_b,
        test_bars=test_b,
        step_bars=step,
        n_bars=n,
    )
    plan.assert_non_overlapping_tests()
    return plan


# Backward-compatible stub API used by older scripts (date-only).
def build_walk_forward_plan_dates(
    start: date,
    end: date,
    *,
    n_folds: int = 3,
) -> WalkForwardPlan:
    """Legacy date stub — prefer :func:`build_walk_forward_plan` with an episode."""
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if end <= start:
        raise ValueError("end must be after start")
    # Synthetic daily bars spanning [start, end].
    n_days = (end - start).days + 1
    from datetime import timedelta

    from rl_trader.environment.synthetic import make_synthetic_episode

    episode = make_synthetic_episode(n_bars=max(n_days, 30), n_assets=2, seed=0)
    stamps = np.array([start + timedelta(days=i) for i in range(episode.n_bars)], dtype=object)
    episode = EpisodeData(
        timestamps=stamps,
        prices=episode.prices,
        market_features=episode.market_features,
        fundamental_features=episode.fundamental_features,
        regime_features=episode.regime_features,
        symbols=episode.symbols,
    )
    settings = BacktestSettings(
        walk_forward_folds=n_folds,
        train_bars=max(4, episode.n_bars // 5),
        val_bars=max(2, episode.n_bars // 10),
        test_bars=max(2, episode.n_bars // 10),
        step_bars=max(2, episode.n_bars // 10),
    )
    return build_walk_forward_plan(episode, settings)


def slice_fold_episodes(
    episode: EpisodeData, fold: WalkForwardFold
) -> tuple[EpisodeData, EpisodeData, EpisodeData]:
    """Materialize train / val / test panels for ``fold``."""
    fold.assert_no_leak()
    train = episode.slice_bars(fold.train_start_idx, fold.train_end_idx)
    val = episode.slice_bars(fold.val_start_idx, fold.val_end_idx)
    test = episode.slice_bars(fold.test_start_idx, fold.test_end_idx)
    return train, val, test
