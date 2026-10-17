"""Unit tests for dashboard loaders, mode labels, and secret redaction."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from rl_trader.monitoring.dashboard.loaders import (
    DashboardPaths,
    build_overview,
    compute_health,
    dataframe_for_csv,
    load_operating_mode,
    load_paper_decisions,
    load_positions_table,
    paper_vs_backtest_summary,
)
from rl_trader.monitoring.dashboard.safety import (
    LIVE_ACTIVATION_BLOCKED_MESSAGE,
    SIMULATED_BANNER,
    assert_no_live_activation_controls,
    contains_secret_plaintext,
    is_simulated_mode,
    mode_label,
    redact_mapping,
)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@pytest.fixture()
def paths(tmp_path: Path) -> DashboardPaths:
    paper = tmp_path / "reports" / "paper_trading"
    broker = tmp_path / "data" / "interim" / "paper_broker"
    cfg = tmp_path / "configs" / "paper.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "app_mode: paper\n"
        "live_trading_enabled: false\n"
        "risk:\n"
        "  max_position_size: 0.25\n"
        "  paper_api_key: SECRET\n",
        encoding="utf-8",
    )
    _write(
        paper / "heartbeat.json",
        {
            "timestamp": datetime(2024, 6, 3, 15, 0, tzinfo=UTC).isoformat(),
            "status": "cycle_complete",
            "details": {},
        },
    )
    _write(
        paper / "metrics.json",
        {
            "cycles": 2,
            "last_equity": 101_000.0,
            "last_cash": 50_000.0,
            "peak_equity": 102_000.0,
            "last_model_id": "demo-model",
            "orders_submitted": 3,
        },
    )
    with (paper / "decisions.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "cycle_id": "p:demo:1",
                    "model_id": "demo-model",
                    "model_version": "1.2.3",
                    "as_of": datetime(2024, 6, 3, 15, 0, tzinfo=UTC).isoformat(),
                    "risk_decision": "approved",
                    "risk_reason": "ok",
                    "target_weights": [0.2, 0.2],
                    "approved_weights": [0.2, 0.2],
                    "dry_run": False,
                    "order_ids": ["o1"],
                    "rejected_orders": [],
                    "violated_rules": [],
                }
            )
            + "\n"
        )
    _write(
        broker / "account.json",
        {
            "cash": 50_000.0,
            "positions": {"AAPL": {"qty": 10.0, "avg_price": 100.0}},
            "updated_at": datetime(2024, 6, 3, 15, 0, tzinfo=UTC).isoformat(),
        },
    )
    _write(
        broker / "orders.json",
        {
            "orders": [
                {
                    "order_id": "o1",
                    "symbol": "AAPL",
                    "side": "buy",
                    "status": "filled",
                    "qty": 10,
                },
                {
                    "order_id": "o2",
                    "symbol": "MSFT",
                    "side": "buy",
                    "status": "rejected",
                    "reject_reason": "insufficient_buying_power",
                    "qty": 1000,
                },
            ]
        },
    )
    return DashboardPaths(
        root=tmp_path,
        paper_state_dir=paper,
        paper_broker_dir=broker,
        config_path=cfg,
        walk_forward_dir=tmp_path / "reports" / "walk_forward",
        agent_comparison_dir=tmp_path / "reports" / "agent_comparison",
        benchmarks_dir=tmp_path / "reports" / "benchmarks",
        data_quality_dir=tmp_path / "reports" / "data_quality",
        feature_quality_dir=tmp_path / "reports" / "feature_quality",
    ).resolve()


def test_mode_labels_distinguish_sources() -> None:
    assert "PAPER" in mode_label("paper")
    assert "BACKTEST" in mode_label("backtest")
    assert "LIVE" in mode_label("live")
    assert "activation disabled" in mode_label("live").lower()
    assert is_simulated_mode("paper")
    assert is_simulated_mode("backtest")
    assert "SIMULATED" in SIMULATED_BANNER
    assert "activation is disabled" in LIVE_ACTIVATION_BLOCKED_MESSAGE.lower()


def test_no_live_activation_controls_guard() -> None:
    assert_no_live_activation_controls({"enable_live_trading": False})
    with pytest.raises(RuntimeError, match="must not expose"):
        assert_no_live_activation_controls({"enable_live_trading": True})


def test_redact_mapping_strips_secrets() -> None:
    raw = {
        "broker": {"paper_api_key": "sk-secret", "environment": "paper"},
        "account_id": "LIVEACCT123456",
        "token": "abc",
        "risk": {"max_position_size": 0.2},
    }
    clean = redact_mapping(raw)
    assert "paper_api_key" not in clean["broker"]
    assert clean["broker"]["environment"] == "paper"
    assert clean["account_id"].startswith("***")
    assert "token" not in clean
    dumped = json.dumps(clean)
    assert not contains_secret_plaintext(dumped)
    assert contains_secret_plaintext("api_key=sk-live-123")


def test_load_operating_mode_and_overview(paths: DashboardPaths) -> None:
    assert load_operating_mode(paths) == "paper"
    overview = build_overview(paths)
    assert overview.mode == "paper"
    assert overview.simulated is True
    assert overview.model_id == "demo-model"
    assert overview.model_version == "1.2.3"
    assert overview.cash == pytest.approx(50_000.0)
    assert overview.account_equity == pytest.approx(101_000.0)
    assert overview.positions_count == 1
    assert overview.drawdown == pytest.approx(1.0 - 101_000.0 / 102_000.0)


def test_positions_and_decisions_loaders(paths: DashboardPaths) -> None:
    pos = load_positions_table(paths)
    assert list(pos["symbol"]) == ["AAPL"]
    decisions = load_paper_decisions(paths)
    assert len(decisions) == 1
    assert decisions.iloc[0]["model_version"] == "1.2.3"


def test_health_stale_and_missing_heartbeat(tmp_path: Path) -> None:
    cfg = tmp_path / "configs" / "paper.yaml"
    _write(cfg, {"app_mode": "paper"})
    paths = DashboardPaths(
        root=tmp_path,
        paper_state_dir=tmp_path / "reports" / "paper_trading",
        paper_broker_dir=tmp_path / "broker",
        config_path=cfg,
    ).resolve()
    now = datetime(2024, 6, 3, 16, 0, tzinfo=UTC)
    missing = compute_health(paths, now=now, heartbeat_stale_seconds=60, data_stale_seconds=60)
    assert missing.heartbeat_missing is True
    assert any("heartbeat missing" in w.lower() for w in missing.warnings)

    _write(
        paths.paper_state_dir / "heartbeat.json",
        {
            "timestamp": (now - timedelta(seconds=600)).isoformat(),
            "status": "ready",
        },
    )
    stale = compute_health(paths, now=now, heartbeat_stale_seconds=60, data_stale_seconds=60)
    assert stale.heartbeat_stale is True
    assert any("stale" in w.lower() for w in stale.warnings)


def test_paper_vs_backtest_includes_live_placeholder(paths: DashboardPaths) -> None:
    df = paper_vs_backtest_summary(paths)
    modes = set(df["mode"].tolist())
    assert "paper" in modes
    assert "backtest" in modes
    assert "live" in modes
    live_row = df.loc[df["mode"] == "live"].iloc[0]
    assert live_row["equity"] is None or pd.isna(live_row["equity"])


def test_csv_helper_empty_safe() -> None:
    assert dataframe_for_csv(pd.DataFrame()).empty


def test_graceful_empty_when_artifacts_missing(tmp_path: Path) -> None:
    paths = DashboardPaths(root=tmp_path).resolve()
    overview = build_overview(paths)
    assert overview.mode == "paper"
    assert overview.positions_count == 0
    assert load_paper_decisions(paths).empty
