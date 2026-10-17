"""Database integrity tests for the SQLAlchemy storage layer."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from rl_trader.storage import (
    REQUIRED_TABLES,
    OrderRow,
    StorageRepository,
    assert_no_secret_columns,
    create_db_engine,
    export_all_csv,
    run_all_reconciliations,
    sanitize_config_for_storage,
    session_scope,
    upgrade_head,
)
from rl_trader.storage.models import table_names
from rl_trader.storage.secrets import FORBIDDEN_COLUMN_SUBSTRINGS


@pytest.fixture()
def engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'test.db'}"
    upgrade_head(url)
    return create_db_engine(url, create_schema=False)


def test_required_tables_exist_after_migration(engine) -> None:
    names = set(inspect(engine).get_table_names())
    missing = set(REQUIRED_TABLES) - names
    assert not missing, f"missing tables: {sorted(missing)}"
    assert set(REQUIRED_TABLES).issubset(table_names())


def test_no_secret_columns_in_schema() -> None:
    assert_no_secret_columns()
    # Sanity: forbidden substrings themselves aren't present as columns.
    from rl_trader.storage.secrets import mapped_column_names

    for full in mapped_column_names():
        col = full.split(".", 1)[1].lower()
        for frag in FORBIDDEN_COLUMN_SUBSTRINGS:
            assert frag not in col, full


def test_sanitize_config_strips_secrets() -> None:
    clean = sanitize_config_for_storage(
        {
            "risk": {"max_position_size": 0.2},
            "broker": {
                "environment": "paper",
                "paper_api_key": "SECRET",
                "live_api_secret": "SECRET2",
            },
            "token": "nope",
        }
    )
    assert "paper_api_key" not in clean["broker"]
    assert "live_api_secret" not in clean["broker"]
    assert "token" not in clean
    assert clean["broker"]["environment"] == "paper"


def test_order_idempotency_unique(engine) -> None:
    repo = StorageRepository(engine)
    first = repo.record_order(
        broker_order_id="b1",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        status="filled",
        idempotency_key="idem-1",
    )
    second = repo.record_order(
        broker_order_id="b2",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        status="filled",
        idempotency_key="idem-1",
    )
    assert first == second
    with session_scope(engine) as session:
        n = session.scalar(select(OrderRow).where(OrderRow.idempotency_key == "idem-1"))
        assert n is not None
        count = len(session.scalars(select(OrderRow)).all())
    assert count == 1


def test_order_links_to_model_agent_risk_config(engine) -> None:
    repo = StorageRepository(engine)
    cfg = repo.upsert_configuration_version(
        "risk-v-test",
        {"risk": {"x": 1}, "paper_api_key": "should-strip"},
    )
    with session_scope(engine) as session:
        from rl_trader.storage.models import ConfigurationVersionRow

        row = session.get(ConfigurationVersionRow, cfg)
        assert row is not None
        payload = json.loads(row.config_json)
        assert "paper_api_key" not in payload

    fm = repo.record_feature_manifest(
        name="unit",
        version="1",
        feature_specs=[{"name": "returns_1d", "lookback": 1}],
    )
    run = repo.record_model_run(run_key="run-1", algorithm="stub", feature_manifest_id=fm)
    ckpt = repo.record_checkpoint(
        model_id="m1",
        algorithm="stub",
        checkpoint_path="/tmp/ckpt",
        model_run_id=run,
        feature_manifest_id=fm,
    )
    agent = repo.record_agent_decision(
        model_id="m1",
        model_version="1",
        cycle_id="c1",
        as_of=datetime(2024, 6, 3, 15, 0, tzinfo=UTC),
        target_weights=[0.2, 0.2],
        model_checkpoint_id=ckpt,
    )
    risk = repo.record_risk_decision(
        model_id="m1",
        decision="approved",
        reason="ok",
        violated_rules=[],
        approved_weights=[0.2, 0.2],
        agent_decision_id=agent,
        configuration_version_id=cfg,
    )
    order_id = repo.record_order(
        broker_order_id="ord-1",
        symbol="AAPL",
        side="buy",
        qty=2.0,
        status="filled",
        filled_qty=2.0,
        avg_fill_price=100.0,
        idempotency_key="link-1",
        model_checkpoint_id=ckpt,
        agent_decision_id=agent,
        risk_decision_id=risk,
        configuration_version_id=cfg,
    )
    repo.record_fill(
        order_id=order_id,
        qty=2.0,
        price=100.0,
        filled_at=datetime(2024, 6, 3, 15, 1, tzinfo=UTC),
        commission=0.2,
        spread_cost=0.1,
        slippage_cost=0.05,
    )
    with session_scope(engine) as session:
        order = session.get(OrderRow, order_id)
        assert order is not None
        assert order.model_checkpoint_id == ckpt
        assert order.agent_decision_id == agent
        assert order.risk_decision_id == risk
        assert order.configuration_version_id == cfg
        assert order.fills[0].commission == pytest.approx(0.2)
        assert order.fills[0].spread_cost == pytest.approx(0.1)
        assert order.fills[0].slippage_cost == pytest.approx(0.05)


def test_rejected_orders_stored_with_reason(engine) -> None:
    repo = StorageRepository(engine)
    oid = repo.record_order(
        broker_order_id="rej-1",
        symbol="MSFT",
        side="buy",
        qty=1.0,
        status="rejected",
        reject_reason="insufficient_buying_power",
        idempotency_key="rej-key",
    )
    with session_scope(engine) as session:
        order = session.get(OrderRow, oid)
        assert order is not None
        assert order.status == "rejected"
        assert order.reject_reason == "insufficient_buying_power"


def test_uuid_pks_immutable_style(engine) -> None:
    repo = StorageRepository(engine)
    oid = repo.record_order(
        broker_order_id="u1",
        symbol="QQQ",
        side="sell",
        qty=1.0,
        status="cancelled",
        reject_reason=None,
    )
    assert isinstance(oid, uuid.UUID)


def test_csv_export(engine, tmp_path: Path) -> None:
    repo = StorageRepository(engine)
    repo.record_alert(level="info", message="hello", context={"k": 1})
    out = tmp_path / "csv"
    counts = export_all_csv(engine, out)
    assert counts["alerts"] >= 1
    assert (out / "alerts.csv").is_file()
    assert (out / "orders.csv").is_file()


def test_reconciliation_queries(engine) -> None:
    repo = StorageRepository(engine)
    cfg = repo.upsert_configuration_version("cfg", {"a": 1})
    agent = repo.record_agent_decision(
        model_id="m",
        model_version="1",
        cycle_id="c",
        as_of=datetime(2024, 1, 2, tzinfo=UTC),
        target_weights=[0.1],
    )
    risk = repo.record_risk_decision(
        model_id="m",
        decision="approved",
        reason="ok",
        violated_rules=[],
        agent_decision_id=agent,
        configuration_version_id=cfg,
        approved_weights=[0.1],
    )
    oid = repo.record_order(
        broker_order_id="f1",
        symbol="SPY",
        side="buy",
        qty=1.0,
        status="filled",
        filled_qty=1.0,
        avg_fill_price=10.0,
        idempotency_key="rec-1",
        agent_decision_id=agent,
        risk_decision_id=risk,
        configuration_version_id=cfg,
    )
    repo.record_fill(
        order_id=oid,
        qty=1.0,
        price=10.0,
        filled_at=datetime(2024, 1, 2, 1, tzinfo=UTC),
        commission=0.01,
        spread_cost=0.02,
        slippage_cost=0.03,
    )
    repo.record_portfolio_snapshot(
        as_of=datetime(2024, 1, 2, 2, tzinfo=UTC),
        cash=90.0,
        equity=100.0,
        buying_power=90.0,
        positions=[
            {
                "symbol": "SPY",
                "qty": 1.0,
                "avg_price": 10.0,
                "market_price": 10.0,
                "market_value": 10.0,
            }
        ],
    )
    findings = run_all_reconciliations(engine)
    by_name = {f.check: f for f in findings}
    assert by_name["duplicate_idempotency_keys"].ok
    assert by_name["filled_orders_without_fills"].ok
    assert by_name["fill_qty_vs_order_filled_qty"].ok
    assert by_name["orders_without_risk_link"].ok
    assert by_name["position_snapshot_cash_vs_equity"].ok
    assert by_name["rejected_orders_have_reason"].ok


def test_unique_idempotency_enforced_at_db(engine) -> None:
    with session_scope(engine) as session:
        session.add(
            OrderRow(
                broker_order_id="x1",
                idempotency_key="dup",
                symbol="A",
                side="buy",
                order_type="market",
                qty=1.0,
                status="new",
            )
        )
    with pytest.raises(IntegrityError):
        with session_scope(engine) as session:
            session.add(
                OrderRow(
                    broker_order_id="x2",
                    idempotency_key="dup",
                    symbol="A",
                    side="buy",
                    order_type="market",
                    qty=1.0,
                    status="new",
                )
            )


def test_timestamps_are_utc(engine) -> None:
    repo = StorageRepository(engine)
    oid = repo.record_order(
        broker_order_id="ts1",
        symbol="AAPL",
        side="buy",
        qty=1.0,
        status="new",
        created_at=datetime(2024, 6, 3, 12, 0, tzinfo=UTC),
    )
    with session_scope(engine) as session:
        order = session.get(OrderRow, oid)
        assert order is not None
        assert order.created_at.tzinfo is not None
        assert order.created_at.utcoffset() is not None


def test_foreign_keys_enabled(engine) -> None:
    with session_scope(engine) as session:
        val = session.execute(text("PRAGMA foreign_keys")).scalar()
    assert int(val or 0) == 1
