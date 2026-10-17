"""Artifact / SQLite loaders for the monitoring dashboard.

All loaders return empty structures when sources are missing (graceful empty
states). They never open live broker connections or read secret env values
for display.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from rl_trader.monitoring.dashboard.safety import redact_mapping

DEFAULT_PROJECT_ROOT = Path.cwd()


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, (dict, list)) else None


def _read_mapping(path: Path) -> dict[str, Any] | None:
    """Load a JSON or YAML object mapping (for configs / reports)."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except Exception:
            return None
        return data if isinstance(data, dict) else None
    raw = _read_json(path)
    return raw if isinstance(raw, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(doc, dict):
                rows.append(doc)
    except OSError:
        return []
    return rows


def _parse_ts(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        ts = value
    else:
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


@dataclass(slots=True)
class DashboardPaths:
    """Resolved artifact locations."""

    root: Path = field(default_factory=lambda: DEFAULT_PROJECT_ROOT)
    paper_state_dir: Path = Path("reports/paper_trading")
    paper_broker_dir: Path = Path("data/interim/paper_broker")
    walk_forward_dir: Path = Path("reports/walk_forward")
    agent_comparison_dir: Path = Path("reports/agent_comparison")
    benchmarks_dir: Path = Path("reports/benchmarks")
    data_quality_dir: Path = Path("reports/data_quality")
    feature_quality_dir: Path = Path("reports/feature_quality")
    backtests_dir: Path = Path("reports/backtests")
    database_url: str = "sqlite:///./reports/rl_trader.db"
    config_path: Path = Path("configs/paper.yaml")

    def resolve(self) -> DashboardPaths:
        """Make relative paths absolute under ``root``."""
        root = self.root.resolve()

        def _abs(p: Path) -> Path:
            return p if p.is_absolute() else root / p

        return DashboardPaths(
            root=root,
            paper_state_dir=_abs(self.paper_state_dir),
            paper_broker_dir=_abs(self.paper_broker_dir),
            walk_forward_dir=_abs(self.walk_forward_dir),
            agent_comparison_dir=_abs(self.agent_comparison_dir),
            benchmarks_dir=_abs(self.benchmarks_dir),
            data_quality_dir=_abs(self.data_quality_dir),
            feature_quality_dir=_abs(self.feature_quality_dir),
            backtests_dir=_abs(self.backtests_dir),
            database_url=self.database_url,
            config_path=_abs(self.config_path),
        )


@dataclass(slots=True)
class HealthStatus:
    """System-health snapshot for warnings."""

    mode: str
    heartbeat_status: str | None
    heartbeat_ts: datetime | None
    heartbeat_age_seconds: float | None
    heartbeat_stale: bool
    heartbeat_missing: bool
    latest_data_ts: datetime | None
    data_stale: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OverviewSnapshot:
    """Top-level KPIs for the Overview page."""

    mode: str
    mode_label: str
    simulated: bool
    model_id: str | None
    model_version: str | None
    account_equity: float | None
    cash: float | None
    exposure: float | None
    daily_pnl: float | None
    cumulative_pnl: float | None
    drawdown: float | None
    peak_equity: float | None
    positions_count: int
    open_orders: int
    kill_switch_active: bool | None
    latest_data_ts: datetime | None
    heartbeat_ts: datetime | None
    source_notes: list[str] = field(default_factory=list)


def load_operating_mode(paths: DashboardPaths) -> str:
    """Infer mode from config YAML (defaults to paper)."""
    data = _read_mapping(paths.config_path)
    if isinstance(data, dict):
        mode = data.get("app_mode") or data.get("mode")
        if isinstance(mode, str) and mode:
            return mode.lower()
    return "paper"


def load_heartbeat(paths: DashboardPaths) -> dict[str, Any]:
    raw = _read_json(paths.paper_state_dir / "heartbeat.json")
    return redact_mapping(raw) if isinstance(raw, dict) else {}


def load_paper_metrics(paths: DashboardPaths) -> dict[str, Any]:
    raw = _read_json(paths.paper_state_dir / "metrics.json")
    return redact_mapping(raw) if isinstance(raw, dict) else {}


def load_paper_decisions(paths: DashboardPaths) -> pd.DataFrame:
    rows = _read_jsonl(paths.paper_state_dir / "decisions.jsonl")
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_paper_account(paths: DashboardPaths) -> dict[str, Any]:
    raw = _read_json(paths.paper_broker_dir / "account.json")
    return redact_mapping(raw) if isinstance(raw, dict) else {}


def load_paper_orders(paths: DashboardPaths) -> pd.DataFrame:
    raw = _read_json(paths.paper_broker_dir / "orders.json")
    if not isinstance(raw, dict):
        return pd.DataFrame()
    orders = raw.get("orders") or []
    if not isinstance(orders, list):
        return pd.DataFrame()
    return pd.DataFrame(orders)


def load_paper_fills(paths: DashboardPaths) -> pd.DataFrame:
    rows = _read_jsonl(paths.paper_broker_dir / "fills.jsonl")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_positions_table(paths: DashboardPaths) -> pd.DataFrame:
    account = load_paper_account(paths)
    positions = account.get("positions") or {}
    if not isinstance(positions, dict) or not positions:
        return pd.DataFrame(columns=["symbol", "qty", "avg_price"])
    rows = [
        {
            "symbol": str(sym),
            "qty": float(pos.get("qty", 0.0)),
            "avg_price": float(pos.get("avg_price", 0.0)),
        }
        for sym, pos in positions.items()
        if isinstance(pos, dict) and abs(float(pos.get("qty", 0.0))) > 1e-12
    ]
    return pd.DataFrame(rows)


def load_agent_comparison(paths: DashboardPaths) -> dict[str, Any]:
    raw = _read_json(paths.agent_comparison_dir / "latest.json")
    return redact_mapping(raw) if isinstance(raw, dict) else {}


def load_benchmarks(paths: DashboardPaths) -> dict[str, Any]:
    raw = _read_json(paths.benchmarks_dir / "latest.json")
    return redact_mapping(raw) if isinstance(raw, dict) else {}


def _latest_walk_forward_dir(paths: DashboardPaths) -> Path | None:
    root = paths.walk_forward_dir
    if not root.is_dir():
        return None
    pointer = root / "latest_pointer.txt"
    # Some runs put pointer inside a wf_* dir; also scan children.
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("wf_")],
        key=lambda p: p.name,
        reverse=True,
    )
    if pointer.is_file():
        try:
            target = Path(pointer.read_text(encoding="utf-8").strip())
            if not target.is_absolute():
                target = root / target
            if target.is_dir():
                return target
        except OSError:
            pass
    for cand in candidates:
        inner = cand / "latest_pointer.txt"
        if inner.is_file():
            return cand
    return candidates[0] if candidates else None


def load_walk_forward(paths: DashboardPaths) -> dict[str, Any]:
    wf_dir = _latest_walk_forward_dir(paths)
    if wf_dir is None:
        return {"available": False, "path": None}
    aggregate = _read_json(wf_dir / "aggregate_report.json")
    stability = _read_json(wf_dir / "model_stability_report.json")
    fold_csv = wf_dir / "fold_summary.csv"
    folds = pd.read_csv(fold_csv) if fold_csv.is_file() else pd.DataFrame()
    return {
        "available": True,
        "path": str(wf_dir),
        "aggregate": redact_mapping(aggregate) if isinstance(aggregate, dict) else {},
        "stability": redact_mapping(stability) if isinstance(stability, dict) else {},
        "folds": folds,
    }


def load_quality_reports(paths: DashboardPaths) -> dict[str, Any]:
    """Load newest JSON reports from data/feature quality dirs."""

    def _newest(directory: Path) -> list[dict[str, Any]]:
        if not directory.is_dir():
            return []
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        out: list[dict[str, Any]] = []
        for path in files[:20]:
            raw = _read_json(path)
            if isinstance(raw, dict):
                doc = dict(redact_mapping(raw))
                doc["_source_file"] = path.name
                out.append(doc)
        return out

    return {
        "data_quality": _newest(paths.data_quality_dir),
        "feature_quality": _newest(paths.feature_quality_dir),
    }


def load_risk_limits(paths: DashboardPaths) -> dict[str, Any]:
    data = _read_mapping(paths.config_path)
    if not isinstance(data, dict):
        return {}
    risk = data.get("risk")
    if not isinstance(risk, dict):
        return {}
    return dict(redact_mapping(risk))


def try_load_db_tables(paths: DashboardPaths) -> dict[str, pd.DataFrame]:
    """Best-effort SQLite table reads; empty frames when DB missing."""
    url = paths.database_url
    out: dict[str, pd.DataFrame] = {
        "orders": pd.DataFrame(),
        "fills": pd.DataFrame(),
        "agent_decisions": pd.DataFrame(),
        "risk_decisions": pd.DataFrame(),
        "alerts": pd.DataFrame(),
        "kill_switch_events": pd.DataFrame(),
        "performance_metrics": pd.DataFrame(),
    }
    if url.startswith("sqlite:///"):
        db_path = Path(url.removeprefix("sqlite:///"))
        if not db_path.is_absolute():
            db_path = paths.root / db_path
        if not db_path.is_file():
            return out
    try:
        from sqlalchemy import select

        from rl_trader.storage.db import create_db_engine, session_scope
        from rl_trader.storage.models import (
            AgentDecisionRow,
            AlertRow,
            FillRow,
            KillSwitchEventRow,
            OrderRow,
            PerformanceMetricsRow,
            RiskDecisionRow,
            row_to_dict,
        )

        engine = create_db_engine(url, create_schema=False)
        mapping = {
            "orders": OrderRow,
            "fills": FillRow,
            "agent_decisions": AgentDecisionRow,
            "risk_decisions": RiskDecisionRow,
            "alerts": AlertRow,
            "kill_switch_events": KillSwitchEventRow,
            "performance_metrics": PerformanceMetricsRow,
        }
        with session_scope(engine) as session:
            for key, cls in mapping.items():
                try:
                    rows = [row_to_dict(r) for r in session.scalars(select(cls)).all()]
                    out[key] = pd.DataFrame(rows)
                except Exception:
                    out[key] = pd.DataFrame()
    except Exception:
        return out
    return out


def compute_health(
    paths: DashboardPaths,
    *,
    now: datetime | None = None,
    heartbeat_stale_seconds: float = 300.0,
    data_stale_seconds: float = 300.0,
) -> HealthStatus:
    """Heartbeat + data freshness warnings."""
    clock = now or datetime.now(UTC)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)
    mode = load_operating_mode(paths)
    hb = load_heartbeat(paths)
    hb_ts = _parse_ts(hb.get("timestamp")) if hb else None
    hb_missing = not hb or hb_ts is None
    hb_age = (clock - hb_ts).total_seconds() if hb_ts is not None else None
    hb_stale = bool(hb_age is not None and hb_age > heartbeat_stale_seconds)

    decisions = load_paper_decisions(paths)
    latest_data: datetime | None = None
    if not decisions.empty and "as_of" in decisions.columns:
        parsed = [_parse_ts(v) for v in decisions["as_of"].tolist()]
        parsed_f = [p for p in parsed if p is not None]
        if parsed_f:
            latest_data = max(parsed_f)
    account = load_paper_account(paths)
    acct_ts = _parse_ts(account.get("updated_at"))
    if acct_ts is not None and (latest_data is None or acct_ts > latest_data):
        latest_data = acct_ts

    data_age = (clock - latest_data).total_seconds() if latest_data is not None else None
    data_stale = latest_data is None or (
        data_age is not None and data_age > data_stale_seconds
    )

    warnings: list[str] = []
    if hb_missing:
        warnings.append("Process heartbeat missing — paper trader may be down.")
    elif hb_stale:
        warnings.append(
            "Process heartbeat stale "
            f"({hb_age:.0f}s old; threshold {heartbeat_stale_seconds:.0f}s)."
        )
    if latest_data is None:
        warnings.append("No market/decision timestamps found — data may be incomplete.")
    elif data_stale:
        warnings.append(
            "Latest data timestamp stale "
            f"({data_age:.0f}s old; threshold {data_stale_seconds:.0f}s)."
        )

    return HealthStatus(
        mode=mode,
        heartbeat_status=str(hb.get("status")) if hb else None,
        heartbeat_ts=hb_ts,
        heartbeat_age_seconds=hb_age,
        heartbeat_stale=hb_stale,
        heartbeat_missing=hb_missing,
        latest_data_ts=latest_data,
        data_stale=data_stale,
        warnings=warnings,
    )


def build_overview(paths: DashboardPaths) -> OverviewSnapshot:
    """Assemble overview KPIs from paper ledger + metrics + decisions."""
    from rl_trader.monitoring.dashboard.safety import is_simulated_mode, mode_label

    mode = load_operating_mode(paths)
    metrics = load_paper_metrics(paths)
    account = load_paper_account(paths)
    positions = load_positions_table(paths)
    orders = load_paper_orders(paths)
    decisions = load_paper_decisions(paths)
    health = compute_health(paths)

    cash = account.get("cash")
    if cash is None:
        cash = metrics.get("last_cash")
    equity = metrics.get("last_equity")
    if equity is None and cash is not None and not positions.empty:
        equity = float(cash) + float((positions["qty"] * positions["avg_price"]).sum())
    peak = metrics.get("peak_equity")
    drawdown = None
    if peak and equity is not None and float(peak) > 0:
        drawdown = max(0.0, 1.0 - float(equity) / float(peak))

    exposure = None
    if equity and float(equity) > 0 and not positions.empty:
        mv = float((positions["qty"] * positions["avg_price"]).sum())
        exposure = mv / float(equity)

    initial = 100_000.0
    cumulative_pnl = None if equity is None else float(equity) - initial
    daily_pnl = None
    if not decisions.empty and equity is not None:
        # Approximate: no marked daily series — leave None unless metrics provide it.
        daily_pnl = metrics.get("daily_pnl")

    model_id = metrics.get("last_model_id")
    model_version = None
    if not decisions.empty and "model_version" in decisions.columns:
        model_version = str(decisions.iloc[-1]["model_version"])
        if model_id is None and "model_id" in decisions.columns:
            model_id = str(decisions.iloc[-1]["model_id"])

    open_orders = 0
    if not orders.empty and "status" in orders.columns:
        open_orders = int(
            orders["status"].isin(["new", "accepted", "partially_filled"]).sum()
        )

    notes = [
        "PaperBroker ledger + paper_trading metrics (simulated).",
        "Live data panels are read-only placeholders; activation is disabled.",
    ]
    if not account:
        notes.append("Paper broker account.json missing — portfolio figures may be empty.")

    return OverviewSnapshot(
        mode=mode,
        mode_label=mode_label(mode),
        simulated=is_simulated_mode(mode),
        model_id=None if model_id is None else str(model_id),
        model_version=model_version,
        account_equity=None if equity is None else float(equity),
        cash=None if cash is None else float(cash),
        exposure=exposure,
        daily_pnl=None if daily_pnl is None else float(daily_pnl),
        cumulative_pnl=cumulative_pnl,
        drawdown=drawdown,
        peak_equity=None if peak is None else float(peak),
        positions_count=int(len(positions)),
        open_orders=open_orders,
        kill_switch_active=_kill_switch_from_db(paths),
        latest_data_ts=health.latest_data_ts,
        heartbeat_ts=health.heartbeat_ts,
        source_notes=notes,
    )


def _kill_switch_from_db(paths: DashboardPaths) -> bool | None:
    tables = try_load_db_tables(paths)
    ks = tables.get("kill_switch_events", pd.DataFrame())
    if ks.empty or "active" not in ks.columns:
        return None
    # Latest event by created_at when present.
    if "created_at" in ks.columns:
        ks = ks.sort_values("created_at")
    return bool(ks.iloc[-1]["active"])


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe for CSV download (empty OK)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy()


def paper_vs_backtest_summary(paths: DashboardPaths) -> pd.DataFrame:
    """Side-by-side paper metrics vs latest walk-forward aggregate (when present)."""
    metrics = load_paper_metrics(paths)
    wf = load_walk_forward(paths)
    rows: list[dict[str, Any]] = [
        {
            "source": "paper (simulated)",
            "mode": "paper",
            "equity": metrics.get("last_equity"),
            "cash": metrics.get("last_cash"),
            "cycles": metrics.get("cycles"),
            "orders_submitted": metrics.get("orders_submitted"),
            "model_id": metrics.get("last_model_id"),
        }
    ]
    agg = wf.get("aggregate") if wf.get("available") else None
    if isinstance(agg, dict) and agg:
        rows.append(
            {
                "source": "walk-forward / backtest (simulated)",
                "mode": "backtest",
                "equity": agg.get("final_nav") or agg.get("mean_final_nav"),
                "cash": None,
                "cycles": agg.get("n_folds") or agg.get("folds"),
                "orders_submitted": None,
                "model_id": agg.get("model_id") or agg.get("algorithm"),
            }
        )
    else:
        rows.append(
            {
                "source": "walk-forward / backtest (simulated)",
                "mode": "backtest",
                "equity": None,
                "cash": None,
                "cycles": None,
                "orders_submitted": None,
                "model_id": None,
            }
        )
    rows.append(
        {
            "source": "live (placeholder — not connected)",
            "mode": "live",
            "equity": None,
            "cash": None,
            "cycles": None,
            "orders_submitted": None,
            "model_id": None,
        }
    )
    return pd.DataFrame(rows)
