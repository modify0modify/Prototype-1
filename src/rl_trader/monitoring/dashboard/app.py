"""Streamlit monitoring dashboard — read-only, no live activation.

Launch via ``scripts/run_dashboard.py`` or ``make dashboard``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from rl_trader.monitoring.dashboard.loaders import (
    DashboardPaths,
    build_overview,
    compute_health,
    dataframe_for_csv,
    load_agent_comparison,
    load_benchmarks,
    load_operating_mode,
    load_paper_decisions,
    load_paper_fills,
    load_paper_orders,
    load_positions_table,
    load_quality_reports,
    load_risk_limits,
    load_walk_forward,
    paper_vs_backtest_summary,
    try_load_db_tables,
)
from rl_trader.monitoring.dashboard.safety import (
    LIVE_ACTIVATION_BLOCKED_MESSAGE,
    SIMULATED_BANNER,
    mode_label,
)

PAGES = [
    "Overview",
    "Portfolio",
    "Performance",
    "Trades",
    "Agent decisions",
    "Risk decisions",
    "Model comparison",
    "Walk-forward results",
    "Data quality",
    "System health",
]


def _paths_from_session() -> DashboardPaths:
    root = Path(st.session_state.get("project_root") or Path.cwd())
    db_url = st.session_state.get("database_url") or "sqlite:///./reports/rl_trader.db"
    return DashboardPaths(root=root, database_url=str(db_url)).resolve()


def _banner(mode: str) -> None:
    st.warning(SIMULATED_BANNER)
    st.info(f"Operating mode: **{mode_label(mode)}**")
    if mode == "live":
        st.error(LIVE_ACTIVATION_BLOCKED_MESSAGE)


def _csv_button(df: pd.DataFrame, filename: str, key: str) -> None:
    data = dataframe_for_csv(df)
    if data.empty:
        st.caption("No rows to export.")
        return
    st.download_button(
        label=f"Download CSV ({filename})",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _empty(msg: str) -> None:
    st.info(msg)


def page_overview(paths: DashboardPaths) -> None:
    overview = build_overview(paths)
    health = compute_health(paths)
    _banner(overview.mode)
    for w in health.warnings:
        st.warning(w)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", overview.model_id or "—")
    c2.metric("Model version", overview.model_version or "—")
    c3.metric(
        "Account value (sim)",
        f"{overview.account_equity:,.2f}" if overview.account_equity is not None else "—",
    )
    c4.metric("Cash (sim)", f"{overview.cash:,.2f}" if overview.cash is not None else "—")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        "Exposure",
        f"{overview.exposure:.1%}" if overview.exposure is not None else "—",
    )
    c6.metric(
        "Cumulative P&L (sim)",
        f"{overview.cumulative_pnl:,.2f}" if overview.cumulative_pnl is not None else "—",
    )
    c7.metric(
        "Drawdown",
        f"{overview.drawdown:.1%}" if overview.drawdown is not None else "—",
    )
    c8.metric("Positions", overview.positions_count)

    st.subheader("Paper vs backtest vs live")
    cmp_df = paper_vs_backtest_summary(paths)
    st.dataframe(cmp_df, use_container_width=True)
    _csv_button(cmp_df, "paper_vs_backtest.csv", "csv_overview_cmp")
    st.caption("Live row is a read-only placeholder — this UI cannot activate live trading.")
    for note in overview.source_notes:
        st.caption(note)


def page_portfolio(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    overview = build_overview(paths)
    st.subheader("Current positions (PaperBroker ledger — simulated)")
    positions = load_positions_table(paths)
    if positions.empty:
        _empty("No open positions in the paper ledger.")
    else:
        st.dataframe(positions, use_container_width=True)
        _csv_button(positions, "positions.csv", "csv_positions")
    st.metric("Cash", f"{overview.cash:,.2f}" if overview.cash is not None else "—")
    st.metric(
        "Equity",
        f"{overview.account_equity:,.2f}" if overview.account_equity is not None else "—",
    )
    st.metric(
        "Exposure",
        f"{overview.exposure:.1%}" if overview.exposure is not None else "—",
    )


def page_performance(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    overview = build_overview(paths)
    metrics_cols = st.columns(4)
    metrics_cols[0].metric(
        "Daily P&L (sim)",
        f"{overview.daily_pnl:,.2f}" if overview.daily_pnl is not None else "n/a",
    )
    metrics_cols[1].metric(
        "Cumulative P&L (sim)",
        f"{overview.cumulative_pnl:,.2f}" if overview.cumulative_pnl is not None else "—",
    )
    metrics_cols[2].metric(
        "Drawdown",
        f"{overview.drawdown:.1%}" if overview.drawdown is not None else "—",
    )
    metrics_cols[3].metric(
        "Peak equity",
        f"{overview.peak_equity:,.2f}" if overview.peak_equity is not None else "—",
    )

    fills = load_paper_fills(paths)
    st.subheader("Transaction costs (simulated fills)")
    if fills.empty:
        _empty("No fill cost rows yet.")
        turnover = None
    else:
        cost_names = ("commission", "spread_cost", "slippage_cost")
        cost_cols = [c for c in cost_names if c in fills.columns]
        if "commission" in fills.columns and "spread_cost" not in fills.columns:
            # Paper fills.jsonl stores commission only.
            st.dataframe(fills, use_container_width=True)
            if "commission" in fills.columns:
                st.metric("Total commission (sim)", f"{float(fills['commission'].sum()):,.4f}")
        else:
            st.dataframe(fills, use_container_width=True)
            if cost_cols:
                totals = {c: float(fills[c].sum()) for c in cost_cols}
                st.write(totals)
        _csv_button(fills, "fills_costs.csv", "csv_fills")
        if "qty" in fills.columns and "price" in fills.columns:
            notional = (fills["qty"].astype(float).abs() * fills["price"].astype(float)).sum()
            turnover = float(notional)
        else:
            turnover = None
    st.metric("Turnover notional (sim)", f"{turnover:,.2f}" if turnover is not None else "—")

    st.subheader("Benchmark comparison (research artifact)")
    bench = load_benchmarks(paths)
    if not bench:
        _empty("No benchmarks/latest.json found.")
    else:
        st.json(bench)
        if isinstance(bench.get("rows"), list):
            bdf = pd.DataFrame(bench["rows"])
            st.dataframe(bdf, use_container_width=True)
            _csv_button(bdf, "benchmarks.csv", "csv_bench")


def page_trades(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    orders = load_paper_orders(paths)
    db = try_load_db_tables(paths)
    st.subheader("PaperBroker orders (simulated)")
    if orders.empty:
        _empty("No orders in paper ledger.")
    else:
        st.dataframe(orders, use_container_width=True)
        _csv_button(orders, "orders_paper.csv", "csv_orders_paper")
        rejected = (
            orders.loc[orders["status"] == "rejected"]
            if "status" in orders.columns
            else pd.DataFrame()
        )
        st.subheader("Rejected orders")
        if rejected.empty:
            st.caption("No rejected orders in the ledger.")
        else:
            st.dataframe(rejected, use_container_width=True)
            _csv_button(rejected, "orders_rejected.csv", "csv_orders_rej")

    st.subheader("SQLite orders (if dual-write enabled)")
    if db["orders"].empty:
        _empty("No rows in storage.orders (enable database.persist_paper_trading to dual-write).")
    else:
        st.dataframe(db["orders"], use_container_width=True)
        _csv_button(db["orders"], "orders_db.csv", "csv_orders_db")


def page_agent_decisions(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    decisions = load_paper_decisions(paths)
    st.subheader("Recent agent actions (paper decisions.jsonl — simulated)")
    if decisions.empty:
        _empty("No agent decisions recorded yet.")
        return
    cols = [
        c
        for c in (
            "as_of",
            "cycle_id",
            "model_id",
            "model_version",
            "target_weights",
            "approved_weights",
            "risk_decision",
            "dry_run",
        )
        if c in decisions.columns
    ]
    view = decisions[cols].iloc[::-1] if cols else decisions.iloc[::-1]
    st.dataframe(view, use_container_width=True)
    _csv_button(view, "agent_decisions.csv", "csv_agent")

    db = try_load_db_tables(paths)
    if not db["agent_decisions"].empty:
        st.subheader("SQLite agent_decisions")
        st.dataframe(db["agent_decisions"], use_container_width=True)


def page_risk_decisions(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    decisions = load_paper_decisions(paths)
    st.subheader("Risk decisions (from paper cycle log — simulated)")
    if decisions.empty:
        _empty("No risk decisions recorded yet.")
    else:
        cols = [
            c
            for c in (
                "as_of",
                "risk_decision",
                "risk_reason",
                "violated_rules",
                "rejected_orders",
                "model_id",
                "model_version",
            )
            if c in decisions.columns
        ]
        view = decisions[cols].iloc[::-1]
        st.dataframe(view, use_container_width=True)
        _csv_button(view, "risk_decisions.csv", "csv_risk")

    st.subheader("Active risk limits (config — not secrets)")
    limits = load_risk_limits(paths)
    if not limits:
        _empty("No risk section in config.")
    else:
        st.json(limits)

    overview = build_overview(paths)
    ks = overview.kill_switch_active
    if ks is None:
        st.caption("Kill-switch state: unknown (no kill_switch_events in DB).")
    elif ks:
        st.error("Kill-switch ACTIVE (from storage events).")
    else:
        st.success("Kill-switch inactive (latest storage event).")

    db = try_load_db_tables(paths)
    if not db["risk_decisions"].empty:
        st.subheader("SQLite risk_decisions")
        st.dataframe(db["risk_decisions"], use_container_width=True)


def page_model_comparison(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    st.subheader("Agent comparison (research — simulated)")
    cmp_ = load_agent_comparison(paths)
    if not cmp_:
        _empty("No reports/agent_comparison/latest.json found.")
        return
    st.caption(str(cmp_.get("disclaimer") or "Research comparison only."))
    rows = cmp_.get("rows")
    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
        # Drop bulky nested hyperparams for table view if present.
        show = df.drop(columns=["algorithm_hyperparameters"], errors="ignore")
        st.dataframe(show, use_container_width=True)
        _csv_button(show, "agent_comparison.csv", "csv_cmp")
    if isinstance(cmp_.get("random_control"), dict):
        st.subheader("Random-policy control")
        st.json(cmp_["random_control"])


def page_walk_forward(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    st.subheader("Walk-forward results (backtest simulation)")
    wf = load_walk_forward(paths)
    if not wf.get("available"):
        _empty("No walk-forward reports under reports/walk_forward/.")
        return
    st.caption(f"Source: `{wf.get('path')}`")
    if wf.get("aggregate"):
        st.json(wf["aggregate"])
    folds = wf.get("folds")
    if isinstance(folds, pd.DataFrame) and not folds.empty:
        st.dataframe(folds, use_container_width=True)
        _csv_button(folds, "walk_forward_folds.csv", "csv_wf")
    if wf.get("stability"):
        with st.expander("Model stability report"):
            st.json(wf["stability"])


def page_data_quality(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    reports = load_quality_reports(paths)
    st.subheader("Data quality reports")
    dq = reports.get("data_quality") or []
    if not dq:
        _empty("No reports/data_quality/*.json files.")
    else:
        for doc in dq[:5]:
            st.markdown(f"**{doc.get('_source_file', 'report')}**")
            st.json(doc)
        _csv_button(pd.DataFrame(dq), "data_quality.csv", "csv_dq")

    st.subheader("Feature quality reports")
    fq = reports.get("feature_quality") or []
    if not fq:
        _empty("No reports/feature_quality/*.json files.")
    else:
        for doc in fq[:5]:
            st.markdown(f"**{doc.get('_source_file', 'report')}**")
            st.json(doc)
        _csv_button(pd.DataFrame(fq), "feature_quality.csv", "csv_fq")


def page_system_health(paths: DashboardPaths) -> None:
    mode = load_operating_mode(paths)
    _banner(mode)
    health = compute_health(paths)
    overview = build_overview(paths)

    for w in health.warnings:
        st.warning(w)
    if not health.warnings:
        st.success("No stale-data or missing-heartbeat warnings at current thresholds.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Heartbeat status", health.heartbeat_status or "missing")
    c2.metric(
        "Heartbeat age (s)",
        f"{health.heartbeat_age_seconds:.0f}" if health.heartbeat_age_seconds is not None else "—",
    )
    c3.metric(
        "Latest data ts",
        health.latest_data_ts.isoformat() if health.latest_data_ts else "—",
    )

    st.subheader("Process / model")
    st.write(
        {
            "mode": mode_label(mode),
            "model_id": overview.model_id,
            "model_version": overview.model_version,
            "heartbeat_ts": None
            if health.heartbeat_ts is None
            else health.heartbeat_ts.isoformat(),
            "kill_switch_active": overview.kill_switch_active,
        }
    )

    st.subheader("Live trading controls")
    st.error(LIVE_ACTIVATION_BLOCKED_MESSAGE)
    st.checkbox(
        "Enable live trading (disabled)",
        value=False,
        disabled=True,
        key="live_enable_disabled",
    )
    st.caption("The enable control is permanently disabled in this dashboard.")

    db = try_load_db_tables(paths)
    if not db["alerts"].empty:
        st.subheader("Recent DB alerts")
        st.dataframe(db["alerts"].iloc[::-1].head(50), use_container_width=True)
        _csv_button(db["alerts"], "alerts.csv", "csv_alerts")


PAGE_RENDERERS: dict[str, Any] = {
    "Overview": page_overview,
    "Portfolio": page_portfolio,
    "Performance": page_performance,
    "Trades": page_trades,
    "Agent decisions": page_agent_decisions,
    "Risk decisions": page_risk_decisions,
    "Model comparison": page_model_comparison,
    "Walk-forward results": page_walk_forward,
    "Data quality": page_data_quality,
    "System health": page_system_health,
}


def run() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(
        page_title="RL Trader Monitor",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.sidebar.title("RL Trader Monitor")
    st.sidebar.caption("Read-only · simulated paper/backtest · no live activation")

    root = st.sidebar.text_input("Project root", value=str(Path.cwd()))
    db_url = st.sidebar.text_input(
        "Database URL",
        value="sqlite:///./reports/rl_trader.db",
    )
    st.session_state["project_root"] = root
    st.session_state["database_url"] = db_url

    page = st.sidebar.radio("Pages", PAGES, index=0)
    paths = _paths_from_session()
    st.sidebar.markdown("---")
    st.sidebar.caption(SIMULATED_BANNER)

    renderer = PAGE_RENDERERS[page]
    st.title(page)
    renderer(paths)


if __name__ == "__main__":
    run()
