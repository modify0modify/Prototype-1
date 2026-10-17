"""Enforce: research modules must not import live brokers/execution."""

from __future__ import annotations

from pathlib import Path

from rl_trader.utils.boundaries import (
    FORBIDDEN_LIVE_MODULES,
    assert_research_has_no_live_imports,
    assert_risk_has_no_agent_imports,
    scan_file_for_forbidden_imports,
)


def test_research_surface_has_no_live_imports() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "rl_trader"
    assert_research_has_no_live_imports(package_root)


def test_risk_surface_has_no_agent_imports() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "rl_trader"
    assert_risk_has_no_agent_imports(package_root)


def test_forbidden_modules_documented() -> None:
    assert "rl_trader.brokers.live" in FORBIDDEN_LIVE_MODULES
    assert "rl_trader.execution.live" in FORBIDDEN_LIVE_MODULES


def test_scan_detects_forbidden_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("from rl_trader.brokers.live import LiveBroker\n", encoding="utf-8")
    assert scan_file_for_forbidden_imports(bad) == ["rl_trader.brokers.live"]
