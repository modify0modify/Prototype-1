"""Unit tests for every live-trading safety gate (fail-fast before broker connect)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_trader.config import (
    LIVE_TRADING_CONFIRMATION_PHRASE,
    AppSettings,
    LiveTradingBlockedError,
    RiskSettings,
    RuntimeMode,
    assert_live_trading_allowed,
    evaluate_live_activation,
)
from rl_trader.config.models import (
    BacktestSettings,
    BrokerSettings,
    DatabaseSettings,
    DataSettings,
    FeatureSettings,
    FundamentalValuationSettings,
    LiveTradingSettings,
    MonitoringSettings,
    TrainingSettings,
)
from rl_trader.risk.startup import run_risk_startup_checks


def _write_manifest(path: Path, *, approved: bool = True) -> Path:
    """Write a sealed formal approval manifest for live-gate tests."""
    from rl_trader.approval.testing import (
        write_sealed_approved_manifest,
        write_sealed_unapproved_manifest,
    )

    root = path.parent / (path.stem + "_bundle")
    if approved:
        sealed = write_sealed_approved_manifest(root, tier="live", model_id="unit-test-model")
    else:
        sealed = write_sealed_unapproved_manifest(root, model_id="unit-test-model")
    # Keep the path callers expect.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sealed.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def _live_base(tmp_path: Path, **overrides: object) -> AppSettings:
    """Build settings that pass all live gates unless overrides break one."""
    manifest = _write_manifest(tmp_path / "manifest.json", approved=True)
    kwargs: dict[str, object] = {
        "_env_file": None,
        "app_mode": RuntimeMode.LIVE,
        "live_trading_enabled": True,
        "data": DataSettings(survivorship_bias_acknowledged=True),
        "broker": BrokerSettings(environment="live"),
        "risk": RiskSettings(
            max_position_size=0.1,
            max_portfolio_exposure=0.5,
            daily_loss_limit=0.02,
        ),
        "live": LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=manifest,
            approved_account_allowlist=("LIVEACCT123456",),
            rollout_stage=0,
            manual_activated=False,
        ),
    }
    kwargs.update(overrides)
    return AppSettings(**kwargs)  # type: ignore[arg-type]


def _failed_names(settings: AppSettings) -> set[str]:
    report = evaluate_live_activation(settings)
    return {c.name for c in report.failed}


def test_confirmation_phrase_constant() -> None:
    assert LIVE_TRADING_CONFIRMATION_PHRASE == "I_UNDERSTAND_THIS_ENABLES_LIVE_BROKER_ORDERS"


def test_gate_survivorship_universe_policy(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        data=DataSettings(
            point_in_time_universe=False,
            survivorship_bias_acknowledged=False,
        ),
    )
    assert "survivorship_universe_policy" in _failed_names(settings)


def test_all_gates_pass_allows_live(tmp_path: Path) -> None:
    settings = _live_base(tmp_path)
    report = assert_live_trading_allowed(settings)
    assert report.allowed is True
    assert report.failed == ()


def test_gate_app_mode_live(tmp_path: Path) -> None:
    settings = _live_base(tmp_path, app_mode=RuntimeMode.PAPER, live_trading_enabled=False)
    assert "app_mode_live" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="app_mode_live"):
        assert_live_trading_allowed(settings)


def test_gate_live_trading_enabled(tmp_path: Path) -> None:
    settings = _live_base(tmp_path, live_trading_enabled=False)
    assert "live_trading_enabled" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="live_trading_enabled"):
        assert_live_trading_allowed(settings)


def test_gate_broker_environment_live(tmp_path: Path) -> None:
    settings = _live_base(tmp_path, broker=BrokerSettings(environment="paper"))
    assert "broker_environment_live" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="broker_environment_live"):
        assert_live_trading_allowed(settings)


def test_gate_live_trading_acknowledgement(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement="wrong-phrase",
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=_write_manifest(tmp_path / "m.json"),
        ),
    )
    assert "live_trading_acknowledgement" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="live_trading_acknowledgement"):
        assert_live_trading_allowed(settings)


def test_gate_acknowledgement_rejects_whitespace_variant(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE + " ",
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=_write_manifest(tmp_path / "m.json"),
        ),
    )
    assert "live_trading_acknowledgement" in _failed_names(settings)


def test_gate_approved_model_manifest_missing(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=tmp_path / "does-not-exist.json",
        ),
    )
    assert "approved_model_manifest" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="approved_model_manifest"):
        assert_live_trading_allowed(settings)


def test_gate_approved_model_manifest_not_approved(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path / "denied.json", approved=False)
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=path,
        ),
    )
    assert "approved_model_manifest" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="approved_model_manifest"):
        assert_live_trading_allowed(settings)


def test_gate_risk_engine_startup(tmp_path: Path) -> None:
    bad = RiskSettings.model_construct(
        max_position_size=0.9,
        max_portfolio_exposure=0.5,
        min_weight=0.0,
        daily_loss_limit=0.02,
        max_drawdown_limit=0.2,
    )
    report = run_risk_startup_checks(bad)
    assert report.ok is False
    assert any(c.name == "position_within_exposure" and not c.passed for c in report.checks)

    settings = _live_base(tmp_path).model_copy(update={"risk": bad})
    assert "risk_engine_startup" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="risk_engine_startup"):
        assert_live_trading_allowed(settings)


def test_gate_live_account_id(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="   ",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=_write_manifest(tmp_path / "m.json"),
        ),
    )
    assert "live_account_id" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="live_account_id"):
        assert_live_trading_allowed(settings)


def test_gate_max_live_capital(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=None,
            max_order_value=1_000.0,
            approved_model_manifest=_write_manifest(tmp_path / "m.json"),
        ),
    )
    assert "max_live_capital" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="max_live_capital"):
        assert_live_trading_allowed(settings)


def test_gate_max_order_value(tmp_path: Path) -> None:
    settings = _live_base(
        tmp_path,
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=None,
            approved_model_manifest=_write_manifest(tmp_path / "m.json"),
        ),
    )
    assert "max_order_value" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="max_order_value"):
        assert_live_trading_allowed(settings)


def test_gate_daily_loss_limit(tmp_path: Path) -> None:
    """Missing daily loss limit: rejected at AppSettings load for live, and by gate."""
    with pytest.raises(ValueError, match="daily_loss_limit"):
        _live_base(
            tmp_path,
            risk=RiskSettings(
                max_position_size=0.1,
                max_portfolio_exposure=0.5,
                daily_loss_limit=None,
            ),
        )

    # Gate-level coverage via model_construct (bypasses AppSettings risk rule).
    manifest = _write_manifest(tmp_path / "manifest.json")
    settings = AppSettings.model_construct(
        app_mode=RuntimeMode.LIVE,
        live_trading_enabled=True,
        seed=42,
        data=DataSettings(),
        features=FeatureSettings(),
        training=TrainingSettings(),
        backtest=BacktestSettings(),
        risk=RiskSettings.model_construct(
            max_position_size=0.1,
            max_portfolio_exposure=0.5,
            min_weight=0.0,
            daily_loss_limit=None,
            max_drawdown_limit=0.2,
        ),
        broker=BrokerSettings(environment="live"),
        monitoring=MonitoringSettings(),
        database=DatabaseSettings(),
        fundamentals=FundamentalValuationSettings(),
        live=LiveTradingSettings(
            acknowledgement=LIVE_TRADING_CONFIRMATION_PHRASE,
            account_id="LIVEACCT123456",
            max_live_capital=25_000.0,
            max_order_value=1_000.0,
            approved_model_manifest=manifest,
        ),
        broker_environment=None,
        live_trading_acknowledgement=None,
        live_account_id=None,
        live_max_capital=None,
        live_max_order_value=None,
        approved_model_manifest=None,
        database_url=None,
        project_root=Path.cwd(),
    )
    assert "daily_loss_limit" in _failed_names(settings)
    with pytest.raises(LiveTradingBlockedError, match="daily_loss_limit"):
        assert_live_trading_allowed(settings)
