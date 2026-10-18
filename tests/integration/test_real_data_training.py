"""Integration: train_agent real-data path vs refuse-missing (offline fixtures)."""

from __future__ import annotations

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_agent.py"


def _load_train_mod():
    spec = importlib.util.spec_from_file_location("train_agent_under_test", _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _business_days(start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _write_panel(processed: Path, symbols: tuple[str, ...], days: list[date]) -> None:
    out = processed / "ohlcv"
    out.mkdir(parents=True, exist_ok=True)
    for i, sym in enumerate(symbols):
        rng = np.random.default_rng(100 + i)
        n = len(days)
        close = 50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, size=n))
        idx = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") for d in days])
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000,
                "adjustment_factor": 1.0,
                "open_adj": close,
                "high_adj": close * 1.01,
                "low_adj": close * 0.99,
                "close_adj": close,
            },
            index=idx,
        )
        frame.to_parquet(
            out / f"{sym}_{days[0].isoformat()}_{days[-1].isoformat()}.parquet"
        )


def _write_training_config(path: Path, processed: Path, features_dir: Path) -> None:
    cfg = {
        "app_mode": "research",
        "seed": 0,
        "live_trading_enabled": False,
        "data": {
            "universe": ["SPY", "QQQ"],
            "provider": "yfinance",
            "timezone": "UTC",
            "start_date": "2020-01-02",
            "end_date": "2020-05-29",
            "processed_dir": str(processed),
            "raw_dir": str(processed.parent / "raw"),
            "interim_dir": str(processed.parent / "interim"),
            "cache_dir": str(processed.parent / "cache"),
            "quality_report_dir": str(processed.parent / "dq"),
            "survivorship_bias_acknowledged": True,
        },
        "features": {
            "lookback_bars": 5,
            "causal_only": True,
            "benchmark_symbol": "SPY",
            "output_dir": str(features_dir),
            "manifest_dir": str(features_dir),
            "quality_report_dir": str(features_dir / "quality"),
            "return_windows": [1, 5],
            "ma_windows": [5],
            "rsi_period": 5,
            "macd_fast": 3,
            "macd_slow": 6,
            "macd_signal": 3,
            "bollinger_period": 5,
            "atr_period": 5,
            "volatility_window": 5,
            "downside_vol_window": 5,
            "volume_zscore_window": 5,
            "drawdown_window": 10,
            "rolling_high_window": 10,
            "regime_trend_window": 5,
            "regime_vol_window": 5,
            "vol_percentile_window": 10,
            "include_portfolio_state": False,
            "include_fundamentals": False,
            "include_regime": True,
            "drop_na_rows": False,
        },
        "training": {
            "algorithm": "ppo",
            "model_id": "integration-real",
            "total_timesteps": 128,
            "batch_size": 32,
            "n_steps": 64,
            "n_epochs": 1,
            "seed": 0,
            "device": "cpu",
            "checkpoint_dir": str(processed.parent / "checkpoints"),
            "train_start": "2020-01-02",
            "train_end": "2020-03-31",
            "val_start": "2020-04-01",
            "val_end": "2020-04-30",
            "test_start": "2020-05-01",
            "test_end": "2020-05-29",
            "eval_freq": 64,
            "n_eval_episodes": 1,
            "early_stopping_patience": 2,
            "min_evals_before_stop": 1,
            "normalize_features": True,
            "evaluate_test_after_training": True,
            "buffer_size": 2000,
            "learning_starts": 64,
        },
        "environment": {
            "initial_cash": 10000.0,
            "max_weight": 0.5,
            "commission_bps": 0.0,
            "bid_ask_spread_bps": 0.0,
            "slippage_bps": 0.0,
            "market_impact_coeff": 0.0,
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_train_agent_refuses_missing_processed_data(tmp_path: Path) -> None:
    train_mod = _load_train_mod()
    cfg = tmp_path / "missing.yaml"
    processed = tmp_path / "processed_empty"
    processed.mkdir()
    _write_training_config(cfg, processed, tmp_path / "features")
    code = train_mod.main(
        ["--config", str(cfg), "--algorithm", "ppo", "--timesteps", "64"]
    )
    assert code == 2


def test_train_agent_uses_real_loader_when_fixtures_present(tmp_path: Path) -> None:
    train_mod = _load_train_mod()
    processed = tmp_path / "processed"
    features_dir = tmp_path / "features"
    days = _business_days(date(2020, 1, 2), date(2020, 5, 29))
    _write_panel(processed, ("SPY", "QQQ"), days)
    cfg = tmp_path / "real.yaml"
    _write_training_config(cfg, processed, features_dir)
    out = tmp_path / "out"
    code = train_mod.main(
        [
            "--config",
            str(cfg),
            "--algorithm",
            "ppo",
            "--timesteps",
            "128",
            "--model-id",
            "integration-real",
            "--output-dir",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "feature_scaler.json").is_file()
    assert (out / "training_manifest.json").is_file()
    scaler = json.loads((out / "feature_scaler.json").read_text(encoding="utf-8"))
    assert scaler["fitted_on"] == "train"
    manifest = json.loads((out / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_selected_on"] == "validation"
