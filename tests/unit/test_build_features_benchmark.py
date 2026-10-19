"""build_features must emit parquet for the benchmark (Bugbot finding)."""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "build_features.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("build_features_under_test", _SCRIPT)
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


def test_build_features_writes_benchmark_parquet(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    ohlcv_dir = processed / "ohlcv"
    ohlcv_dir.mkdir(parents=True)
    features_dir = tmp_path / "features"
    days = _business_days(date(2020, 1, 2), date(2020, 3, 31))
    for i, sym in enumerate(("SPY", "QQQ", "AAPL")):
        rng = np.random.default_rng(7 + i)
        close = 80.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=len(days)))
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
            ohlcv_dir / f"{sym}_{days[0].isoformat()}_{days[-1].isoformat()}.parquet"
        )

    cfg = {
        "app_mode": "research",
        "seed": 0,
        "live_trading_enabled": False,
        "data": {
            "universe": ["SPY", "QQQ", "AAPL"],
            "provider": "yfinance",
            "timezone": "UTC",
            "processed_dir": str(processed),
            "survivorship_bias_acknowledged": True,
        },
        "features": {
            "benchmark_symbol": "SPY",
            "output_dir": str(features_dir),
            "manifest_dir": str(features_dir),
            "quality_report_dir": str(tmp_path / "fq"),
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
            "causal_only": True,
            "drop_na_rows": False,
        },
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    mod = _load_mod()
    code = mod.main(["--config", str(cfg_path)])
    assert code == 0
    assert (features_dir / "SPY_features.parquet").is_file()
    assert (features_dir / "QQQ_features.parquet").is_file()
    assert (features_dir / "AAPL_features.parquet").is_file()
