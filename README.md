# RL Trading System (Architecture Skeleton)

Modular reinforcement-learning stock-trading scaffold under `src/rl_trader/`.
**No trained model or live broker integration yet** — typed stubs, YAML configs, safety guards, and tests.

## Pipeline (future)

```
Historical data → validation → feature engineering → RL training
  → validation → OOS backtest → walk-forward → paper trading
  → risk approval → optional live deployment
```

## Safety principles

- **RL never sends orders.** Agents emit target portfolios only.
- **Risk engine is independent** of the RL agent; it alone approves / rejects / resizes.
- **Research logic is separate from broker execution.**
- **Broker adapters share one interface**; paper and live credentials are separate env vars.
- **Default mode is paper** (`APP_MODE=paper`). Live routing needs multi-check activation.
- **No silent missing-data fills.** Gaps fail validation or require an audited remediation.
- **Look-ahead bias prevention** is documented on data/features/validation stubs.
- **Secrets are never logged**; account IDs are masked in repr/logs.

## Configuration

Typed nested models (pydantic / pydantic-settings) cover: **data**, **features**, **training**,
**backtest**, **risk**, **broker**, **monitoring**, and **database** (plus fundamentals + live
activation). YAML under `configs/{research,backtest,paper,live}.yaml` supplies structured
defaults; environment variables override (see `.env.example`).

Modes: `research` | `backtest` | `paper` | `live`. Root default is **paper**.

### Live-trading confirmation phrase

Exact value required in `LIVE_TRADING_ACKNOWLEDGEMENT` (character-for-character):

```text
I_UNDERSTAND_THIS_ENABLES_LIVE_BROKER_ORDERS
```

Defined as `rl_trader.config.LIVE_TRADING_CONFIRMATION_PHRASE`.

### Live gate flow (fail-fast before broker connect)

`assert_live_trading_allowed(settings)` / `evaluate_live_activation(settings)` require **all** of:

1. `APP_MODE=live`
2. `LIVE_TRADING_ENABLED=true`
3. `BROKER_ENVIRONMENT=live`
4. Acknowledgement equals the confirmation phrase above
5. Approved model manifest exists with `"approved": true`
6. Risk-engine startup checks pass
7. Non-empty `LIVE_ACCOUNT_ID`
8. Explicit `LIVE_MAX_CAPITAL` (> 0)
9. Explicit `LIVE_MAX_ORDER_VALUE` (> 0)
10. Explicit `risk.daily_loss_limit`

If any gate fails, the process raises `LiveTradingBlockedError` and **must not** open a live
broker connection.
## Dependency direction

```
research surface          orchestration              gated live
─────────────────         ──────────────             ──────────
config, data,             scripts/*                  brokers.live
features, fundamentals,   execution.paper            execution.live
environment, agents,      brokers.paper
backtesting, validation,
portfolio, risk,
monitoring, storage,
utils
        │                        │                         │
        │   may use →            │   may use →             │
        ▼                        ▼                         ▼
   portfolio / risk         BrokerAdapter            live credentials
   (no broker I/O)          (paper fills)            + activation guards
```

**Hard import rule:** research-surface packages must **never** import
`rl_trader.brokers.live` or `rl_trader.execution.live`.

Enforced by:

1. AST scan in `tests/unit/test_import_boundaries.py` via `assert_research_has_no_live_imports`
2. Runtime caller checks in the live modules (`guard_live_import`)
3. Package `__init__` files that do not re-export live adapters

## Market data pipeline (daily OHLCV)

Provider interface: `MarketDataProvider` (`src/rl_trader/data/provider.py`). Initial
implementation: **yfinance** (`YFinanceProvider`) — swap providers in
`get_provider()` without changing downstream modules.

| Artifact | Path |
|----------|------|
| Cache | `data/interim/cache/` |
| Raw (immutable) | `data/raw/{provider}/` |
| Cleaned | `data/processed/ohlcv/` |
| Quality reports | `reports/data_quality/` |

**Timezone:** all stored bars use **UTC** (midnight UTC = trading session calendar date).

**Adjusted prices:** raw keeps provider `Close` + `Adj Close` unchanged. Cleaned adds
`adjustment_factor = Adj Close / Close` and scaled `open_adj`/`high_adj`/`low_adj`/
`close_adj`. Volume is not split-adjusted. **Missing dates are never interpolated.**

```bash
python scripts/download_data.py --config configs/research.yaml
# or
make download-data
```

## Feature engineering (look-ahead-safe)

Causal technical, regime, portfolio-state, and (publication-date-lagged) fundamental
features. **Row policy:** never drop rows silently — warm-up / missing values stay as
explicit NaNs and are summarized in the quality report.

| Artifact | Path |
|----------|------|
| Feature matrices | `data/processed/features/{SYMBOL}_features.parquet` |
| Feature manifest | `data/processed/features/{SYMBOL}_feature_manifest.json` |
| Quality report | `reports/feature_quality/{SYMBOL}_feature_quality.json` |

```bash
python scripts/build_features.py --config configs/research.yaml
make build-features
```

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env   # optional local overrides
```

Or via Make:

```bash
make install
make test
make lint
make typecheck
make run-smoke
```

## Top-level layout

| Path | Purpose |
|------|---------|
| `configs/*.yaml` | research / paper / live mode templates |
| `data/{raw,interim,processed,external}` | dataset stages |
| `models/{checkpoints,approved}` | training artifacts + approved models |
| `reports/{backtests,walk_forward,paper_trading}` | evaluation outputs |
| `scripts/` | thin CLI entrypoints (stubs) |
| `src/rl_trader/` | typed package (see modules below) |
| `tests/{unit,integration,regression}` | test suites |

### Package modules (`src/rl_trader/`)

| Module | Purpose |
|--------|---------|
| `config` | pydantic-settings + YAML overlays; separate paper/live secrets |
| `data` | historical download + validation (research) |
| `features` | causal feature specs / builders |
| `fundamentals` | point-in-time fundamental data stubs |
| `environment` | Gymnasium portfolio env |
| `agents` | RL agents + checkpoint metadata (features + training period) |
| `backtesting` | OOS metrics + benchmarks |
| `validation` | feature-match checks + walk-forward plans |
| `portfolio` | target portfolio types |
| `execution` | paper executor (live executor gated, not re-exported) |
| `risk` | deterministic approve / reject / resize |
| `brokers` | shared `BrokerAdapter`; paper + guarded live |
| `monitoring` | structured logging + audit recorder |
| `storage` | SQLAlchemy / SQLite |
| `utils` | seeds + import-boundary helpers |

## Smoke checks

```bash
pytest -q
python -m rl_trader.cli
```

These import the package, assert paper-mode defaults, confirm live trading is blocked, enforce research↔live import boundaries, and refuse checkpoint loads on feature mismatch.
