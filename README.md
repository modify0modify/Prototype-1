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
- **Default mode is paper.** Live routing needs multi-check activation.
- **No silent missing-data fills.** Gaps fail validation or require an audited remediation.
- **Look-ahead bias prevention** is documented on data/features/validation stubs.

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
