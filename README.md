# RL Trading Platform (Skeleton)

Modular reinforcement-learning stock-trading scaffold. **No trained model or live broker integration yet** — typed stubs, config, safety guards, and smoke tests only.

## Pipeline (future)

```
Historical data → validation → feature engineering → RL training
  → validation → OOS backtest → walk-forward → paper trading
  → risk approval → optional live deployment
```

## Safety principles

- **RL never sends orders.** Agents emit target portfolios only.
- **Deterministic risk engine** approves, rejects, or resizes targets before any execution adapter runs.
- **Default mode is paper.** Live routing is blocked unless multi-check activation passes.
- **No accidental live trading.** Mode, explicit enable flag, activation token, and broker credentials must all be present.
- **Secrets via env vars** (see `.env.example`). Never commit `.env` or hard-code API keys.
- **No silent missing-data fills.** Gaps must fail validation or be explicitly audited.
- **Look-ahead bias prevention** is documented in data/features/evaluation stubs; causal transforms only.

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

## Top-level modules (`src/rl_trading/`)

| Module | Purpose |
|--------|---------|
| `config` | pydantic-settings for research / paper / live; env-backed secrets |
| `logging` | Structured (text/JSON) logging setup |
| `data` | Historical download + validation stubs (no silent fills) |
| `features` | Causal feature-engineering interfaces |
| `rl` | Gymnasium env, agent interfaces, deterministic seed hooks |
| `evaluation` | OOS / backtest metric stubs |
| `walk_forward` | Walk-forward fold planning stubs |
| `benchmarks` | Non-RL baselines (equal-weight, buy-and-hold) |
| `risk` | Deterministic approve / reject / resize engine |
| `paper` | Simulated execution adapter (default safe path) |
| `live` | Multi-check live activation guards (blocked by default) |
| `persistence` | SQLAlchemy / SQLite storage |
| `audit` | Decision / order audit recording |

Supporting repo paths: `configs/` (mode TOML templates), `tests/`, `scripts/`, `data/` + `artifacts/` placeholders, `notebooks/`.

## Smoke checks

```bash
pytest -q
python -m rl_trading.cli
```

These import the package, assert paper-mode defaults, and confirm live trading is blocked without full activation.
