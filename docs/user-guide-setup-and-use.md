# RL Trading System — Setup and Use Guide

**Package:** `rl_trader`  
**Repository:** [modify0modify/Prototype-1](https://github.com/modify0modify/Prototype-1)  
**Grounded commit:** `71d5803` (`cursor/paper-broker-ef36` / main tip)

> **SAFETY:** This guide covers research, backtesting, and **paper trading only**. It does not enable live trading, place real broker orders, provide investment advice, or promise returns.

**Audience:** Intermediate Python developers and quant students familiar with git, virtual environments, CLI tools, and basic Docker.

**Prepared:** 17 October 2026 (rewritten against the checked-out codebase; Copilot “Verify in repo” hedges removed where Makefile/docs confirm commands).

---

## Outline / table of contents

1. Purpose, scope, and safety posture
2. Architecture and operating modes
3. Installation: local Linux, cloud VM, and Docker Compose
4. Research workflow: data, features, benchmarks, training, comparison, walk-forward
5. Model approval for paper trading
6. Paper-trading operator runbook and first-day checklist
7. Risk and broker mental model
8. Configuration and secrets cheat sheet
9. Troubleshooting FAQ
10. Glossary and out-of-scope topics

---

## 1. What this project is — and is not

RL Trading System is a modular Python project under `src/rl_trader/`. It supports research, backtesting, multi-algo training (PPO / SAC / TD3), formal model approval, and **controlled paper trading** with a simulated `PaperBroker`.

It is **not** investment advice, a profitability guarantee, or permission to trade real capital. Live trading ships as a gated Stage-0 sandbox/stub (`configs/live.yaml`, `docs/live-trading.md`) and is **out of scope** for this guide.

**Sources:** `README.md`, `docs/deployment.md`, `docs/live-trading.md`, `reports/audits/final_audit.md`

### 1.1 End-to-end pipeline

```
Historical / market data
        |
        v
Validation → causal feature build → benchmarks / train / walk-forward
        |                                      |
        |                                      v
        |                           checkpoint + evaluation artifacts
        |                                      |
        |                                      v
        +----------------------------> approval gates
                                               |
                                               v
                                  hash-sealed approved manifest
                                               |
                                               v
Fresh market data → features → agent target weights → RiskEngine
                                                        |
                                  approved / resized / rejected
                                                        |
                                                        v
                                         PaperBroker simulation
                                                        |
                                                        v
                         reconcile → decisions / metrics / alerts / ledger
                                                        |
                                                        v
                                           read-only dashboard
```

> **SAFETY:** The agent proposes target weights. It is not the broker and must never bypass the RiskEngine or route directly to an execution adapter.

### 1.2 Core safety principles

- **Agent ≠ broker:** inference proposes portfolio targets; it does not send orders.
- **Risk approval is mandatory:** every proposed change passes through the RiskEngine, which may approve, resize, or reject it (`DecisionRecord`).
- **Paper is the operating default** for this guide (`APP_MODE=paper`, `LIVE_TRADING_ENABLED=false`).
- **No silent data fills:** missing, stale, or invalid data fails validation or refuses the cycle. With `use_fixture_market_data: false`, the paper service **refuses to start** if no real `MarketDataProvider` is injected (no silent fixture fallback).
- **Look-ahead awareness:** features must be causal; candidates are selected on validation only; the held-out test set is reported after selection.
- **Secrets live in environment variables.** Never place credentials in YAML, logs, source, images, or git. Never commit `.env` (listed in `.gitignore`).
- **Approval thresholds and capital limits in config are EXAMPLE defaults**, not recommendations.
- **Transaction costs stay enabled** for fair research and paper simulation (commission / spread / slippage).

### 1.3 Operating modes

| Mode | Purpose | Status in this guide |
|------|---------|----------------------|
| Research | Data validation, feature work, benchmarks, training, comparison, walk-forward | Covered |
| Backtest | Historical simulation under explicit cost and causality assumptions | Covered as part of research |
| Paper | Simulated orders, costs, reconciliation, monitoring | Covered — **controlled paper** is the target posture |
| Live | Real broker order path | Disabled / Stage-0 stub — **not enabled here** |

### 1.4 Readiness posture: controlled paper trading

`reports/audits/final_audit.md` concludes: **Ready for controlled paper trading**.

Not ready for shadow-live as an operational posture, and not ready for a restricted live pilot (live broker remains sandbox/stub; no external live API).

For a new operator, **controlled** means a bounded, observable, reversible paper deployment:

- approved (hash-sealed) model only
- explicit configuration (`configs/paper.yaml` + `.env`)
- real-data provider when fixtures are off (`use_fixture_market_data: false`)
- risk checks on every decision
- process lock + heartbeat + health monitoring
- recorded decisions and paper broker ledger
- tested shutdown, backup, restore, and rollback

> **WARNING:** Controlled paper readiness is not live-trading readiness. It is also not evidence of future returns.

See also `reports/audits/deployment_checklist.md` (Path A = labelled demo/smoke; Path B = controlled paper).

---

## 2. Installation

> **SAFETY:** Keep `LIVE_TRADING_ENABLED=false`. Do not set live flags, add live-broker credentials, or expose the dashboard broadly. Secrets belong only in environment variables; `.env` must stay out of git.

### 2.1 Common preparation

Requires **Python 3.12** (`pyproject.toml`: `requires-python = ">=3.12,<3.13"`).

```bash
git clone https://github.com/modify0modify/Prototype-1
cd Prototype-1
# Paper-broker tip used for this guide (also matches GitHub main tip ~71d5803):
git checkout cursor/paper-broker-ef36   # or: git switch main
git status --short --branch
git rev-parse --short HEAD
```

```bash
cp .env.example .env
# Edit .env locally. Do not commit it.
```

Confirm paper defaults:

```bash
grep -E '^(APP_MODE|LIVE_TRADING_ENABLED|BROKER_ENVIRONMENT)=' .env.example
# Expect:
# APP_MODE=paper
# LIVE_TRADING_ENABLED=false
# BROKER_ENVIRONMENT=paper

git check-ignore -v .env
# Expect a .gitignore hit (e.g. .gitignore:2:.env)
```

Paper-relevant env names from `.env.example`:

| Variable | Paper-safe value |
|----------|------------------|
| `APP_MODE` | `paper` |
| `LIVE_TRADING_ENABLED` | `false` |
| `BROKER_ENVIRONMENT` | `paper` |
| `LIVE_TRADING_ACKNOWLEDGEMENT` | empty |
| `LIVE_ACCOUNT_ID` / `LIVE_MAX_*` | empty |
| `APPROVED_MODEL_MANIFEST` | `models/approved/manifest.json` (after you seal a real one) |
| `DATABASE_URL` | `sqlite:///./reports/rl_trader.db` |
| `BROKER__PAPER_API_KEY` / `BROKER__PAPER_API_SECRET` | optional paper-only placeholders |
| `BROKER__LIVE_*` | **leave empty** |

### 2.2 Three installation paths

| Path | Best for | Core approach | Operational note |
|------|----------|---------------|------------------|
| A. Local Linux | Development and supervised paper tests | Python 3.12 venv, editable install, SQLite migration | Keep the shell supervised during initial runs |
| B. Cloud VM | Long-running controlled paper | Same as A; optional systemd / cron from `docs/deployment.md` | Restrict inbound access; dashboard port 8505 should not be public |
| C. Docker Compose | Reproducible `paper-trader` + `dashboard` | Compose services as shipped | Default compose includes `--demo` (smoke only) |

#### A. Local Linux — venv and editable install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
# make install creates .venv (if needed) and runs: pip install -e ".[dev]"
```

Equivalent without Make (same result as the install target’s pip step):

```bash
pip install -e ".[dev]"
```

Apply database migrations:

```bash
make migrate
# DATABASE_URL=sqlite:///./reports/rl_trader.db .venv/bin/python -m alembic upgrade head
```

#### B. Cloud VM

Same clone / venv / install / migrate steps as Path A. On Ubuntu 22.04+:

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git build-essential
```

For a long-running loop, prefer the in-process scheduler documented in `docs/deployment.md`:

```bash
.venv/bin/python scripts/run_paper_trader.py --config configs/paper.yaml --loop --interval-seconds 60
```

Optional systemd sketch (paper only) is documented in `docs/deployment.md` — `Restart=on-failure`, `KillSignal=SIGTERM`, `TimeoutStopSec=120`. Do **not** start a second unit against the same `state_dir`.

> **WARNING:** The dashboard binds at **http://127.0.0.1:8505** (`make dashboard` → `scripts/run_dashboard.py --host 127.0.0.1 --port 8505`). Prefer SSH port forwarding or a tightly restricted firewall; do not expose `:8505` to the public Internet.

```bash
# From your workstation:
ssh -L 8505:127.0.0.1:8505 user@vm-host
# Then open http://127.0.0.1:8505 locally.
```

Firewall example from `docs/deployment.md`:

```bash
sudo ufw allow OpenSSH
sudo ufw allow from <trusted> to any port 8505
```

#### C. Docker Compose

Services: `paper-trader` and `dashboard` (`docker-compose.yml`, project name `rl-trader-paper`).

```bash
cp .env.example .env   # keep live flags false
docker compose config
docker compose build
docker compose up -d paper-trader
docker compose up -d dashboard
docker compose ps
docker compose logs -f paper-trader
```

Makefile shortcuts:

```bash
make docker-build
make docker-up      # docker compose up -d paper-trader
make docker-logs
make docker-down
```

> **DEMO-ONLY:** Compose’s default `paper-trader` command includes `--demo` (synthetic fixture market data + stub approved model). That is a **labelled smoke/deploy path**, not controlled paper. For controlled paper, remove `--demo`, mount a sealed approved manifest under `models/approved/`, inject a real `MarketDataProvider`, and keep `paper_trading.use_fixture_market_data: false`.

### 2.3 Verification gate

```bash
make test
make lint
make typecheck
make run-smoke
```

> **WARNING:** Do not proceed to paper operation if tests, lint, type checking, migration, or smoke verification fails. Resolve failures against `README.md`, `docs/deployment.md`, and `docs/test-plan.md`.

---

## 3. Research workflow

> **WARNING:** Fixture or synthetic data must be labelled (`--fixture`, `--demo`, smoke targets). Never select a model on test-set performance. Keep transaction costs enabled for fair comparisons. Benchmarks and RL results are research tools — **not investment advice**.

Default research config: `configs/research.yaml` (`CONFIG ?= configs/research.yaml` in the Makefile). Seed: `42`.

### 3.1 Download and validate historical data

Provider: **yfinance** (`YFinanceProvider`). Missing dates are never interpolated.

```bash
make download-data
# or: python scripts/download_data.py --config configs/research.yaml
```

| Artifact | Path |
|----------|------|
| Cache | `data/interim/cache/` |
| Raw (immutable) | `data/raw/{provider}/` |
| Cleaned OHLCV | `data/processed/ohlcv/` |
| Quality reports | `reports/data_quality/` |

### 3.2 Build causal features

```bash
make build-features
# or: python scripts/build_features.py --config configs/research.yaml
```

| Artifact | Path |
|----------|------|
| Feature matrices | `data/processed/features/{SYMBOL}_features.parquet` |
| Feature manifest | `data/processed/features/{SYMBOL}_feature_manifest.json` |
| Quality report | `reports/feature_quality/{SYMBOL}_feature_quality.json` |

Warm-up / missing values stay as explicit NaNs (not silently dropped).

### 3.3 Deterministic benchmarks

Seven baselines share one harness (cash-only, buy-and-hold SPY, equal-weight, MA trend, momentum, mean-reversion, inverse-volatility). **No RL training** is started.

```bash
make run-benchmarks
# → reports/benchmarks/latest.md (+ JSON)
```

### 3.4 Train a smoke model, then a full model

```bash
make train-smoke
# → scripts/train_agent.py --config configs/research.yaml --algorithm ppo --fixture
```

Smoke training is a plumbing check, not a candidate-selection result.

```bash
make train-agent ALGORITHM=ppo
# Accepted algorithms: ppo | sac | td3
```

Checkpoints land under `models/checkpoints/`. A checkpoint alone is **not** an approved paper model.

### 3.5 Compare PPO, SAC, and TD3

```bash
make compare-agents
# → scripts/compare_agents.py --config configs/research.yaml --fixture
# Artifacts: reports/agent_comparison/
```

Use validation-only selection. Comparison reports do **not** declare a winner. A random-policy control checks the env does not accidentally mint profits.

### 3.6 Walk-forward evaluation

```bash
make walk-forward-smoke
# → --fixture --no-train → reports/walk_forward/

make walk-forward
# full rolling train → val → test (no test leakage; scaler fit on train only)
```

Artifacts include `fold_summary.csv`, trade logs, equity/drawdown charts, and stability reports under `reports/walk_forward/`.

Optional evaluation entrypoint:

```bash
python scripts/evaluate_agent.py --config configs/research.yaml
```

### 3.7 Expected artifact areas

| Area | Expected role | Operator check |
|------|---------------|----------------|
| `data/` | Raw, validated, or feature data | Confirm subdirectories and fixture labels |
| `models/checkpoints/` | Training checkpoints | Do not treat a checkpoint as approved |
| `models/approved/` | Approved manifest / model bundle | Verify hash and gate checks before paper |
| `reports/` | Benchmarks, comparison, walk-forward, paper ledgers, audits | Record config, seed, data identity, costs, selection basis |

---

## 4. Approve a model for paper trading

Paper operation must reject an unapproved model. Approval binds model identity and evaluation evidence into a **hash-sealed** manifest checked before load (`src/rl_trader/approval/`).

### 4.1 Hash-sealed manifest

Start from the template (do not invent fields):

```text
models/approved/manifest.example.json
```

Key fields (exact names in the example):

- Identity: `model_id`, `model_version`, `algorithm`, `n_assets`, `checkpoint_dir`
- Periods: `train_period`, `val_period`, `test_period`
- Features: `feature_manifest`
- Evidence: `benchmark_results`, `evaluation_evidence`, `transaction_cost_assumptions`
- Approval: `approval_status` (`paper` or `live`), `approval_timestamp`, `approver`, `code_commit_id`
- Integrity: `content_hash` (and optional `signature`)

**Hash scheme:** canonical JSON (sorted keys, compact separators) of all fields except `content_hash` / `signature` → SHA-256 hex. Optional HMAC-SHA256 when `approval.require_manifest_signature: true` (key from `RL_TRADER_MANIFEST_SIGNING_KEY`).

Seal / verify with the package API (example):

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
from rl_trader.approval import seal_manifest, verify_manifest, write_manifest_file, load_manifest_file

src = Path("models/approved/manifest.example.json")
payload = json.loads(src.read_text())
# Fill real evaluation_evidence, periods, checkpoint_dir, code_commit_id, etc.
# Then reseal — the template content_hash is a placeholder and will fail verify.
sealed = seal_manifest(payload)
out = Path("models/approved/manifest.json")
write_manifest_file(out, sealed)
print("sealed →", out, "hash=", verify_manifest(load_manifest_file(out)))
PY
```

> Do **not** edit `content_hash` by hand to force a pass. Re-run `seal_manifest` after legitimate field updates.

### 4.2 Configurable approval gates

Paper gates live under `approval.paper.*` in `configs/paper.yaml`. They are labelled **EXAMPLE defaults** — override for your risk appetite. Examples include:

- `min_oos_return`, `max_drawdown`, `max_turnover`
- `min_evaluated_market_periods`, `min_regimes_evaluated`
- `require_transaction_costs_included`, `require_feature_manifest_match`
- `require_regression_tests_passed`, `require_benchmark_comparison`

Paper accepts `approval_status` in `{paper, live}` and evaluates **paper gates only**. Live-tier gates exist in config for completeness but are **not used** by this guide.

### 4.3 Point the paper service at the approved manifest

Default path: `live.approved_model_manifest: models/approved/manifest.json` in `configs/paper.yaml` (the paper service reads this field even though mode is paper).

Override via env:

```bash
# in .env
APPROVED_MODEL_MANIFEST=models/approved/manifest.json
```

Or roll back to a prior sealed file:

```bash
bash scripts/rollback_model.sh models/approved/archive/manifest.prev.json configs/paper.yaml
```

### 4.4 Demo shortcut

> **DEMO-ONLY**

```bash
python scripts/run_paper_trader.py --config configs/paper.yaml --demo
# Makefile: make run-paper / make run-paper-dry  (both pass --demo)
```

`--demo` provisions a **stub** approved model under `reports/paper_trading/demo_model/` and forces fixture market data. Logs include: `demo mode: provisioned stub approved model`. This is plumbing only — **not** model approval and **not** controlled-paper evidence.

### 4.5 Controlled paper with real data

Requirements (`configs/paper.yaml` + audit Path B):

1. `paper_trading.use_fixture_market_data: false` (repo default)
2. **No** `--demo` on the command / Compose service
3. Inject a real `MarketDataProvider` implementing the paper protocol (research `YFinanceProvider` is **not** auto-wired into the paper loop — residual M3 in `final_audit.md`)
4. Human-sealed paper-tier manifest (not the demo stub)

If fixtures are off and no provider is supplied, `PaperTradingService.start` raises `RuntimeError` (fail-closed). Do **not** turn fixtures back on and call the result controlled paper.

---

## 5. Paper-trading operator runbook

### 5.1 Preflight

Use `reports/audits/deployment_checklist.md`. Condensed:

- [ ] Branch tip known; `make test` / `lint` / `typecheck` green
- [ ] `.env` from `.env.example` — **no** live credentials
- [ ] `APP_MODE=paper`, `LIVE_TRADING_ENABLED=false`, `BROKER_ENVIRONMENT=paper`
- [ ] `use_fixture_market_data: false` for controlled paper (or labelled `--demo` for smoke only)
- [ ] Risk limits reviewed (EXAMPLE defaults)
- [ ] Sealed approved manifest path set
- [ ] `make migrate` applied; log dir writable

### 5.2 Dry run — decisions without simulated orders

```bash
make run-paper-dry
# → python scripts/run_paper_trader.py --config configs/paper.yaml --demo --dry-run
```

`--dry-run` still runs gates / risk decisions but **skips** `PaperBroker` simulated orders. Inspect `reports/paper_trading/decisions.jsonl` before continuing.

For a controlled-paper dry run (after removing demo and injecting a provider):

```bash
python scripts/run_paper_trader.py --config configs/paper.yaml --dry-run
```

### 5.3 One-shot or loop

One demo cycle with PaperBroker fills:

```bash
make run-paper
# → --config configs/paper.yaml --demo   (single cycle; no --loop)
```

Continuous deploy-like loop:

```bash
python scripts/run_paper_trader.py --config configs/paper.yaml --demo --loop --interval-seconds 60
# Graceful stop: Ctrl-C or kill -TERM <pid>
```

Docker (smoke as shipped):

```bash
docker compose up -d paper-trader
docker compose up -d dashboard
```

### 5.4 What one cycle should do

Aligned with `scripts/run_paper_trader.py` and `PaperTradingService`:

1. Refuse non-paper mode / live flags / non-paper broker environment
2. (If `--demo`) provision stub approved model + fixture market data
3. Acquire exclusive process lock (`reports/paper_trading/paper_trader.lock`)
4. Load and verify the approved sealed manifest; load model; check feature compatibility
5. Obtain market / feature inputs; enforce freshness (`max_data_age_seconds`, default 300)
6. Agent emits **target weights only**
7. RiskEngine approves / resizes / rejects → `DecisionRecord`
8. On approve/resize: `PaperBroker.submit_target` (skipped on `--dry-run`); apply costs
9. Reconcile; append `decisions.jsonl`; update `metrics.json` / `alerts.jsonl` / heartbeat
10. Release lock on `stop()` (SIGTERM finishes the current cycle first)

Exit codes of note: `2` approved-model gate failed; `3` stale data / safe shutdown; `4` process lock conflict.

### 5.5 Lock, heartbeat, and graceful shutdown

- Lock path: `reports/paper_trading/paper_trader.lock` (configurable via `paper_trading.lock_filename`)
- Second writer exits with code **4** (`ProcessLockError`)
- Heartbeat: `reports/paper_trading/heartbeat.json`
- Prefer `SIGTERM` / `docker compose stop` over `kill -9`

```bash
# If the lock file stores the PID (deployment docs example):
kill -TERM "$(cat reports/paper_trading/paper_trader.lock)"
```

> **WARNING:** Do not delete a lock while its owner may still be running.

### 5.6 Health check

```bash
make paper-health
# → python scripts/healthcheck_paper.py \
#      --heartbeat reports/paper_trading/heartbeat.json --allow-stopped
```

Deploy-style (matches Compose HEALTHCHECK):

```bash
python scripts/healthcheck_paper.py \
  --heartbeat reports/paper_trading/heartbeat.json \
  --max-age-seconds 180 \
  --allow-starting
```

### 5.7 Operational records

Configured under `paper_trading.*` in `configs/paper.yaml`:

| Record | Default path | Purpose |
|--------|--------------|---------|
| `decisions.jsonl` | `reports/paper_trading/decisions.jsonl` | Per-cycle proposal + risk decision |
| `metrics.json` | `reports/paper_trading/metrics.json` | Heartbeat/cycle/recon metrics |
| `alerts.jsonl` | `reports/paper_trading/alerts.jsonl` | Operational / risk alerts |
| `heartbeat.json` | `reports/paper_trading/heartbeat.json` | Liveness |
| Paper broker ledger | `data/interim/paper_broker/` | Simulated cash, positions, fills, costs |
| SQLite (optional dual-write) | `reports/rl_trader.db` | When `database.persist_paper_trading: true` |

### 5.8 Dashboard

```bash
make dashboard
# → python scripts/run_dashboard.py --host 127.0.0.1 --port 8505
```

Open **http://127.0.0.1:8505**. Read-only Streamlit UI over paper ledgers and reports. Results are labelled **simulated**; the UI does not activate live trading.

### 5.9 Shutdown, backup, restore, and rollback

```bash
# Graceful local shutdown
kill -TERM <paper-process-pid>

# Compose
docker compose stop paper-trader dashboard
docker compose down          # keeps named volumes
# docker compose down -v     # DELETES volumes — destructive

# Backup → backups/paper_<UTC>/
make backup-paper
# or: bash scripts/backup_paper.sh backups

# Restore (stop writer first)
make restore-paper SRC=backups/paper_<UTC>
# Recreate .env from .env.example — secrets are never in backups

# Model / config rollback
bash scripts/rollback_model.sh models/approved/archive/manifest.prev.json configs/paper.yaml
bash scripts/rollback_config.sh backups/paper_<UTC>/configs/paper.yaml configs/paper.yaml
# rollback_config.sh refuses sources with app_mode: live or live_trading_enabled: true
```

Backups include configs (no `.env`), paper state, SQLite DB, paper broker ledger, and approved models/checkpoints when present.

### 5.10 First successful day checklist

- [ ] Branch/commit and deployment configuration recorded
- [ ] Quality gates passed; migration current
- [ ] No live activation setting changed; no live credentials present
- [ ] Approved manifest and hashes verified **or** run clearly labelled `--demo`
- [ ] Data provider fresh; fixture/synthetic mode correctly labelled or off
- [ ] Dry run produced expected decisions
- [ ] Paper loop holds one lock; heartbeat stays fresh; healthcheck passes
- [ ] Risk decisions show intelligible approved / resized / rejected reasons
- [ ] Paper ledger reconciles; simulated costs present
- [ ] Dashboard at http://127.0.0.1:8505 shows paper/simulated labels
- [ ] Graceful SIGTERM path tested
- [ ] Backup completed; restore/rollback reviewed

---

## 6. Risk and broker mental model

### 6.1 Sequence

```
MarketDataProvider → FeatureBuilder → Agent → RiskEngine → PaperBroker → Records
       |                  |             |          |              |           |
       |--- fresh data -->|             |          |              |           |
       |                  |-- vector -->|          |              |           |
       |                  |             |-- target weights ------>|           |
       |                  |             |          |-- approve / resize /     |
       |                  |             |          |   reject --------------->|
       |                  |             |          |              |-- fills ->|
       |                  |             |          |              |           |-- decisions /
       |                  |             |          |              |              metrics / alerts
```

- **Agent:** proposes target weights only; cannot modify risk limits, approval status, credentials, mode, or kill switch.
- **RiskEngine:** mandatory gate; may trip kill switch; reset requires `kill_switch_reset_phrase` (`MANUAL_KILL_SWITCH_RESET` in paper YAML — EXAMPLE).
- **PaperBroker:** simulated fills with commission / spread / slippage; `submit_target` raises on `RiskDecision.REJECTED`.
- **Live path:** stubbed / Stage 0; paper submits must never reach the live adapter (`OrderRouter`).

> **SAFETY:** Production entrypoints always review with RiskEngine first. Do not call low-level `PaperBroker.submit_order` from operator workflows (unit-test residual only).

---

## 7. Configuration and secrets cheat sheet

| Concern | Config / env | Safe paper default | Notes |
|---------|--------------|--------------------|-------|
| Application mode | `APP_MODE` / `app_mode` | `paper` | Must agree with runner |
| Live guard | `LIVE_TRADING_ENABLED` | `false` | Never change for this guide |
| Broker environment | `BROKER_ENVIRONMENT` / `broker.environment` | `paper` | |
| Seed | `SEED` / `seed` | `42` | Record with reports |
| Universe | `data.universe` | paper.yaml: SPY, QQQ, AAPL, MSFT | Research YAML has a wider list |
| Risk limits | `risk.*` / `RISK__*` | EXAMPLE defaults | Review every limit |
| Fixtures | `paper_trading.use_fixture_market_data` | `false` | `--demo` is the labelled fixture path |
| Approval | `approval.paper.*` | EXAMPLE thresholds | Not investment advice |
| Manifest path | `APPROVED_MODEL_MANIFEST` / `live.approved_model_manifest` | `models/approved/manifest.json` | Must be sealed |
| Database | `DATABASE_URL` | `sqlite:///./reports/rl_trader.db` | `make migrate` before ops |
| State / ledgers | `paper_trading.state_dir` | `reports/paper_trading` | |
| Broker ledger | `paper_trading.broker_data_dir` | `data/interim/paper_broker` | |
| Logs | `MONITORING__LOG_DIR` | `logs` | |
| Dashboard | `make dashboard` | `127.0.0.1:8505` | Monitoring only |
| Paper broker secrets | `BROKER__PAPER_*` | env only | Never in YAML/git |
| Live broker secrets | `BROKER__LIVE_*` | empty | Unused in this guide |

### 7.1 Never set for this guide

- `LIVE_TRADING_ENABLED=true`
- `APP_MODE=live` / `BROKER_ENVIRONMENT=live`
- `LIVE_TRADING_ACKNOWLEDGEMENT=I_UNDERSTAND_THIS_ENABLES_LIVE_BROKER_ORDERS`
- Non-empty `LIVE_ACCOUNT_ID`, `LIVE_MAX_CAPITAL`, `LIVE_MAX_ORDER_VALUE`
- Any `BROKER__LIVE_*` credentials
- Promoting compose/`--demo` PnL as controlled-paper evidence

---

## 8. Troubleshooting FAQ

### 8.1 Make target missing or wrong interpreter

**Symptom:** `make: *** No rule to make target...` or wrong Python.

**Fix:** Confirm you are at the repo root with the checked-out Makefile. Targets include `install`, `test`, `lint`, `typecheck`, `run-smoke`, `download-data`, `build-features`, `run-benchmarks`, `train-agent`, `train-smoke`, `compare-agents`, `walk-forward`, `walk-forward-smoke`, `run-paper`, `run-paper-dry`, `migrate`, `dashboard`, `paper-health`, `backup-paper`, `restore-paper`, `docker-*`. Use `PYTHON=python3.12` if needed.

### 8.2 Second paper process exits with code 4

**Cause:** Process lock held (`reports/paper_trading/paper_trader.lock`).

**Fix:** Stop the first process with SIGTERM; wait for lock release. Do not delete the lock while the owner may still be running.

### 8.3 Heartbeat is stale or missing

**Cause:** Service crashed, hung mid-cycle, or never started.

**Fix:** Check logs under `logs/`; run `make paper-health`; restart after investigating. Compose health budget is 180s.

### 8.4 Model load fails (exit code 2)

**Cause:** Missing/unsealed/tampered manifest, feature mismatch, or gate failure.

**Fix:** Re-run `verify_manifest` / reseal with `seal_manifest`; restore a sealed archive via `rollback_model.sh`. Never hand-edit hashes.

### 8.5 Service refuses to start when fixtures are disabled

**Cause:** `use_fixture_market_data: false` and no injected `MarketDataProvider` (intentional fail-closed).

**Fix:** Inject a real provider for controlled paper, **or** use labelled `--demo` for smoke only. Do not flip fixtures on and claim controlled paper.

### 8.6 Risk result is REJECTED or resized

**Cause:** Limit, kill switch, stale data, exposure, or other gate.

**Fix:** Read `decisions.jsonl` / `alerts.jsonl` for the reason. Correct data/config/state. Do not bypass RiskEngine.

### 8.7 Database migration needed

```bash
make migrate
make db-current
```

Back up paper state first when applicable.

### 8.8 Docker starts in demo / synthetic mode

**Cause:** Compose command includes `--demo` by design.

**Fix:** For controlled paper, edit the service command to remove `--demo`, set fixtures false, mount a sealed manifest, and inject a real provider. Confirm with `docker compose config`.

---

## 9. Glossary

| Term | Meaning in this guide |
|------|------------------------|
| Agent | Model component that proposes target portfolio weights; not a broker |
| Approved manifest | Hash-sealed metadata identifying an approved model and evaluation evidence |
| Backtest | Historical simulation under explicit data, timing, and cost assumptions |
| Causal feature | Feature computed without future information |
| Controlled paper trading | Bounded, observable, reversible simulated operation with approved artifacts and enforced risk |
| `DecisionRecord` | Structured risk decision (`src/rl_trader/risk/decisions.py`) |
| `--demo` / fixture data | Labelled synthetic smoke path — not controlled paper |
| Heartbeat | Freshness signal in `heartbeat.json` |
| HPO | Hyperparameter optimization — out of scope |
| Kill switch | Stops/rejects after configured conditions; reset is deliberate |
| Look-ahead bias | Using future information in features, selection, or evaluation |
| PaperBroker | Simulated execution adapter with recorded fills/costs |
| Reconciliation | Checking expected portfolio/cash vs paper ledger |
| RiskEngine | Mandatory control that approves, resizes, or rejects targets |
| Target weights | Desired allocation proposed by the agent before risk approval |
| Walk-forward | Time-ordered train/val/test evaluation mimicking sequential deployment |

---

## 10. Out of scope

- Live trading, live broker configuration, production accounts, real orders, and any live activation procedure
- Restricted live pilot or Stage-0→N enablement steps (live mentioned only as a disabled boundary)
- Hyperparameter optimization (HPO)
- Investment recommendations, portfolio advice, “good enough” Sharpe/drawdown targets, or return guarantees
- Broker-specific account opening, funding, permissions, or production API use
- Changes that bypass approval, risk controls, routing separation, or audit records

> **WARNING:** This document is a software-operations guide for paper trading. It is not investment advice and does not establish that any model is profitable, safe for real capital, or suitable for any person.

---

## Appendix — Primary sources (this rewrite)

1. `README.md`
2. `docs/deployment.md`
3. `docs/live-trading.md`
4. `docs/test-plan.md`
5. `reports/audits/deployment_checklist.md`
6. `reports/audits/final_audit.md`
7. `.env.example`
8. `configs/paper.yaml`, `configs/research.yaml`
9. `Makefile`
10. `docker-compose.yml`, `Dockerfile`
11. `scripts/run_paper_trader.py`, `scripts/run_dashboard.py`, `scripts/download_data.py`, backup/restore/rollback/health scripts
12. `models/approved/manifest.example.json`
13. `src/rl_trader/approval/`, `src/rl_trader/paper_trading/`, `src/rl_trader/risk/`
