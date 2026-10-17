# Paper-trading deployment guide

**Scope:** paper trading only. Live trading is **not** configured or enabled by
these procedures. Compose and Makefile targets force `APP_MODE=paper` and
`LIVE_TRADING_ENABLED=false`.

Related: [`docs/live-trading.md`](live-trading.md) (disabled template),
[`docs/test-plan.md`](test-plan.md).

---

## Architecture (deploy)

| Piece | Role |
|-------|------|
| `scripts/run_paper_trader.py` | Entrypoint; `--dry-run` / `--demo` / `--loop`; SIGTERM graceful stop |
| `ProcessLock` | `reports/paper_trading/paper_trader.lock` — one writer process |
| `heartbeat.json` | Liveness for health checks |
| SQLite + file ledgers | `reports/rl_trader.db`, `reports/paper_trading/`, `data/interim/paper_broker/` |
| Structured logs | stdout (+ rotating files under `logs/` / `MONITORING__LOG_DIR`) |
| Docker Compose | `paper-trader` + read-only `dashboard` |

### Process lock

The paper service acquires an exclusive flock on
`{paper_trading.state_dir}/{lock_filename}` (default
`reports/paper_trading/paper_trader.lock`). A second process exits with code **4**.
Lock is released on graceful `stop()`, `atexit`, and after SIGTERM/SIGINT once the
current cycle finishes.

### Graceful shutdown

With `--loop`, `SIGTERM` / `SIGINT` set a stop flag. The running cycle completes,
then `service.stop()` disconnects the paper broker, writes `heartbeat.status=stopped`,
and releases the lock. Prefer `docker compose stop` / `kill -TERM` over `kill -9`.

### Health check

```bash
python scripts/healthcheck_paper.py \
  --heartbeat reports/paper_trading/heartbeat.json \
  --max-age-seconds 180 \
  --allow-starting
```

Compose wires this as `HEALTHCHECK` on `paper-trader`. One-shot jobs may pass
`--allow-stopped`.

### Scheduler

The in-process loop (`--loop --interval-seconds N`) is the default container
scheduler. For host cron / systemd instead:

```cron
# Every minute — relies on process lock to avoid overlap
* * * * * cd /opt/rl-trader && .venv/bin/python scripts/run_paper_trader.py --config configs/paper.yaml >> logs/cron.log 2>&1
```

Systemd tip: `Restart=on-failure`, `KillSignal=SIGTERM`, `TimeoutStopSec=120`.
Do **not** start a second unit against the same `state_dir`.

---

## 1) Local Linux

### Installation

```bash
git clone <repo> && cd <repo>
git checkout cursor/paper-broker-ef36
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
cp .env.example .env
# Edit .env — keep APP_MODE=paper, LIVE_TRADING_ENABLED=false
```

### Data download

```bash
make download-data CONFIG=configs/research.yaml
# or: python scripts/download_data.py --config configs/research.yaml
```

### Testing

```bash
make test
make lint
make typecheck
make run-smoke
```

### Training / evaluation / walk-forward

```bash
make train-smoke                    # fixture training smoke
make train-agent ALGORITHM=ppo
python scripts/evaluate_agent.py --config configs/research.yaml
make walk-forward-smoke
make walk-forward
make run-benchmarks
```

### Paper dry-run / paper service

```bash
make run-paper-dry                  # demo fixtures, no simulated orders
make run-paper                      # one demo cycle with PaperBroker fills
# Continuous (deploy-like):
python scripts/run_paper_trader.py --config configs/paper.yaml --demo --loop --interval-seconds 60
# Graceful stop: Ctrl-C or kill -TERM <pid>
```

### Monitoring dashboard

```bash
make dashboard
# → http://127.0.0.1:8505
```

### Shutdown

```bash
# Looping process: SIGTERM
kill -TERM "$(cat reports/paper_trading/paper_trader.lock)"
# Or Ctrl-C in the foreground terminal
```

### Backup / restore

```bash
make backup-paper                   # → backups/paper_<UTC>/
make restore-paper SRC=backups/paper_<UTC>
# Recreate .env from .env.example (secrets never stored in backups)
```

### Model / config rollback

```bash
# Keep prior sealed manifests under models/approved/archive/
bash scripts/rollback_model.sh models/approved/archive/manifest.prev.json configs/paper.yaml
bash scripts/rollback_config.sh backups/paper_<UTC>/configs/paper.yaml configs/paper.yaml
# Restart paper service after either rollback
```

---

## 2) Cloud VM

Same as Local Linux, plus:

```bash
# Example Ubuntu 22.04+
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv git build-essential
# Clone, install, configure .env as above

# Optional: systemd unit sketch (paper only)
# /etc/systemd/system/rl-trader-paper.service
# ExecStart=/opt/rl-trader/.venv/bin/python scripts/run_paper_trader.py --config configs/paper.yaml --loop
# Restart=on-failure
# KillSignal=SIGTERM

# Firewall: allow SSH + dashboard 8505 only to trusted CIDRs
sudo ufw allow OpenSSH
sudo ufw allow from <trusted> to any port 8505

# Cron backup daily
0 2 * * * cd /opt/rl-trader && bash scripts/backup_paper.sh /var/backups/rl-trader
```

Persistent disks should cover: `data/`, `reports/`, `models/`, `logs/`, `backups/`.

---

## 3) Containerised server

### Build & start (paper only)

```bash
cp .env.example .env   # keep live flags false
docker compose build
docker compose up -d paper-trader
docker compose up -d dashboard
docker compose ps
docker compose logs -f paper-trader
```

Services:

| Service | Purpose | Ports |
|---------|---------|-------|
| `paper-trader` | Looping paper cycles (`--demo --loop`) | — |
| `dashboard` | Streamlit read-only UI | `8505` |

Volumes: `paper-data`, `paper-reports`, `paper-models`, `paper-logs` (+ bind-mount
`configs/` and `models/approved` read-only).

Logging: Docker `json-file` driver `max-size=10m`, `max-file=5` plus in-app
rotating files under `/app/logs`.

Health: heartbeat script; restart policy `unless-stopped`.

### Common commands

```bash
# One-shot dry-run inside the image (smoke)
docker compose run --rm paper-trader \
  python scripts/run_paper_trader.py --config configs/paper.yaml --demo --dry-run

# Migrations
docker compose run --rm paper-trader \
  sh -c 'DATABASE_URL=sqlite:////app/reports/rl_trader.db python -m alembic upgrade head'

# Shutdown
docker compose stop paper-trader dashboard
docker compose down          # keeps named volumes
docker compose down -v       # DELETES volumes — destructive

# Backup from host (volume contents)
docker compose stop paper-trader
bash scripts/backup_paper.sh backups   # if bind-mounted; else docker cp from volumes
docker compose start paper-trader
```

For training / walk-forward, prefer a host venv or a one-off compose `run` with
CPU/memory limits — keep the long-running `paper-trader` service paper-only.

---

## Backup procedure

1. Stop or quiesce the paper writer (`docker compose stop paper-trader` or SIGTERM).
2. `bash scripts/backup_paper.sh backups`
3. Verify `BACKUP_TIMESTAMP` + `MANIFEST.txt` in the new directory.
4. Copy off-box (object storage / another disk).
5. Restart the paper service.

Backups include: configs (no `.env`), paper state, SQLite DB, paper broker ledger,
approved models / checkpoints when present. **Secrets are never copied.**

## Restoration procedure

1. Stop paper service.
2. `bash scripts/restore_paper.sh backups/paper_<UTC>`
3. Recreate `.env` from `.env.example` (re-enter secrets manually).
4. `make migrate` / alembic upgrade if schema drifted.
5. Start paper service; confirm `make paper-health` or compose health.

## Model rollback

1. Keep prior **sealed** paper-approved manifests (e.g. `models/approved/archive/`).
2. `bash scripts/rollback_model.sh models/approved/archive/<prior>.json`
3. Restart paper service (lock released → new process loads new manifest).
4. Confirm decisions JSONL shows the rolled-back `model_id` / `content_hash`.

## Configuration rollback

1. Prefer restoring from a backup:  
   `bash scripts/rollback_config.sh backups/paper_<UTC>/configs/paper.yaml`
2. Script refuses sources with `app_mode: live` or `live_trading_enabled: true`.
3. Restart paper service.

---

## Env-var handling

- Template: `.env.example` (committed). Runtime: `.env` (gitignored).
- Compose `env_file: .env` plus hard-coded paper overrides.
- Never commit API keys. Never put secrets in SQLite (storage layer sanitises).

---

## Smoke test record

Executed on this branch: **`make run-paper-dry` → exit 0**.  
Details: [`reports/audits/paper_deploy_smoke.md`](../reports/audits/paper_deploy_smoke.md).
