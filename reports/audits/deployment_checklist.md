# Deployment checklist — controlled paper trading

**Branch:** `cursor/paper-broker-ef36`  
**Do not** enable live trading with this checklist.

---

## Pre-flight

- [ ] Branch tip pulled; `pytest`, `ruff`, `mypy` green
- [ ] `.env` copied from `.env.example` — **no** live credentials filled
- [ ] `APP_MODE=paper`, `LIVE_TRADING_ENABLED=false`, `BROKER_ENVIRONMENT=paper`
- [ ] `configs/paper.yaml`: `use_fixture_market_data: false` for controlled paper
- [ ] Risk limits reviewed (not left at EXAMPLE values without intent)
- [ ] Sealed approved model manifest mounted / path set (`APPROVED_MODEL_MANIFEST`)
- [ ] Database URL writable; `alembic upgrade head` applied
- [ ] Log directory writable; rotation settings reviewed

---

## Choose a path

### A) Labelled smoke / demo (synthetic)

- [ ] Use `--demo` **or** compose as shipped (includes `--demo`)
- [ ] Confirm logs show `demo mode: provisioned stub approved model`
- [ ] Do **not** use demo PnL for promotion decisions

### B) Controlled paper (required for audit conclusion)

- [ ] Remove `--demo` from compose/command
- [ ] Confirm start **fails** if no `MarketDataProvider` is injected (fail-closed)
- [ ] Inject real market/feature provider implementing `MarketDataProvider`
- [ ] Use a human-sealed paper-tier approval (not demo stub)
- [ ] Run `--dry-run` for one cycle; inspect `decisions.jsonl` risk records
- [ ] Run one live paper cycle (simulated broker only); verify fills + costs
- [ ] Confirm heartbeat updates; healthcheck passes
- [ ] Confirm second process cannot acquire lock (exit 4)

---

## Runtime ops

- [ ] Single writer: process lock held
- [ ] Heartbeat max age within healthcheck budget (compose: 180s)
- [ ] File ledgers under `reports/paper_trading/` and broker data dir persist across restart
- [ ] Dual-write errors alert but do not halt file ledger
- [ ] SIGTERM finishes current cycle then stops
- [ ] Dashboard (optional) on port 8505 is read-only; cannot activate live

---

## Explicit non-goals

- [ ] **No** `LIVE_TRADING_ENABLED=true`
- [ ] **No** live acknowledgement phrase set
- [ ] **No** live API keys in `.env`
- [ ] **No** promotion to shadow/live without a separate live readiness audit

---

## Rollback

- [ ] `docker compose stop paper-trader` (or kill process; lock releases)
- [ ] Restore SQLite + `reports/paper_trading/` + broker ledger from backup
- [ ] Trip kill switch / keep service stopped until investigation complete

---

## Sign-off

| Role | Name | Date | Notes |
| --- | --- | --- | --- |
| Operator | | | Path A or B? |
| Reviewer | | | Manifest + risk config reviewed |
| Audit | final_audit.md | 2026-10-17 | Conclusion: Ready for controlled paper trading |
