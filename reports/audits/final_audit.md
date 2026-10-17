# Final production-readiness audit

**Branch:** `cursor/paper-broker-ef36`  
**Date:** 2026-10-17  
**Scope:** Full repo — correctness, financial accounting, costs, leakage, overfitting, broker isolation, paper/live separation, risk, config, DB integrity, process recovery, logging, secrets, tests, docs, reproducibility  
**Live activation:** not performed (intentionally disabled / Stage-0 stub)

---

## Conclusion

**Ready for controlled paper trading**

Not ready for shadow live evaluation as an operational posture (Stage-0 stub exists but live remains disabled by default and requires multi-gate activation). Not ready for restricted live pilot (live broker is sandbox/shadow-only; no external live API).

---

## Severity summary (this audit)

| Severity | Found | Fixed this pass | Remaining |
| --- | ---: | ---: | ---: |
| critical | 0 | 0 | **0** |
| high | 4 | 4 | **0** |
| medium | 6 | 2 (hardened) | **4** (documented residuals) |
| low | 4 | 2 | **2** |

---

## High findings (fixed)

### H1 — Silent fixture market-data fallback

1. **Where:** `src/rl_trader/paper_trading/service.py` (`PaperTradingService.start`, ~L153–163)
2. **Risk:** With `use_fixture_market_data=false`, `start()` still constructed `FixtureMarketDataProvider` when no provider was injected — synthetic prices could run as “controlled paper” without labelling.
3. **Severity:** high
4. **Correction:** Refuse start with `RuntimeError` when fixtures are disabled and no provider is supplied.
5. **Regression:** `tests/regression/test_audit_regressions.py::test_service_refuses_silent_fixture_fallback`

### H2 — Paper YAML defaulted fixtures on

1. **Where:** `configs/paper.yaml` (`paper_trading.use_fixture_market_data`)
2. **Risk:** Default deploy config enabled synthetic market data for all paper runs.
3. **Severity:** high
4. **Correction:** Default set to `false`; `--demo` remains the only labelled fixture path.
5. **Regression:** `tests/regression/test_audit_regressions.py::test_paper_yaml_disables_fixture_market_data_by_default`

### H3 — Risk REJECTED could reach paper broker / executor

1. **Where:** `src/rl_trader/execution/paper.py` (`PaperExecutor.execute`); `src/rl_trader/brokers/paper.py` (`PaperBroker.submit_target`)
2. **Risk:** Defense-in-depth gap — a caller skipping the service’s REJECTED branch could still submit. Broker previously only checked `portfolio is None`.
3. **Severity:** high
4. **Correction:** Executor short-circuits before broker; broker raises on `RiskDecision.REJECTED`.
5. **Regression:** `test_paper_executor_never_calls_broker_on_risk_reject`, `test_paper_broker_rejects_risk_rejected_targets`

### H4 — Live executor lacked reject short-circuit; router defense was a no-op

1. **Where:** `src/rl_trader/execution/live.py` (`LiveExecutor.execute`); `src/rl_trader/live_trading/routing.py` (`OrderRouter.submit`)
2. **Risk:** Live path called `submit_target` even after reject (broker already blocked — gap was layered). Paper-routing “defense” used a bare `pass` that never detected live adapter use.
3. **Severity:** high
4. **Correction:** LiveExecutor mirrors paper short-circuit. Router wraps live adapter so any `submit_order` increments a counter; paper path fails if the counter moves.
5. **Regression:** `test_live_executor_never_calls_broker_on_risk_reject`, `test_order_router_detects_paper_path_calling_live`

---

## Medium findings

### M1 — Compose stack defaults to `--demo` (residual)

1. **Where:** `docker-compose.yml` paper-trader command
2. **Risk:** Operators may treat the default compose loop as controlled paper with real feeds; it uses synthetic fixtures + stub approval.
3. **Severity:** medium (labelled; intentional smoke path)
4. **Correction:** Header comments require removing `--demo` for controlled paper; docs updated. No real `MarketDataProvider` beyond fixtures ships yet.
5. **Regression:** `test_compose_demo_flag_is_labelled_smoke_only`

### M2 — Approval DB persist soft-fail was silent (fixed)

1. **Where:** `src/rl_trader/approval/service.py` (`maybe_persist_verified`)
2. **Risk:** Persistence failures left no operational signal (in-memory approval still valid).
3. **Severity:** medium
4. **Correction:** `logging.exception` on failure.
5. **Regression:** `test_approval_persist_failure_is_logged`

### M3 — No production MarketDataProvider implementation (residual)

1. **Where:** `src/rl_trader/paper_trading/market_data.py`
2. **Risk:** Controlled paper with live quotes requires an injected provider implementing the protocol; research `YFinanceProvider` is not wired into the paper loop.
3. **Severity:** medium
4. **Correction:** Fail-closed without fixtures (H1). Residual: operator must supply a provider for non-demo paper.
5. **Regression:** covered by H1

### M4 — Direct `PaperBroker.submit_order` bypasses risk (residual)

1. **Where:** `src/rl_trader/brokers/paper.py` (`submit_order`)
2. **Risk:** Low-level API can place orders without `RiskEngine` — acceptable for unit tests; production path is `PaperTradingService` / `PaperExecutor` → risk → `submit_target`.
3. **Severity:** medium
4. **Correction:** Documented; production entrypoints always review first. Broker `submit_target` rejects REJECTED.
5. **Regression:** executor/broker reject tests

### M5 — Dashboard loaders swallow parse/DB errors (residual)

1. **Where:** `src/rl_trader/monitoring/dashboard/loaders.py` (~L44, L361–364)
2. **Risk:** Read-only UI may show empty panels instead of errors; does not affect trading.
3. **Severity:** medium → accepted for dashboard UX
4. **Correction:** none this pass (non-trading surface)
5. **Regression:** n/a

### M6 — EXAMPLE capital / approval thresholds in defaults (residual)

1. **Where:** `src/rl_trader/config/models.py` (approval / live stage capital fields)
2. **Risk:** Operators shipping defaults without review.
3. **Severity:** medium
4. **Correction:** Fields labelled EXAMPLE; human override required before live gates.
5. **Regression:** existing live-gate incompleteness tests

---

## Low findings

### L1 — Bare `pass` in secret heuristic (fixed)

`monitoring/dashboard/safety.py` — cleaned dead branch; heuristic still flags secret patterns.

### L2 — Optional import `pass` in seed helpers (accepted)

`utils/seeds.py` — ImportError for numpy/torch optional; intentional.

### L3 — Fundamentals early-publication `pass` (accepted)

`fundamentals/statements.py` — allows publication_date &lt; fiscal_period_end; as-of path still point-in-time.

### L4 — Paper broker RNG reject path (accepted)

`reject_probability` defaults to 0.0; seeded when enabled.

---

## Area review (pass / residual)

| Area | Verdict |
| --- | --- |
| Correctness / portfolio accounting | Pass — costs applied on fill; reconcile + weight-error checks |
| Transaction costs | Pass — commission/spread/slippage configurable; paper + env models |
| Data leakage | Pass — prior overfitting audit + asof fundamentals; fail-closed stale data |
| Model overfitting | Pass — walk-forward / approval gates; demo stub labelled |
| Broker isolation | Pass — agents have no broker imports; research live-import boundary |
| Paper/live separation | Pass — router + credential profiles + compose forces paper |
| Risk controls | Pass — fail-closed missing context; kill switch; notional resize |
| Configuration | Pass — paper.yaml fixtures off; live flags false |
| Database integrity | Pass — alembic + dual-write with logged failure |
| Process recovery | Pass — flock lock, heartbeat, SIGTERM finish-cycle |
| Logging / secrets | Pass — masking helpers; `.env.example` empty secrets |
| Tests / docs / reproducibility | Pass — regression suite + deployment guide; seeded RNGs |

---

## Search checklist

| Pattern | Result |
| --- | --- |
| TODO / FIXME in `src/` | None blocking |
| Mocked production logic | `--demo` / fixture providers labelled only |
| Bare `pass` | Cleared in router + dashboard safety; remaining are intentional |
| Hard-coded credentials / account IDs | None in prod paths (test IDs only) |
| Hard-coded financial thresholds | EXAMPLE-labelled defaults only |
| Swallowed exceptions | Approval persist now logs; dashboard soft-fail residual |
| Future-data access | Guarded (asof / execution delay / stale checks) |
| Agent → broker calls | None; boundary scanners enforce |
| Risk-engine bypass | Production paths blocked; low-level `submit_order` residual |

---

## Gates (this pass)

See commit message / CI locally: `pytest`, `ruff`, `mypy` required green before push.
