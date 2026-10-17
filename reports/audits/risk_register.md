# Risk register — paper-broker readiness

**Branch:** `cursor/paper-broker-ef36`  
**Audit companion:** [`final_audit.md`](final_audit.md)  
**Operational posture:** controlled paper trading (live disabled)

---

## Open risks

| ID | Title | Severity | Likelihood | Impact | Status | Owner action |
| --- | --- | --- | --- | --- | --- | --- |
| R1 | Compose defaults to `--demo` synthetic loop | medium | high if compose used blindly | Misleading paper performance | **Open / labelled** | Remove `--demo`, mount sealed manifest, inject real `MarketDataProvider` |
| R2 | No shipped live-quote MarketDataProvider for paper loop | medium | certain for non-demo | Controlled paper blocked until wired | **Open** | Implement provider + integration test against real calendar bars |
| R3 | Low-level `PaperBroker.submit_order` skips risk | medium | low (requires custom caller) | Unreviewed paper fills | **Accepted** | Prefer service/executor; do not call broker from agents |
| R4 | Dashboard soft-fails on bad files/DB | medium | medium | Silent empty UI | **Accepted** | Ops check heartbeats + file ledgers, not dashboard alone |
| R5 | EXAMPLE approval / capital thresholds | medium | medium | Premature approval if defaults reused | **Open** | Override gates in config before any promotion |
| R6 | Live path is Stage-0 stub only | high (for live) | n/a while disabled | False confidence if “live” toggled | **Mitigated** | Keep `LIVE_TRADING_ENABLED=false`; no external API |
| R7 | Dual-write DB failure leaves file ledger only | low | low | Incomplete SQL analytics | **Mitigated** | Logged + alerted; file ledger primary |
| R8 | Process crash mid-cycle | low | low | Partial cycle artifacts | **Mitigated** | Lock + idempotent orders + restart resumes next cycle |

---

## Closed this audit

| ID | Title | Severity | Resolution |
| --- | --- | --- | --- |
| C-H1 | Silent fixture fallback in `PaperTradingService.start` | high | Fail-closed when fixtures disabled |
| C-H2 | `configs/paper.yaml` fixtures default true | high | Default `false` |
| C-H3 | Rejected risk targets could hit paper broker | high | Executor + broker guards |
| C-H4 | Router paper-path defense was no-op; live executor no short-circuit | high | Counted live wrap + executor guard |
| C-M2 | Approval persist failures silent | medium | `logging.exception` |

---

## Promotion gates (do not skip)

| Target posture | Blockers that must be cleared |
| --- | --- |
| Controlled paper | H1–H4 closed ✓; non-demo needs provider + sealed approval |
| Shadow live evaluation | Live gates + manual activation + Stage 0 only; still no external fills |
| Restricted live pilot | External broker adapter, capital allowlist, sandbox proof, human approvals — **not present** |

---

## Monitoring signals

- `reports/paper_trading/heartbeat.json` age
- `alerts.jsonl` for `risk_rejected`, `database_dual_write_failed`, stale data
- Process lock contention (exit code 4)
- Log line `failed to persist approval` / `demo mode: provisioned stub`
