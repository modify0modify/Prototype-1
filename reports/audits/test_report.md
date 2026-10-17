# Test report — comprehensive coverage pass

**Branch:** `cursor/paper-broker-ef36`  
**Date:** 2026-10-17  
**Live trading:** remains **inactive** (not enabled by this work)

## Quality gates

| Command | Result |
|---------|--------|
| `pytest -q` (`make test`) | **285 passed**, 2 warnings (gymnasium box bounds) |
| `ruff check src tests scripts` (`make lint`) | **All checks passed** |
| `mypy src` (`make typecheck`) | **Success: no issues found in 148 source files** |

Prior mypy failures in live broker / approval evidence were fixed as part of this pass.

## Counts

| Suite area | Approx. nodes |
|------------|---------------|
| Total collected / passed | 285 |
| New unit (reward/costs/accounting, order sizing, failure modes) | 16 |
| New integration (pipeline chains) | 6 |
| New regression guards | 8 |
| Pre-existing (config, data, features, DCF, risk, paper, live gates, approval, …) | remainder |

## Gaps closed

Plan: [`docs/test-plan.md`](../../docs/test-plan.md)

| Required group | Gap closed by |
|----------------|---------------|
| Reward / transaction costs / portfolio accounting | `tests/unit/test_reward_costs_accounting.py` |
| Order sizing | `tests/unit/test_order_sizing.py` |
| Integration chains (data→…→reconcile) | `tests/integration/test_pipeline_chains.py` |
| Regression pack (leakage, determinism, kill-switch, scaler, …) | `tests/regression/test_regression_guards.py` |
| Failure modes (missing data, stale, disconnect, partial fill, …) | `tests/unit/test_failure_modes.py` |

Existing suites already covered configuration, data validation, features, DCF, risk rules, paper broker, model approval, and live-disabled guards; the plan maps each required case to a node id.

## Files added / updated

**Added**

- `docs/test-plan.md`
- `reports/audits/test_report.md`
- `tests/unit/test_reward_costs_accounting.py`
- `tests/unit/test_order_sizing.py`
- `tests/unit/test_failure_modes.py`
- `tests/integration/test_pipeline_chains.py`
- `tests/regression/test_regression_guards.py`

**Updated (mypy / lint)**

- `src/rl_trader/brokers/live.py` — use `RiskDecision` / non-optional portfolio
- `src/rl_trader/execution/live.py` — typed `dict[str, Any]`
- `src/rl_trader/approval/evidence.py` — remove unused ignore

## Confirmation

No live trading activated. Defaults and Stage-0 shadow policy unchanged; suites assert live remains blocked without full approval / arming.
