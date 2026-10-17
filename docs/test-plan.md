# RL Trader — Comprehensive Test Plan

Maps every required case to a pytest node id (existing or new). Synthetic
fixtures are labelled in test names / docstrings. **Live trading is never
activated** by these suites.

Run quality gates:

```bash
make test        # pytest -q
make lint        # ruff check src tests scripts
make typecheck   # mypy src
```

---

## Unit groups

| Case | Test file / node id | Status |
|------|---------------------|--------|
| Configuration validation | `tests/unit/test_config_validation.py` (all) | existing |
| YAML overlays / modes | `tests/unit/test_yaml_config.py` (all) | existing |
| Data validation (dupes, jumps, negatives) | `tests/unit/test_data_validation.py` (all) | existing |
| Features (technical / regime / portfolio) | `tests/unit/test_features_technical.py` (all) | existing |
| Feature leakage guards | `tests/unit/test_feature_leakage.py` (all) | existing |
| DCF / FCFF / as-of | `tests/unit/test_dcf.py` (all) | existing |
| Reward calculation | `tests/unit/test_reward_costs_accounting.py::test_reward_breakdown_penalties` | **new** |
| Reward no double-count costs | `tests/unit/test_overfitting_audit_guards.py::test_reward_does_not_double_count_embedded_costs` | existing |
| Portfolio accounting (cash/equity/weights) | `tests/unit/test_reward_costs_accounting.py::test_portfolio_accounting_after_fill` | **new** |
| Transaction costs components | `tests/unit/test_reward_costs_accounting.py::test_transaction_cost_components` | **new** |
| Env cost wiring | `tests/unit/test_portfolio_env.py::test_costs_reduce_nav_on_rebalance` | existing |
| Risk rules (all limits) | `tests/unit/test_risk_engine.py` (all) | existing |
| Order sizing / notional resize | `tests/unit/test_order_sizing.py::test_order_notional_resizes_to_cap` | **new** |
| Order sizing cash reserve | `tests/unit/test_order_sizing.py::test_cash_reserve_scales_order` | **new** |

---

## Integration groups

| Case | Test file / node id | Status |
|------|---------------------|--------|
| data → features | `tests/integration/test_pipeline_chains.py::test_data_to_features_chain` | **new** |
| data pipeline cache/raw | `tests/integration/test_data_pipeline.py` (all) | existing |
| features → environment | `tests/integration/test_pipeline_chains.py::test_features_to_environment_chain` | **new** |
| environment → agent | `tests/integration/test_pipeline_chains.py::test_environment_to_agent_chain` | **new** |
| agent → risk engine | `tests/integration/test_pipeline_chains.py::test_agent_to_risk_engine_chain` | **new** |
| risk → PaperBroker | `tests/integration/test_paper_trading_cycle.py::test_submit_target_applies_costs` | existing |
| risk → PaperBroker (sized) | `tests/integration/test_pipeline_chains.py::test_risk_to_paper_broker_chain` | **new** |
| fills → portfolio reconciliation | `tests/integration/test_pipeline_chains.py::test_fills_to_portfolio_reconciliation` | **new** |
| model approval → paper service | `tests/integration/test_paper_trading_loop.py::test_prospective_paper_loop_end_to_end` | existing |
| unapproved model blocked | `tests/unit/test_paper_trading_service.py::test_approved_model_only_load` | existing |
| storage dual-write | `tests/integration/test_storage_paper_dual_write.py::test_paper_dual_write_and_file_ledger` | existing |

---

## Regression groups

| Case | Test file / node id | Status |
|------|---------------------|--------|
| No look-ahead leakage | `tests/regression/test_regression_guards.py::test_no_lookahead_leakage` | **new** (+ existing feature/env tests) |
| Deterministic fixture results | `tests/regression/test_regression_guards.py::test_deterministic_fixture_results` | **new** |
| Saved-model compatibility | `tests/regression/test_regression_guards.py::test_saved_model_compatibility` | **new** (+ `test_ppo_training` / checkpoint) |
| No duplicate orders | `tests/regression/test_regression_guards.py::test_no_duplicate_orders` | **new** |
| No live activation without approval | `tests/regression/test_regression_guards.py::test_no_live_activation_without_approval` | **new** |
| No order after kill-switch | `tests/regression/test_regression_guards.py::test_no_order_after_kill_switch` | **new** |
| No fundamentals before publication | `tests/regression/test_regression_guards.py::test_no_fundamentals_before_publication` | **new** |
| No scaler fit on val/test | `tests/regression/test_regression_guards.py::test_no_scaler_fit_on_val_test` | **new** |

---

## Failure groups

| Case | Test file / node id | Status |
|------|---------------------|--------|
| Missing market data | `tests/unit/test_failure_modes.py::test_missing_market_data` | **new** |
| Malformed price | `tests/unit/test_failure_modes.py::test_malformed_price` | **new** |
| Stale quote | `tests/unit/test_failure_modes.py::test_stale_quote` (+ paper/risk) | **new**/existing |
| Disconnected broker | `tests/unit/test_failure_modes.py::test_disconnected_broker` | **new** |
| Database failure | `tests/unit/test_failure_modes.py::test_database_failure_fail_closed` | **new** |
| Partial fill | `tests/unit/test_failure_modes.py::test_partial_fill` | **new** |
| Rejected order | `tests/unit/test_failure_modes.py::test_rejected_order` | **new** |
| Process restart | `tests/unit/test_failure_modes.py::test_process_restart_idempotency` | **new** (+ paper service) |
| Corrupted model | `tests/unit/test_failure_modes.py::test_corrupted_model` | **new** |
| Mismatched features | `tests/unit/test_failure_modes.py::test_mismatched_features` | **new** |
| Invalid risk configuration | `tests/unit/test_failure_modes.py::test_invalid_risk_configuration` | **new** |

---

## Live safety (must stay inactive)

| Case | Node id |
|------|---------|
| Default blocked | `tests/unit/test_smoke.py::test_live_trading_blocked_by_default` |
| Stage 0 shadow only | `tests/unit/test_live_trading_integration.py::test_stage_0_never_submits_live_orders` |
| Incomplete vars | `tests/unit/test_live_trading_integration.py::test_live_cannot_activate_with_incomplete_safety_variables` |
| Paper ≠ live router | `tests/unit/test_live_trading_integration.py::test_paper_orders_cannot_reach_live_adapter` |

---

## Explicitly out of scope

- Real broker network calls
- Enabling `live_trading_enabled` in CI defaults
- Unsupervised live capital increases
