# Overfitting & leakage audit — `rl_trader`

**Branch:** `cursor/rl-trading-skeleton-ef36`  
**Auditor role:** adversarial quantitative review  
**Scope:** data, features, fundamentals, environment, agents, validation, benchmarks, brokers/paper stubs  
**Gate:** paper trading remains **disabled** until criticals are cleared (see status below).

This document lists every finding with severity, location, why it matters, the correction applied, and the regression test. Items reviewed and found OK are summarized at the end.

---

## Severity summary

| Severity | Count (pre-fix) | Remaining after fixes |
| --- | ---: | ---: |
| critical | 3 | **0** |
| high | 6 | 0 (fixed or acknowledged with controls) |
| medium | 4 | 0 fixed / documented residual research limits |
| low | 3 | documented |

**Criticals remaining: 0.** Paper trading is **not** enabled.

---

## Critical findings (fixed)

### C1 — Same-bar close observation + fill (look-ahead)

1. **Where:** `src/rl_trader/environment/portfolio_env.py` (`PortfolioEnv.step`); default `EnvironmentSettings.execution_delay_bars`
2. **Why:** Features/marks at bar `t` include that bar’s close, while `execution_delay_bars=0` filled at `prices[t]` — trading on information not available before the close auction.
3. **Severity:** critical
4. **Correction:** Default `execution_delay_bars=1`. `execution_delay_bars=0` requires `allow_same_bar_execution=True`. YAML configs updated. Benchmark harness queues targets with the same delay.
5. **Regression:** `tests/unit/test_overfitting_audit_guards.py::test_same_bar_execution_requires_explicit_allow`

### C2 — Silent synthetic prices on real calendar dates

1. **Where:** `scripts/train_agent.py` (`main`, non-`--fixture` path)
2. **Why:** Building RNG price paths stamped with configured train/val/test calendar dates invites mistaking synthetic panels for market history (false confidence / pseudo-OOS).
3. **Severity:** critical
4. **Correction:** Non-fixture training refused unless `--allow-synthetic` (loud warning) or `--fixture` (labelled smoke).
5. **Regression:** `tests/unit/test_overfitting_audit_guards.py::test_train_agent_refuses_silent_synthetic`

### C3 — Paper broker ignored transaction costs

1. **Where:** `src/rl_trader/brokers/paper.py` (`PaperBroker`); `scripts/run_paper_trader.py`
2. **Why:** Zero-cost fills vs research `simulate_rebalance` costs make any paper PnL incomparable and unsafe as a go-live gate.
3. **Severity:** critical
4. **Correction:** `PaperBroker` now persists a simulated ledger and applies configurable commission, spread, and slippage on every fill. `PaperTradingDisabledError` is retained only for import compatibility. Live broker remains an unimplemented stub.
5. **Regression:** `tests/unit/test_overfitting_audit_guards.py::test_paper_broker_applies_costs_by_default`; `tests/unit/test_paper_broker.py`; `tests/integration/test_paper_trading_cycle.py`

---

## High findings (fixed)

### H1 — ATR mixed adjusted close with raw high/low

1. **Where:** `src/rl_trader/features/technical.py` (`compute_technical_features`)
2. **Why:** Split events inflate true range when raw OHLC is mixed with `close_adj`.
3. **Severity:** high
4. **Correction:** Prefer `high_adj` / `low_adj` when present.
5. **Regression:** `test_atr_uses_adjusted_high_low_not_raw_mixed_units`

### H2 — Fundamentals visible on publication midnight

1. **Where:** `src/rl_trader/features/fundamental_features.py` (`align_fundamentals_asof`)
2. **Why:** Filings dated `D` were visible for feature rows on `D` (UTC midnight), before typical after-hours availability / next session.
3. **Severity:** high
4. **Correction:** Next-calendar-day lag (`publication_date + 1 day`) before asof merge.
5. **Regression:** `test_fundamentals_not_visible_on_publication_day` (+ updated feature/DCF asof tests)

### H3 — Expanding walk-forward reused prior test as next validation

1. **Where:** `src/rl_trader/validation/windows.py` (`build_walk_forward_plan`)
2. **Why:** Expanding by `step_bars` made fold \(i+1\) validation equal fold \(i\) test → selection leakage / repeated test use.
3. **Severity:** high
4. **Correction:** Purged expanding: next `train_end = previous test_end`; val/test follow after that.
5. **Regression:** `test_expanding_walk_forward_does_not_reuse_prior_test_as_val`

### H4 — Reward double-counted transaction costs

1. **Where:** `src/rl_trader/environment/reward.py` (`compute_reward`)
2. **Why:** Period return already nets simulated costs; also applying `λ_tc · cost_frac` distorted the learning signal (reward hacking risk).
3. **Severity:** high
4. **Correction:** `costs_already_in_return=True` (default) skips explicit TC penalty; opt-in explicit mode retained.
5. **Regression:** `test_reward_does_not_double_count_embedded_costs`

### H5 — Fine-tune path skipped validation selection

1. **Where:** `src/rl_trader/validation/runner.py` (fine-tune branch); `src/rl_trader/agents/trainer.py` (`continue_from`)
2. **Why:** Loading prior weights and calling `learn` without `EvalCallback` saved last weights, not validation-selected.
3. **Severity:** high
4. **Correction:** Fine-tune goes through `train_agent(..., continue_from=prev_fold)` with EvalCallback; refuses missing prior weights.
5. **Regression:** covered by trainer continue_from path + existing val-selection tests; WF expanding purge test ensures geometry.

### H6 — Survivorship / ignored delistings

1. **Where:** `DataSettings` universe; `src/rl_trader/data/validation.py`
2. **Why:** Static ticker lists omit delisted names; series ending early was not flagged.
3. **Severity:** high
4. **Correction:** `point_in_time_universe` / `survivorship_bias_acknowledged` flags; `assert_survivorship_policy` in download + LIVE settings; delisting/early-end warning in `validate_ohlcv_frame`.
5. **Regression:** `test_survivorship_policy_requires_acknowledgement`, `test_delisting_warning_when_series_ends_early`

---

## Medium / low (fixed or documented)

| ID | Issue | Severity | Disposition |
| --- | --- | --- | --- |
| M1 | Benchmark harness ignored execution delay | medium | Fixed — same delay queue as env |
| M2 | Yahoo Adj Close retroactive revisions / cache vintage | medium | Documented residual; `auto_adjust=False` + schema kept; re-download for vintage control |
| M3 | Volume not split-adjusted vs adj prices | medium | Documented; ATR/OHLC adj path fixed; volume z-score remains on raw volume (common) |
| M4 | Incomplete cost field sync backtest↔env | medium | Runner syncs commission/slippage/cash; remaining env fields come from `environment` when `use_environment_costs` |
| L1 | Juneteenth calendar edge cases pre-2021 | low | Documented; calendar module unchanged this pass |
| L2 | Unstable seeds if callers omit `set_global_seeds` | low | Train/compare/WF CLIs seed; SB3 `seed=` set in registry |
| L3 | No Optuna / large HPO in repo | low | **OK** — confirmed absent |

---

## Paper-trading status

| Item | Status |
| --- | --- |
| `PaperBroker` default | **Enabled** with cost model + persistent ledger |
| `scripts/run_paper_trader.py` | Connects paper ledger; `--demo` runs a local cycle |
| Live broker | **Unimplemented stub** (still gated; no unsupervised live) |
| Live broker import from research | Still blocked by import boundaries |
| Proceed to live trading? | **No** — paper sim only; live adapter not implemented |

---

## Reviewed and found OK (brief)

- Causal rolling helpers / feature leakage mutation tests (`test_feature_leakage.py`)
- Train-only `FeatureScaler.fit(split="train")` guard
- Ordered train/val/test date masks; WF `step_bars >= test_bars`; non-overlapping test concat refusal
- Compare-agents: val-only candidate selection; `winner_declared=False`; random-policy control
- Checkpoint feature-match on load (`validation/features.py`)
- `yfinance` `auto_adjust=False` with explicit adj columns in schema/cleaning
- Fundamentals `publication_date <= as_of` filter in `fundamentals/asof.py` (plus next-day feature lag)
- No Optuna / large HPO
- Manifests store algorithm hyperparameters; WF fine-tune cannot load a *future* fold checkpoint (only `prev_checkpoint`)

---

## Residual research limitations (not silent bugs)

- Adjusted-close vendor revisions remain a data-vintage risk; pin cache downloads for reproducible studies.
- True point-in-time share/float universes are not supplied — research configs must keep `survivorship_bias_acknowledged: true` or provide PIT membership.
- Live cost-accurate execution is still unimplemented (intentionally stubbed / gated). PaperBroker applies its own configurable cost model.

---

## How to re-run guards

```bash
make test lint typecheck
pytest -q tests/unit/test_overfitting_audit_guards.py
```
