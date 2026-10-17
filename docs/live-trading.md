# Live trading (disabled by default)

This project ships a gated live sandbox adapter. **Live trading stays inactive**
unless every safety layer is configured, the process is manually armed, and the
rollout stage permits orders. Stage 0 (the default) never submits live orders.

See also the README section “Live trading (disabled by default)” and
`configs/live.yaml` (EXAMPLE thresholds labelled as examples).

## Adapter layout

| Path | Role |
|------|------|
| `src/rl_trader/brokers/live.py` | `LiveBroker` — `BrokerAdapter` sandbox/stub |
| `src/rl_trader/live_trading/stages.py` | Stage 0–3 policy |
| `src/rl_trader/live_trading/activation.py` | Manual activate / deactivate |
| `src/rl_trader/live_trading/capital.py` | Capital guard (no auto-increase) |
| `src/rl_trader/live_trading/routing.py` | Paper vs live order separation |
| `src/rl_trader/live_trading/safety.py` | Pre-submit safety stack |
| `src/rl_trader/live_trading/isolation.py` | Agent control isolation |
| `src/rl_trader/execution/live.py` | Gated live executor |
| `src/rl_trader/config/live_gates.py` | Fail-fast activation gates |

## Stage machine

- **0 — shadow:** record decisions only; reject live fills
- **1 — tiny long-only:** EXAMPLE `stage1_capital_limit`; long-only
- **2 — reviewed capital:** requires `stage2_review_acknowledged` (human/config)
- **3 — expanded universe:** requires stage-3 review + `stage3_expanded_universe`

Capital increases are **never** applied by the running system.

## Paper / live routing

`OrderRouter` binds a `mode="paper"` adapter and a `mode="live"` adapter.
Paper submits never invoke the live adapter (`assert_paper_never_reached_live`).
LiveBroker ignores paper credential fields.
