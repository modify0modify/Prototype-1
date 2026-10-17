# Paper deploy smoke test

**Branch:** `cursor/paper-broker-ef36`  
**When:** 2026-10-17  
**Command:** `make run-paper-dry`  
(`python scripts/run_paper_trader.py --config configs/paper.yaml --demo --dry-run`)

## Result

| Check | Outcome |
|-------|---------|
| Exit code | **0** (clean) |
| Live trading | Disabled (`app_mode=paper`, dry-run demo) |
| Cycle | `decision=resized`, `orders=[]`, `dry_run=True` |
| Heartbeat | Written (`status=stopped` after graceful `service.stop()`) |
| Process lock | Acquired for cycle; released on stop |
| Healthcheck | `healthcheck_paper.py --allow-stopped` → **OK** |

## Log excerpt

```
demo mode: provisioned stub approved model at reports/paper_trading/demo_model/manifest.json
paper service ready model_id=paper-demo-stub ... dry_run=True loop=False
dry-run: decision=resized — no simulated orders created
cycle complete ... orders=[] dry_run=True
stopping paper service (release lock, flush broker)
```

No live broker paths exercised.
