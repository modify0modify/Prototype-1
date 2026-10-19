# Training fixtures (offline)

Unit/integration tests under `tests/unit/test_episode_loader.py` and
`tests/integration/test_real_data_training.py` synthesize **labelled** multi-symbol
OHLCV parquet panels in a temporary `data/processed/ohlcv/` tree (no network).

They prove:

1. `load_episode_from_processed` builds `EpisodeData` from cleaned bars.
2. `scripts/train_agent.py` refuses missing processed data (exit 2).
3. Feature scalers still fit on **train** only when the real loader path is used.
