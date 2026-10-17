# Paper-trading deploy image — live trading remains disabled.
# Build: docker build -t rl-trader:paper .
# Run via docker-compose (recommended) or:
#   docker run --rm --env-file .env -v "$(pwd)/reports:/app/reports" rl-trader:paper

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_MODE=paper \
    LIVE_TRADING_ENABLED=false \
    BROKER_ENVIRONMENT=paper \
    PYTHONPATH=/app/src

WORKDIR /app

# System deps for scientific stack wheels; keep image lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY models/approved ./models/approved

RUN pip install --upgrade pip \
 && pip install -e ".[dev]" \
 && chmod +x scripts/*.sh scripts/*.py

# Persistent mounts expected at runtime:
#   /app/data, /app/reports, /app/models/checkpoints, /app/logs
RUN mkdir -p data/raw data/interim data/processed reports/paper_trading \
      models/checkpoints logs backups

# Default: one demo dry-run cycle (override in compose for --loop).
CMD ["python", "scripts/run_paper_trader.py", "--config", "configs/paper.yaml", "--demo", "--dry-run"]
