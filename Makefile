.PHONY: install test lint typecheck run-smoke download-data build-features run-benchmarks train-agent train-smoke compare-agents walk-forward walk-forward-smoke run-paper run-paper-dry migrate db-upgrade db-current db-export dashboard

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
CONFIG ?= configs/research.yaml
PAPER_CONFIG ?= configs/paper.yaml
ALGORITHM ?= ppo
DATABASE_URL ?= sqlite:///./reports/rl_trader.db
EXPORT_DIR ?= reports/db_exports

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests scripts

typecheck:
	$(VENV)/bin/mypy src

run-smoke:
	$(PY) -m rl_trader.cli

download-data:
	$(PY) scripts/download_data.py --config $(CONFIG)

build-features:
	$(PY) scripts/build_features.py --config $(CONFIG)

run-benchmarks:
	$(PY) scripts/run_benchmarks.py --config $(CONFIG)

train-agent:
	$(PY) scripts/train_agent.py --config $(CONFIG) --algorithm $(ALGORITHM)

train-smoke:
	$(PY) scripts/train_agent.py --config $(CONFIG) --algorithm $(ALGORITHM) --fixture

compare-agents:
	$(PY) scripts/compare_agents.py --config $(CONFIG) --fixture

walk-forward:
	$(PY) scripts/run_walk_forward.py --config $(CONFIG)

walk-forward-smoke:
	$(PY) scripts/run_walk_forward.py --config $(CONFIG) --fixture --no-train

run-paper:
	$(PY) scripts/run_paper_trader.py --config $(PAPER_CONFIG) --demo

run-paper-dry:
	$(PY) scripts/run_paper_trader.py --config $(PAPER_CONFIG) --demo --dry-run

migrate db-upgrade:
	DATABASE_URL=$(DATABASE_URL) $(PY) -m alembic upgrade head

db-current:
	DATABASE_URL=$(DATABASE_URL) $(PY) -m alembic current

db-export:
	DATABASE_URL=$(DATABASE_URL) $(PY) scripts/export_db_csv.py --output-dir $(EXPORT_DIR)

dashboard:
	$(PY) scripts/run_dashboard.py --host 127.0.0.1 --port 8505

# --- Paper deploy helpers (never enable live) ---
paper-health:
	$(PY) scripts/healthcheck_paper.py --heartbeat reports/paper_trading/heartbeat.json --allow-stopped

backup-paper:
	bash scripts/backup_paper.sh backups

restore-paper:
	@test -n "$(SRC)" || (echo "usage: make restore-paper SRC=backups/paper_..." && exit 2)
	bash scripts/restore_paper.sh $(SRC)

docker-build:
	docker compose build

docker-up:
	docker compose up -d paper-trader

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f paper-trader
