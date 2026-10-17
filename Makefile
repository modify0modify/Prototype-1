.PHONY: install test lint typecheck run-smoke download-data build-features

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
CONFIG ?= configs/research.yaml

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
