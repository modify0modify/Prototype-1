.PHONY: install test lint typecheck run-smoke

PYTHON ?= python3.12
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

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
