.DEFAULT_GOAL := help
SHELL := /bin/bash

UV ?= uv
PY ?= $(UV) run python
PKG := src/uplift_bench

.PHONY: help install install-dev install-bench install-all sync lock \
        lint format typecheck test test-fast cov pre-commit clean docs serve-docs

help:
	@echo "uplift-bench — common targets"
	@echo ""
	@echo "  make install        install runtime deps (uv sync, no extras)"
	@echo "  make install-dev    install dev tooling (pytest, ruff, mypy, ...)"
	@echo "  make install-bench  install heavy ML deps (catboost, lightgbm, mlflow, hydra)"
	@echo "  make install-all    install everything"
	@echo "  make sync           uv sync from current lock file"
	@echo "  make lock           regenerate uv.lock"
	@echo ""
	@echo "  make lint           ruff check"
	@echo "  make format         ruff format"
	@echo "  make typecheck      mypy --strict on src + tests"
	@echo "  make test           pytest with coverage gate"
	@echo "  make test-fast      pytest skipping slow / data-needing tests"
	@echo "  make cov            open the html coverage report"
	@echo ""
	@echo "  make docs           mkdocs build (strict)"
	@echo "  make serve-docs     mkdocs serve on :8000"
	@echo ""
	@echo "  make clean          wipe caches"

install:
	$(UV) sync --no-dev

install-dev:
	$(UV) sync --extra dev

install-bench:
	$(UV) sync --extra bench --extra dev

install-all:
	$(UV) sync --all-extras

sync:
	$(UV) sync

lock:
	$(UV) lock

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=90

test-fast:
	$(UV) run pytest -m "not slow and not needs_data" --cov --cov-report=term-missing

cov:
	$(UV) run pytest --cov --cov-report=html
	@echo "open htmlcov/index.html"

pre-commit:
	$(UV) run pre-commit run --all-files

docs:
	$(UV) run mkdocs build --strict

serve-docs:
	$(UV) run mkdocs serve -a 127.0.0.1:8000

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
