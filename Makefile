.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check clean build build-install docs uv-install uv-install-dev uv-sync uv-run

# Detect if uv is available
UV := $(shell which uv 2>/dev/null)

ifdef UV
    # Use uv if available (modern workflow)
    PYTHON := .venv/bin/python
    PIP_CMD := $(UV) pip
    PYTEST := .venv/bin/python -m pytest
    RUFF := .venv/bin/python -m ruff
    MYPY := .venv/bin/python -m mypy
else
    # Fall back to pip (legacy workflow)
    PYTHON := python3
    PIP_CMD := $(PYTHON) -m pip
    PYTEST := $(PYTHON) -m pytest
    RUFF := $(PYTHON) -m ruff
    MYPY := $(PYTHON) -m mypy
endif

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
ifdef UV
	@echo ""
	@echo "UV detected - using modern uv-based workflow (recommended)"
else
	@echo ""
	@echo "UV not detected - using legacy pip-based workflow"
	@echo "Install uv from https://docs.astral.sh/uv/ for faster builds"
endif

install: ## Install the package (pip fallback)
	$(PIP_CMD) install -e .

install-dev: ## Install development dependencies (pip fallback)
	$(PIP_CMD) install -e ".[dev]"

install-build: ## Install build dependencies (pip fallback)
	$(PIP_CMD) install -e ".[build]"

# UV-specific targets (modern workflow)
uv-install: ## Create uv venv and install the package
	uv venv
	uv pip install -e ".[dev]"

uv-install-dev: ## Create uv venv and install all dev dependencies
	uv venv
	uv pip install -e ".[dev,build]"

uv-sync: ## Sync uv environment with locked dependencies
	uv sync

uv-run: ## Run the application using uv
	uv run link-models

test: ## Run all tests
	$(PYTEST) -v

test-unit: ## Run unit tests only
	$(PYTEST) -v -m unit

test-integration: ## Run integration tests only
	$(PYTEST) -v -m integration

test-cov: ## Run tests with coverage
	$(PYTEST) -v --cov=link_models --cov-report=html --cov-report=term

lint: ## Run linter (ruff)
	$(RUFF) check src tests

lint-fix: ## Run linter with auto-fix
	$(RUFF) check --fix src tests

format: ## Format code with ruff
	$(RUFF) format src tests

format-check: ## Check code formatting
	$(RUFF) format --check src tests

type-check: ## Run type checker (mypy)
	$(MYPY) src

check: lint type-check test ## Run all checks (lint, type-check, test)

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build standalone executable (PyInstaller)
	$(PYTHON) scripts/build.py --backend pyinstaller --clean

build-nuitka: clean ## Build standalone executable (Nuitka)
	$(PYTHON) scripts/build.py --backend nuitka --clean

build-installer: clean ## Build executable with installer
	$(PYTHON) scripts/build.py --backend pyinstaller --installer --clean

run: ## Run the application in watch mode (for development)
	$(PYTHON) -m link_models --verbose watch

sync: ## Run one-time sync (for development)
	$(PYTHON) -m link_models --verbose sync

generate-config: ## Generate default configuration file
	$(PYTHON) -m link_models config --generate

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"
