.DEFAULT_GOAL := help
SHELL := /bin/bash
ENV ?= dev

# Load .env if present (never fails if missing)
-include .env
export

.PHONY: help setup install lint test test-cov clean \
        gcp-setup ingest baseline notebook \
        docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Local Setup ─────────────────────────────────────────────────────────────
install: ## Install Python dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

setup: install ## Full local dev setup: install + pre-commit hooks
	pre-commit install
	@echo "Setup complete. Copy .env.example to .env and fill in your values."

# ─── Code Quality ────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format: ## Auto-format code
	ruff format src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

typecheck: ## Run mypy type checker
	mypy src/ --ignore-missing-imports

# ─── Tests ────────────────────────────────────────────────────────────────────
test: ## Run unit tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80
	@echo "Coverage report: htmlcov/index.html"

# ─── GCP ─────────────────────────────────────────────────────────────────────
gcp-setup: ## Bootstrap GCP project (idempotent)
	@test -n "$$GCP_PROJECT_ID" || (echo "ERROR: GCP_PROJECT_ID not set"; exit 1)
	bash scripts/01_gcp_setup.sh

ingest: ## Ingest IEEE-CIS data from GCS into BigQuery
	@test -n "$$GCP_PROJECT_ID" || (echo "ERROR: GCP_PROJECT_ID not set"; exit 1)
	ENV=$(ENV) python scripts/02_data_ingestion.py

# ─── Training ────────────────────────────────────────────────────────────────
baseline: ## Train and evaluate the Logistic Regression baseline
	@test -n "$$GCP_PROJECT_ID" || (echo "ERROR: GCP_PROJECT_ID not set"; exit 1)
	ENV=$(ENV) python -m src.training.baseline

# ─── Notebooks ───────────────────────────────────────────────────────────────
notebook: ## Start Jupyter Lab
	jupyter lab notebooks/

# ─── Docker ──────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image locally
	docker build \
	  --tag fraud-detection:dev \
	  --build-arg BUILDKIT_INLINE_CACHE=1 \
	  .

docker-run: ## Run container locally (requires .env)
	docker run --rm -it \
	  --env-file .env \
	  -p 8080:8080 \
	  fraud-detection:dev

# ─── Cleanup ─────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Clean complete."
