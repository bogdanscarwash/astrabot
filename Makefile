# Makefile for Astrabot development tasks

.PHONY: help install install-dev test test-unit test-integration test-coverage test-file test-one test-quick test-watch lint format type-check clean docker-build process-signal train all

# Default target
.DEFAULT_GOAL := help

# Help command
help:
	@echo "Astrabot Development Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install all dependencies including dev tools"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run only unit tests"
	@echo "  make test-integration Run integration tests (requires API keys)"
	@echo "  make test-coverage  Run tests with coverage report"
	@echo "  make test-file      Run specific test file interactively"
	@echo "  make test-one       Run specific test by name"
	@echo "  make test-quick     Quick test run without coverage"
	@echo "  make test-watch     Watch tests and rerun on changes"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           Run flake8 linting"
	@echo "  make format         Format code with black"
	@echo "  make type-check     Run mypy type checking"
	@echo "  make all            Run format, lint, and type-check"
	@echo ""
	@echo "Docker & Data:"
	@echo "  make docker-build   Build Signal backup tools Docker image"
	@echo "  make process-signal Process Signal backup data"
	@echo ""
	@echo "Training:"
	@echo "  make train          Run training pipeline"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean up generated files"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install all dependencies including dev tools
install-dev:
	pip install -e ".[dev,vision]"
	pre-commit install

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest -m unit

# Run integration tests (requires API keys)
test-integration:
	@echo "Running integration tests (requires API keys in .env)..."
	pytest -m integration

# Run tests with coverage
test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
test-file:
	@read -p "Enter test file path (e.g., tests/unit/test_conversation_processor.py): " file; \
	pytest $$file -v

# Run a specific test by name
test-one:
	@read -p "Enter test name (e.g., test_extract_tweet_text): " test; \
	pytest -k $$test -v

# Quick test (no coverage, less verbose)
test-quick:
	pytest -q --no-cov

# Watch tests (requires pytest-watch)
test-watch:
	@which ptw > /dev/null || pip install pytest-watch
	ptw -- -q --no-cov

# Run linting
lint:
	flake8 src/ tests/ scripts/
	@echo "✓ Linting passed"

# Format code
format:
	black src/ tests/ scripts/ notebooks/*.py
	isort src/ tests/ scripts/
	@echo "✓ Code formatted"

# Type checking
type-check:
	mypy src/ --ignore-missing-imports
	@echo "✓ Type checking passed"

# Run all code quality checks
all: format lint type-check
	@echo "✓ All checks passed"

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "✓ Cleaned up generated files"

# Docker operations
docker-build:
	@echo "Building Signal backup tools Docker image..."
	cd docker/signalbackup-tools && docker build -t signalbackup-tools .
	@echo "✓ Docker image built"

# Process Signal backup
process-signal:
	@if [ ! -d "data/raw/signal-flatfiles" ]; then \
		mkdir -p data/raw/signal-flatfiles; \
	fi
	@echo "Processing Signal backup..."
	@echo "Place your Signal backup file in the appropriate location"
	@echo "Then run: docker run -v /path/to/backup:/backup -v $$(pwd)/data/raw/signal-flatfiles:/output signalbackup-tools"
	python scripts/process_signal_data.py

# Run training pipeline
train:
	@echo "Starting training pipeline..."
	@if [ -f "configs/training_config.yaml" ]; then \
		python scripts/train.py --config configs/training_config.yaml; \
	else \
		python scripts/train.py; \
	fi

# Development environment setup
setup-env:
	@echo "Setting up development environment..."
	bash scripts/setup-environment.sh
	cp .env.example .env
	@echo "✓ Environment setup complete. Please edit .env with your API keys."

# Run notebook server
notebook:
	jupyter notebook notebooks/

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html
	@echo "✓ Documentation generated in docs/_build/html/"

# Run pre-commit hooks
pre-commit:
	pre-commit run --all-files