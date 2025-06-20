# Justfile for Astrabot development tasks
# Modern replacement for Makefile with all existing functionality

# Default recipe shows help
default:
    @just --list

# Help command - shows all available tasks
help:
    @echo "Astrabot Development Commands:"
    @echo ""
    @echo "Setup:"
    @echo "  just install         Install production dependencies"
    @echo "  just install-dev     Install all dependencies including dev tools"
    @echo "  just setup-env       Set up development environment (interactive)"
    @echo ""
    @echo "Testing:"
    @echo "  just test           Run all tests"
    @echo "  just test-unit      Run only unit tests"
    @echo "  just test-integration Run integration tests (requires API keys)"
    @echo "  just test-privacy   Run privacy filter tests"
    @echo "  just test-performance Run performance tests"
    @echo "  just test-privacy-full Run comprehensive privacy tests"
    @echo "  just test-coverage  Run tests with coverage report"
    @echo "  just test-file      Run specific test file interactively"
    @echo "  just test-one       Run specific test by name"
    @echo "  just test-quick     Quick test run without coverage"
    @echo "  just test-watch     Watch tests and rerun on changes"
    @echo ""
    @echo "Code Quality:"
    @echo "  just lint           Run flake8 linting"
    @echo "  just format         Format code with black and isort"
    @echo "  just type-check     Run mypy type checking"
    @echo "  just all            Run format, lint, and type-check"
    @echo "  just pre-commit     Run pre-commit hooks"
    @echo ""
    @echo "Docker & Data:"
    @echo "  just docker-build   Build Signal backup tools Docker image"
    @echo "  just process-signal Process Signal backup data"
    @echo ""
    @echo "Training:"
    @echo "  just train          Run training pipeline"
    @echo "  just train-qwen3    Train Qwen3 model with full pipeline"
    @echo "  just train-qwen3-small  Train small Qwen3 model (3B) for testing"
    @echo "  just train-qwen3-multistage  Train with multi-stage approach"
    @echo "  just train-qwen3-test  Test training with debug output"
    @echo ""
    @echo "UV Package Management:"
    @echo "  just uv-sync        Sync dependencies with uv"
    @echo "  just uv-add PKG     Add a new dependency"
    @echo "  just uv-remove PKG  Remove a dependency"
    @echo "  just uv-update      Update all dependencies"
    @echo ""
    @echo "Maintenance:"
    @echo "  just clean          Clean up generated files"
    @echo "  just notebook       Run Jupyter notebook server"
    @echo "  just docs           Generate documentation"

# Install production dependencies using uv
install:
    uv sync --no-dev

# Install all dependencies including dev tools using uv
install-dev:
    uv sync
    pre-commit install

# UV package management commands
uv-sync:
    uv sync

uv-add PKG:
    uv add {{PKG}}

uv-remove PKG:
    uv remove {{PKG}}

uv-update:
    uv sync --upgrade

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

# Run privacy filter tests
test-privacy:
    @echo "Running privacy filter tests..."
    pytest -m privacy -v

# Run performance tests
test-performance:
    @echo "Running performance tests..."
    pytest -m slow -v

# Run all privacy-related tests (unit + integration)
test-privacy-full:
    @echo "Running comprehensive privacy tests..."
    pytest -m "privacy or (unit and privacy)" -v

# Run tests with coverage
test-coverage:
    pytest --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file interactively
test-file:
    #!/usr/bin/env bash
    read -p "Enter test file path (e.g., tests/unit/test_conversation_processor.py): " file
    pytest "$file" -v

# Run a specific test by name interactively
test-one:
    #!/usr/bin/env bash
    read -p "Enter test name (e.g., test_extract_tweet_text): " test
    pytest -k "$test" -v

# Quick test (no coverage, less verbose)
test-quick:
    pytest -q --no-cov

# Watch tests (requires pytest-watch)
test-watch:
    @which ptw > /dev/null || uv add --dev pytest-watch
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

# Run pre-commit hooks
pre-commit:
    pre-commit run --all-files

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
    #!/usr/bin/env bash
    if [ ! -d "data/raw/signal-flatfiles" ]; then
        mkdir -p data/raw/signal-flatfiles
    fi
    echo "Processing Signal backup..."
    echo "Place your Signal backup file in the appropriate location"
    echo "Then run: docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output signalbackup-tools"
    python scripts/process_signal_data.py

# Run training pipeline
train:
    #!/usr/bin/env bash
    echo "Starting training pipeline..."
    if [ -f "configs/training_config.yaml" ]; then
        python scripts/train.py --config configs/training_config.yaml
    else
        python scripts/train.py
    fi

# Train with Qwen3 model
train-qwen3:
    @echo "Starting Qwen3 training pipeline..."
    python scripts/train_qwen3.py \
        --config configs/training_config.yaml \
        --messages data/raw/signal-flatfiles/signal.csv \
        --recipients data/raw/signal-flatfiles/recipient.csv \
        --output ./models/qwen3-personal

# Train Qwen3 small model (3B) for testing
train-qwen3-small:
    @echo "Training small Qwen3 model for testing..."
    python scripts/train_qwen3.py \
        --config configs/training_config.yaml \
        --messages data/raw/signal-flatfiles/signal.csv \
        --recipients data/raw/signal-flatfiles/recipient.csv \
        --output ./models/qwen3-small \
        --test

# Train Qwen3 with multi-stage approach
train-qwen3-multistage:
    @echo "Starting multi-stage Qwen3 training..."
    python scripts/train_qwen3.py \
        --config configs/training_config.yaml \
        --multi-stage \
        --output ./models/qwen3-multistage

# Test Qwen3 training with debug output
train-qwen3-test:
    @echo "Testing Qwen3 training pipeline..."
    python scripts/train_qwen3.py \
        --config configs/training_config.yaml \
        --messages data/raw/signal-flatfiles/signal.csv \
        --recipients data/raw/signal-flatfiles/recipient.csv \
        --output ./models/qwen3-test \
        --debug \
        --test

# Development environment setup
setup-env:
    @echo "Setting up development environment..."
    bash scripts/setup/bootstrap.sh
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

# Setup secrets interactively
setup-secrets:
    python scripts/setup/setup-secrets.py
