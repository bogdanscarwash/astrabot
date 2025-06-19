# Makefile for running tests and other development tasks

.PHONY: test test-unit test-integration test-coverage clean install lint format

# Install dependencies
install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock

# Run all tests
test:
	pytest

# Run only unit tests
test-unit:
	pytest -m unit

# Run integration tests (requires API keys)
test-integration:
	pytest -m integration

# Run tests with coverage
test-coverage:
	pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
test-file:
	@read -p "Enter test file name (e.g., test_conversation_utilities.py): " file; \
	pytest tests/$$file -v

# Clean up generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Run linting
lint:
	flake8 . --exclude=venv,.venv,__pycache__
	mypy conversation_utilities.py structured_schemas.py --ignore-missing-imports

# Format code
format:
	black conversation_utilities.py structured_schemas.py tests/

# Run a specific test
test-one:
	@read -p "Enter test name (e.g., test_extract_tweet_text): " test; \
	pytest -k $$test -v

# Quick test (no coverage, less verbose)
test-quick:
	pytest -q

# Watch tests (requires pytest-watch)
test-watch:
	pip install pytest-watch
	ptw