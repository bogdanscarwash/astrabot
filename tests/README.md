# Astrabot Testing Guide

## Setup

### With pyenv (Recommended)

1. Ensure pyenv is installed and configured (see `docs/pyenv-setup.md`)
2. Run the setup script:
   ```bash
   bash scripts/setup-environment.sh
   ```

### Manual Setup

```bash
# Set Python version with pyenv
pyenv local 3.11.9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock
```

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
./test_runner.py all

# Run unit tests only
./test_runner.py unit

# Run with coverage
./test_runner.py coverage

# Run tests for specific module
./test_runner.py schemas  # Test structured_schemas
./test_runner.py utils    # Test conversation_utilities
```

### Using Make

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test file
make test-file
# Then enter: test_conversation_utilities.py
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_structured_schemas.py

# Run specific test
pytest -k test_extract_tweet_text

# Run with coverage
pytest --cov=. --cov-report=html
```

## Test Structure

```
tests/
├── __init__.py
├── test_conversation_utilities.py  # Tests for tweet/image extraction
├── test_structured_schemas.py      # Tests for data models
└── README.md                       # This file
```

## Writing Tests

### Test File Naming
- Test files must start with `test_`
- Test classes must start with `Test`
- Test methods must start with `test_`

### Example Test

```python
import unittest
from conversation_utilities import extract_tweet_text

class TestTweetExtraction(unittest.TestCase):
    def test_extract_tweet_text(self):
        url = "https://twitter.com/user/status/123"
        result = extract_tweet_text(url)
        self.assertIsNotNone(result)
```

### Using Markers

```python
import pytest

@pytest.mark.unit
def test_simple_function():
    assert 1 + 1 == 2

@pytest.mark.integration
@pytest.mark.requires_api
def test_api_call():
    # This test requires API keys
    pass
```

## Mocking External Services

Tests use mocks to avoid hitting external APIs:

```python
from unittest.mock import patch, MagicMock

@patch('conversation_utilities.requests.get')
def test_with_mock(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'<html>...</html>'
    mock_get.return_value = mock_response
    
    # Your test code here
```

## Coverage Reports

After running tests with coverage:

```bash
# Generate HTML report
pytest --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Run tests with XML output for CI
pytest --junitxml=test-results.xml

# With coverage for CI
pytest --cov=. --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. You're in the project root directory
2. The virtual environment is activated
3. All dependencies are installed

### API-Related Tests Failing

Some tests require API keys. Set them as environment variables:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

Or skip API tests:
```bash
pytest -m "not requires_api"
```