# Astrabot Test Suite

This directory contains the comprehensive test suite for Astrabot, following Test-Driven Development (TDD) principles and best practices.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Shared pytest fixtures and configuration
├── fixtures/                    # Test data and mock responses
│   ├── sample_signal_data.csv   # Sample Signal backup data
│   ├── sample_recipients.csv    # Sample Signal recipients
│   └── sample_tweet_responses.json # Mock Twitter/X responses
├── integration/                 # Integration tests
│   └── test_conversation_pipeline.py # End-to-end pipeline tests
└── unit/                       # Unit tests
    ├── test_conversation_analyzer.py   # Conversation analysis tests
    ├── test_conversation_processor.py  # Core processing tests
    ├── test_conversation_schemas.py    # Schema validation tests
    ├── test_metadata_enricher.py      # Metadata enrichment tests
    ├── test_qa_extractor.py           # Q&A extraction tests
    ├── test_style_analyzer.py         # Communication style tests
    ├── test_twitter_extractor.py      # Twitter/X extraction tests
    ├── test_config.py                 # Configuration tests
    ├── test_training_data_creator.py  # Training data creation tests
    ├── test_adaptive_trainer.py       # Adaptive training tests
    └── test_utils_logging.py          # Logging utility tests
```

## Running Tests

### All Tests
```bash
make test
```

### Unit Tests Only
```bash
make test-unit
```

### Integration Tests (requires API keys)
```bash
make test-integration
```

### Coverage Report
```bash
make test-coverage
```

### Specific Test File
```bash
make test-file
# Enter: tests/unit/test_conversation_processor.py
```

### Specific Test Function
```bash
make test-one
# Enter: test_extract_tweet_content
```

### Quick Tests (no coverage)
```bash
make test-quick
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.unit` - Fast unit tests with no external dependencies
- `@pytest.mark.integration` - Integration tests requiring external services
- `@pytest.mark.twitter` - Tests related to Twitter/X extraction
- `@pytest.mark.llm` - Tests related to LLM training functionality
- `@pytest.mark.requires_api` - Tests requiring API keys
- `@pytest.mark.requires_signal_data` - Tests requiring Signal backup data
- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.smoke` - Quick smoke tests for CI/CD

### Running Specific Marker Groups
```bash
# Run only unit tests
pytest -m unit

# Run only Twitter-related tests
pytest -m twitter

# Run tests that don't require API keys
pytest -m "not requires_api"

# Run smoke tests for CI
pytest -m smoke
```

## Test Configuration

### pytest.ini
- Configures test discovery patterns
- Sets coverage thresholds (80% minimum)
- Defines test markers
- Configures logging for tests

### .flake8
- Linting configuration for test files
- Excludes test-specific patterns
- Allows test-specific imports

### conftest.py
- Shared fixtures across all tests
- Mock configurations for external services
- Sample data generators
- Environment variable management

## Test Data and Fixtures

### Sample Data
- `sample_signal_data.csv` - Realistic Signal message data
- `sample_recipients.csv` - Sample Signal contact data
- `sample_tweet_responses.json` - Mock Twitter API responses

### Shared Fixtures
- `sample_messages_df` - Pandas DataFrame with Signal messages
- `sample_recipients_df` - Pandas DataFrame with Signal recipients
- `sample_tweet_content` - Mock tweet content for testing
- `mock_openai_client` - Mocked OpenAI API client
- `mock_anthropic_client` - Mocked Anthropic API client
- `temp_data_dir` - Temporary directory for test files

## Environment Variables for Testing

Create `.env.test` with test-specific values:

```bash
TESTING=true
LOG_LEVEL=DEBUG
YOUR_RECIPIENT_ID=2
OPENAI_API_KEY=test_key_for_mocking
ANTHROPIC_API_KEY=test_key_for_mocking
```

## Writing New Tests

### Unit Test Template

```python
import pytest
from unittest.mock import Mock, patch

from src.module.to_test import ClassToTest

@pytest.mark.unit
class TestClassToTest:
    """Test class functionality"""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing"""
        return ClassToTest()
    
    def test_basic_functionality(self, instance):
        """Test basic functionality"""
        result = instance.method_to_test()
        assert result is not None
    
    def test_edge_cases(self, instance):
        """Test edge cases and error handling"""
        with pytest.raises(ValueError):
            instance.method_with_invalid_input(None)
    
    @patch('src.module.to_test.external_dependency')
    def test_with_mocks(self, mock_dependency, instance):
        """Test with mocked dependencies"""
        mock_dependency.return_value = "mocked_response"
        result = instance.method_using_dependency()
        assert result == "expected_result"
```

### Integration Test Template

```python
import pytest
from pathlib import Path

@pytest.mark.integration
@pytest.mark.requires_api
class TestIntegrationScenario:
    """Test integration scenarios"""
    
    def test_end_to_end_workflow(self, temp_data_dir):
        """Test complete workflow"""
        # Setup test data
        # Run integration
        # Verify results
        pass
```

## Test Coverage

Current coverage targets:
- **Minimum**: 80% overall coverage
- **Unit Tests**: >90% coverage for core modules
- **Integration Tests**: Cover main user workflows

### Coverage Exclusions
- Test files themselves
- `__init__.py` files
- Development scripts
- Notebooks and documentation

## Continuous Integration

Tests are automatically run on:
- Pull requests
- Main branch commits
- Nightly builds (with external API tests)

### CI Test Stages
1. **Lint and Format** - Code quality checks
2. **Unit Tests** - Fast tests without external dependencies
3. **Integration Tests** - Full pipeline tests (if API keys available)
4. **Coverage Report** - Ensure coverage thresholds are met

## Best Practices

### Test Organization
- One test file per source module
- Group related tests in classes
- Use descriptive test names
- Include docstrings for complex tests

### Test Data
- Use fixtures for reusable test data
- Keep test data minimal but realistic
- Mock external services consistently

### Assertions
- Use specific assertions (not just `assert True`)
- Test both positive and negative cases
- Verify complete object state when needed

### Performance
- Keep unit tests fast (< 1 second each)
- Use `@pytest.mark.slow` for longer tests
- Mock time-consuming operations

### Documentation
- Document test purpose in docstrings
- Explain complex test setups
- Include examples of expected behavior

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure `conftest.py` is properly configured
- Check `PYTHONPATH` includes project root
- Verify module structure matches imports

**Fixture Not Found**
- Check fixture is defined in `conftest.py` or test file
- Verify fixture scope (function, class, module, session)
- Ensure fixture name matches parameter name

**Mock Not Working**
- Check patch target path is correct
- Verify mock is applied before function call
- Use `MagicMock` for complex objects

**Tests Failing in CI**
- Check environment variables are set
- Verify external dependencies are mocked
- Review CI-specific test markers

### Getting Help

1. Check test documentation in docstrings
2. Review similar test patterns in codebase
3. Consult pytest documentation
4. Ask in project discussions/issues

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Test edge cases** and error conditions
3. **Mock external dependencies** appropriately
4. **Update fixtures** if needed for new test data
5. **Add appropriate markers** for test categorization
6. **Ensure coverage** meets project standards

See [CONTRIBUTING.md](../CONTRIBUTING.md) for more details on the development process.