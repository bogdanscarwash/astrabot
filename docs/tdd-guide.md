# Test-Driven Development Guide for Astrabot

This guide documents the TDD approach used in this project, particularly demonstrated through the creation of the centralized logging module.

## TDD Workflow Example: Logging Module

### Step 1: Requirements Gathering

Before writing any code, we identified the requirements:
- Centralized logging configuration
- Automatic sensitive data masking
- Performance tracking
- Thread-safe operations
- Structured logging support
- File rotation
- Context management

### Step 2: Test Creation (Red Phase)

Created `tests/test_utils_logging.py` with comprehensive test coverage:

```python
class TestAstrabotLogger(unittest.TestCase):
    def test_singleton_pattern(self):
        """Test that logger follows singleton pattern"""
        logger1 = get_logger('test1')
        logger2 = get_logger('test2')
        self.assertIs(logger1, logger2)
    
    def test_sensitive_data_masking(self):
        """Test that sensitive data is automatically masked"""
        # Test implementation before the feature exists
```

Key testing principles:
- Each test has a single clear purpose
- Test names describe what they verify
- Tests are independent and isolated
- Fixtures handle setup/teardown

### Step 3: Initial Implementation (Green Phase)

Created `utils/logging.py` with minimal code to pass tests:

```python
def get_logger(name: Optional[str] = None) -> AstrabotLogger:
    """Get the singleton logger instance."""
    global _logger_instance
    
    if _logger_instance is None:
        with _logger_lock:
            if _logger_instance is None:
                # Create logger
                logging.setLoggerClass(AstrabotLogger)
                _logger_instance = logging.getLogger('astrabot')
```

### Step 4: Refactoring (Refactor Phase)

Once tests passed, improved the implementation:
- Added thread safety with locks
- Enhanced sensitive data patterns
- Improved performance
- Added comprehensive documentation

## TDD Best Practices

### 1. Test Organization

```
tests/
├── __init__.py
├── test_utils_logging.py      # Unit tests for logging
├── test_conversation_utilities.py  # Unit tests for conversations
└── test_structured_schemas.py     # Unit tests for data models
```

### 2. Test Categories

#### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution
- Example: `test_mask_sensitive_data()`

#### Integration Tests
- Test component interactions
- Use real dependencies where possible
- Example: `test_logging_in_conversation_context()`

#### Security Tests
- Verify sensitive data handling
- Test access controls
- Example: `test_sensitive_data_masking()`

### 3. Mocking Strategy

```python
def test_log_api_call_decorator(self):
    """Test API call logging decorator"""
    @log_api_call("OpenAI")
    def make_api_call(url, api_key=None):
        return {"status": "success"}
    
    with patch('utils.logging.get_logger') as mock_logger:
        result = make_api_call("https://api.openai.com/v1/test", api_key="sk-secret")
        
        # Verify behavior without making real API calls
        mock_logger.return_value.info.assert_called()
```

### 4. Test Fixtures

```python
def setUp(self):
    """Set up test fixtures"""
    self.temp_dir = tempfile.mkdtemp()
    self.log_file = os.path.join(self.temp_dir, 'test.log')
    
def tearDown(self):
    """Clean up after tests"""
    # Reset logging state
    logger = stdlib_logging.getLogger('astrabot')
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
```

## Benefits Realized

### 1. Design Clarity
- Tests forced clear API design
- Edge cases identified early
- Security requirements built-in

### 2. Confidence
- Changes can be made safely
- Refactoring is low-risk
- New features don't break existing ones

### 3. Documentation
- Tests serve as usage examples
- Expected behavior is codified
- Easy onboarding for new developers

### 4. Quality
- Bugs caught during development
- Performance issues identified
- Thread safety verified

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_utils_logging.py -v

# Run with coverage
python -m pytest tests/ --cov=utils --cov-report=html

# Run specific test
python -m pytest tests/test_utils_logging.py::TestAstrabotLogger::test_singleton_pattern -v
```

## Continuous TDD

1. **New Feature Request**: Start with tests
2. **Bug Report**: Write failing test first
3. **Performance Issue**: Add performance test
4. **Security Concern**: Add security test

Remember: If it's not tested, it's broken!