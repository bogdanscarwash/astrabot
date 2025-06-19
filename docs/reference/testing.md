# Testing Infrastructure Reference

This document provides comprehensive reference for the testing infrastructure in Astrabot, including API mocking fixtures, test utilities, and testing best practices.

## Overview

Astrabot follows a Test-Driven Development (TDD) approach with comprehensive test coverage across unit, integration, and privacy testing. The testing infrastructure includes mock API responses, sample data fixtures, and utilities for testing conversation analysis and AI training components.

## Test Configuration

### Pytest Setup

The testing framework uses pytest with custom markers and configuration:

```python
# pytest.ini configuration
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests  
    privacy: Privacy-related tests
    slow: Slow-running tests
```

### Test Session Management

The test session is configured with automatic cleanup and logging capture:

```python
@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up test session with proper logging and cleanup."""
    # Configure test logging
    # Set up test environment
    # Register cleanup handlers
```

## API Mocking Fixtures

### `mock_openai_api`

Mock OpenAI API responses for testing LLM interactions without making actual API calls.

```python
@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses for testing."""
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = {
            'choices': [{
                'message': {
                    'content': 'This is a mocked OpenAI response for testing purposes.'
                }
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 15,
                'total_tokens': 25
            }
        }
        yield mock_create
```

**Usage Example:**
```python
def test_openai_integration(mock_openai_api):
    """Test OpenAI API integration with mocked responses."""
    from src.llm.openai_client import OpenAIClient
    
    client = OpenAIClient()
    response = client.generate_response("Test prompt")
    
    # Verify mock was called
    mock_openai_api.assert_called_once()
    
    # Verify response content
    assert "mocked OpenAI response" in response
    
    # Check call arguments
    call_args = mock_openai_api.call_args
    assert call_args[1]['model'] == 'gpt-3.5-turbo'
    assert 'Test prompt' in str(call_args)
```

**Advanced Usage with Custom Responses:**
```python
def test_openai_custom_response(mock_openai_api):
    """Test with custom OpenAI response."""
    # Configure custom response
    mock_openai_api.return_value = {
        'choices': [{
            'message': {
                'content': 'Custom test response with specific content.'
            }
        }],
        'usage': {'total_tokens': 50}
    }
    
    client = OpenAIClient()
    response = client.generate_response("Custom prompt")
    
    assert "Custom test response" in response
```

### `mock_anthropic_api`

Mock Anthropic Claude API responses for testing alternative LLM integrations.

```python
@pytest.fixture  
def mock_anthropic_api():
    """Mock Anthropic API responses for testing."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='This is a mocked Anthropic response for testing.')]
        )
        
        yield mock_client
```

**Usage Example:**
```python
def test_anthropic_integration(mock_anthropic_api):
    """Test Anthropic API integration with mocked responses."""
    from src.llm.anthropic_client import AnthropicClient
    
    client = AnthropicClient()
    response = client.generate_response("Test prompt")
    
    # Verify mock was called
    mock_anthropic_api.messages.create.assert_called_once()
    
    # Verify response content
    assert "mocked Anthropic response" in response
    
    # Check call arguments
    call_args = mock_anthropic_api.messages.create.call_args
    assert call_args[1]['model'] == 'claude-3-sonnet-20240229'
    assert call_args[1]['max_tokens'] == 1000
```

**Error Handling Testing:**
```python
def test_anthropic_api_error(mock_anthropic_api):
    """Test Anthropic API error handling."""
    from anthropic import APIError
    
    # Configure mock to raise error
    mock_anthropic_api.messages.create.side_effect = APIError("Rate limit exceeded")
    
    client = AnthropicClient()
    
    with pytest.raises(APIError):
        client.generate_response("Test prompt")
```

## Sample Data Fixtures

### `sample_signal_messages`

Provides realistic Signal message data for testing conversation analysis.

```python
@pytest.fixture
def sample_signal_messages():
    """Sample Signal messages for testing."""
    return [
        {
            '_id': 1,
            'from_recipient_id': 2,
            'body': 'Hey! How are you doing? 😊',
            'date_sent': 1640995200000,
            'thread_id': 'thread_001'
        },
        {
            '_id': 2, 
            'from_recipient_id': 3,
            'body': 'I\'m doing great! Just finished a big project at work 🎉',
            'date_sent': 1640995260000,
            'thread_id': 'thread_001'
        },
        {
            '_id': 3,
            'from_recipient_id': 2,
            'body': 'That\'s awesome! What kind of project was it?',
            'date_sent': 1640995320000,
            'thread_id': 'thread_001'
        }
    ]
```

**Usage Example:**
```python
def test_conversation_analysis(sample_signal_messages):
    """Test conversation analysis with sample data."""
    from src.core.conversation_analyzer import ConversationAnalyzer
    
    analyzer = ConversationAnalyzer()
    analysis = analyzer.analyze_conversation_flow(sample_signal_messages)
    
    assert analysis['message_count'] == 3
    assert analysis['participant_count'] == 2
    assert 'thread_001' in analysis['threads']
```

### `sample_signal_recipients`

Provides recipient data for testing user identification and privacy filtering.

```python
@pytest.fixture
def sample_signal_recipients():
    """Sample Signal recipients for testing."""
    return [
        {
            'recipient_id': 2,
            'phone': '+1234567890',
            'email': 'user1@example.com',
            'profile_name': 'Alice Johnson',
            'profile_joined_name': 'Alice',
            'signal_profile_name': 'alice_signal'
        },
        {
            'recipient_id': 3,
            'phone': '+0987654321', 
            'email': 'user2@example.com',
            'profile_name': 'Bob Smith',
            'profile_joined_name': 'Bob',
            'signal_profile_name': 'bob_signal'
        }
    ]
```

### `mock_twitter_responses`

Mock Twitter API responses for testing URL extraction and content processing.

```python
@pytest.fixture
def mock_twitter_responses():
    """Mock Twitter API responses for testing."""
    return {
        'https://twitter.com/user/status/123': {
            'text': 'This is a sample tweet for testing purposes.',
            'author': 'test_user',
            'created_at': '2024-01-01T12:00:00Z',
            'metrics': {
                'like_count': 10,
                'retweet_count': 5,
                'reply_count': 2
            }
        },
        'https://x.com/user/status/456': {
            'text': 'Another sample tweet with different content.',
            'author': 'another_user', 
            'created_at': '2024-01-02T15:30:00Z',
            'metrics': {
                'like_count': 25,
                'retweet_count': 12,
                'reply_count': 8
            }
        }
    }
```

**Usage Example:**
```python
def test_twitter_extraction(mock_twitter_responses):
    """Test Twitter content extraction."""
    from src.extractors.twitter_extractor import TwitterExtractor
    
    with patch('src.extractors.twitter_extractor.requests.get') as mock_get:
        # Configure mock response
        mock_get.return_value.json.return_value = mock_twitter_responses['https://twitter.com/user/status/123']
        
        extractor = TwitterExtractor()
        content = extractor.extract_tweet_content('https://twitter.com/user/status/123')
        
        assert content['text'] == 'This is a sample tweet for testing purposes.'
        assert content['author'] == 'test_user'
        assert content['metrics']['like_count'] == 10
```

## Test Utilities

### `assert_dataframe_structure`

Utility function for validating DataFrame structure in tests.

```python
def assert_dataframe_structure(df, expected_columns, min_rows=1):
    """Assert DataFrame has expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    
    missing_columns = set(expected_columns) - set(df.columns)
    assert not missing_columns, f"Missing columns: {missing_columns}"
    
    extra_columns = set(df.columns) - set(expected_columns)
    if extra_columns:
        logger.warning(f"Extra columns found: {extra_columns}")
```

**Usage Example:**
```python
def test_conversation_processing(sample_signal_messages):
    """Test conversation processing output structure."""
    from src.core.conversation_processor import ConversationProcessor
    
    processor = ConversationProcessor()
    result_df = processor.process_messages(sample_signal_messages)
    
    expected_columns = ['message_id', 'sender_id', 'body', 'timestamp', 'thread_id']
    assert_dataframe_structure(result_df, expected_columns, min_rows=3)
```

### `check_sensitive_data_patterns`

Utility for testing privacy filtering and data anonymization.

```python
def check_sensitive_data_patterns(text, should_contain_sensitive=False):
    """Check for sensitive data patterns in text."""
    sensitive_patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
        r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card pattern
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
        r'\b\+?1?[- ]?\(?[0-9]{3}\)?[- ]?[0-9]{3}[- ]?[0-9]{4}\b'  # Phone pattern
    ]
    
    found_patterns = []
    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            found_patterns.append(pattern)
    
    if should_contain_sensitive:
        assert found_patterns, "Expected to find sensitive data patterns"
    else:
        assert not found_patterns, f"Found sensitive data patterns: {found_patterns}"
```

**Usage Example:**
```python
def test_privacy_filtering():
    """Test privacy filtering removes sensitive data."""
    from src.utils.privacy_filter import PrivacyFilter
    
    sensitive_text = "Contact me at john.doe@email.com or call 555-123-4567"
    filter = PrivacyFilter()
    
    # Check original contains sensitive data
    check_sensitive_data_patterns(sensitive_text, should_contain_sensitive=True)
    
    # Filter and check sensitive data is removed
    filtered_text = filter.anonymize_text(sensitive_text)
    check_sensitive_data_patterns(filtered_text, should_contain_sensitive=False)
```

## Test Markers and Organization

### Custom Pytest Markers

```python
# Unit tests - fast, isolated tests
@pytest.mark.unit
def test_emoji_extraction():
    """Test emoji extraction functionality."""
    pass

# Integration tests - test component interactions  
@pytest.mark.integration
def test_full_conversation_processing():
    """Test complete conversation processing pipeline."""
    pass

# Privacy tests - test data protection features
@pytest.mark.privacy
def test_sensitive_data_anonymization():
    """Test sensitive data is properly anonymized."""
    pass

# Slow tests - tests that take significant time
@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of large conversation datasets."""
    pass
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run privacy tests
pytest -m privacy

# Skip slow tests
pytest -m "not slow"

# Run specific test combinations
pytest -m "unit and not slow"
```

## Mock Configuration Patterns

### Environment Variable Mocking

```python
@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'TWITTER_BEARER_TOKEN': 'test_twitter_token',
        'LOG_LEVEL': 'DEBUG'
    }):
        yield
```

### File System Mocking

```python
@pytest.fixture
def mock_file_system(tmp_path):
    """Mock file system for testing file operations."""
    # Create temporary directory structure
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample files
    (data_dir / "messages.csv").write_text("id,body,sender\n1,Hello,user1")
    (data_dir / "recipients.csv").write_text("id,name,phone\n1,User1,+1234567890")
    
    with patch('src.utils.config.Config.DATA_DIR', str(data_dir)):
        yield data_dir
```

## Testing Best Practices

### Test Data Management

```python
# Use fixtures for reusable test data
@pytest.fixture
def conversation_data():
    """Provide consistent conversation data across tests."""
    return load_test_conversation_data()

# Parameterize tests for multiple scenarios
@pytest.mark.parametrize("emoji,expected_category", [
    ("😊", "joy_laughter"),
    ("😢", "sadness_crying"), 
    ("😡", "anger_frustration")
])
def test_emoji_categorization(emoji, expected_category):
    """Test emoji categorization for different emojis."""
    analyzer = EmojiAnalyzer()
    result = analyzer.categorize_emoji(emoji)
    assert result == expected_category
```

### Assertion Patterns

```python
# Test data structure and content
def test_training_data_format():
    """Test training data has correct format."""
    creator = TrainingDataCreator()
    data = creator.create_training_examples(sample_messages)
    
    # Structure assertions
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Content assertions
    for example in data:
        assert 'instruction' in example
        assert 'response' in example
        assert isinstance(example['instruction'], str)
        assert len(example['instruction']) > 0

# Test error conditions
def test_invalid_input_handling():
    """Test handling of invalid input data."""
    processor = ConversationProcessor()
    
    with pytest.raises(ValueError, match="Invalid message format"):
        processor.process_messages([{"invalid": "data"}])
```

### Performance Testing

```python
@pytest.mark.slow
def test_large_conversation_performance():
    """Test performance with large conversation datasets."""
    import time
    
    # Generate large dataset
    large_dataset = generate_large_conversation_data(10000)
    
    start_time = time.time()
    processor = ConversationProcessor()
    result = processor.process_messages(large_dataset)
    processing_time = time.time() - start_time
    
    # Performance assertions
    assert processing_time < 30.0, f"Processing took too long: {processing_time}s"
    assert len(result) == len(large_dataset)
```

## Integration with CI/CD

The testing infrastructure integrates with continuous integration:

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.12
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      
      - name: Run unit tests
        run: poetry run pytest -m unit --cov=src
      
      - name: Run integration tests  
        run: poetry run pytest -m integration
      
      - name: Run privacy tests
        run: poetry run pytest -m privacy
```

This comprehensive testing infrastructure ensures reliable, privacy-conscious development while maintaining high code quality and test coverage across all Astrabot components.
