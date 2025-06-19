"""Pytest configuration and shared fixtures for Astrabot tests.

This module provides centralized test configuration, fixtures, and utilities
following the TDD approach established in the project.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import json

# Test environment setup
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG'


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_signal_messages():
    """Create sample Signal messages DataFrame for testing."""
    return pd.DataFrame({
        '_id': [1, 2, 3, 4, 5],
        'thread_id': [101, 101, 102, 102, 103],
        'from_recipient_id': [2, 3, 2, 4, 2],
        'to_recipient_id': [3, 2, 4, 2, 5],
        'type': [10485780, 10485780, 10485780, 10485780, 10485780],
        'body': [
            "Hey! How's it going?",
            "Good! What are you up to?",
            "Working on the project",
            "Cool! Need any help?",
            "Thanks, I'm good for now"
        ],
        'date_sent': [1640995200000, 1640995260000, 1640995320000, 1640995380000, 1640995440000],
        'date_received': [1640995201000, 1640995261000, 1640995321000, 1640995381000, 1640995441000],
        'read': [1, 1, 1, 0, 0],
        'quote_id': [None, None, None, None, None],
        'quote_author': [None, None, None, None, None],
        'quote_body': [None, None, None, None, None],
        'mentions_self': [0, 0, 0, 0, 0],
        'remote_deleted': [0, 0, 0, 0, 0]
    })


@pytest.fixture(scope="session")
def sample_recipients():
    """Create sample recipients DataFrame for testing."""
    return pd.DataFrame({
        '_id': [2, 3, 4, 5],
        'type': [0, 0, 0, 0],  # Individual recipients
        'e164': ['+15551234567', '+15559876543', '+15555551234', '+15552223333'],
        'aci': ['aci-abc123', 'aci-def456', 'aci-ghi789', 'aci-jkl012'],
        'pni': ['pni-123abc', 'pni-456def', 'pni-789ghi', 'pni-012jkl'],
        'username': [None, 'user123', None, 'user456'],
        'blocked': [0, 0, 0, 0],
        'profile_given_name': ['John', 'Jane', 'Bob', 'Alice'],
        'profile_family_name': ['Doe', 'Smith', 'Johnson', 'Brown'],
        'system_given_name': ['John', 'Jane', 'Bob', 'Alice'],
        'system_family_name': ['Doe', 'Smith', 'Johnson', 'Brown'],
        'registered': [1, 1, 1, 1]
    })


@pytest.fixture(scope="session")
def sample_threads():
    """Create sample conversation threads DataFrame for testing."""
    return pd.DataFrame({
        '_id': [101, 102, 103],
        'recipient_id': [3, 4, 5],
        'date': [1640995260000, 1640995380000, 1640995440000],
        'meaningful_messages': [2, 2, 1],
        'read': [1, 0, 0],
        'snippet': ['Good! What are you up to?', 'Cool! Need any help?', 'Thanks, I\'m good for now'],
        'snippet_type': [0, 0, 0],
        'unread_count': [0, 1, 1],
        'archived': [0, 0, 0],
        'pinned_order': [None, None, None]
    })


@pytest.fixture(scope="session")
def sample_sensitive_messages():
    """Create sample messages with sensitive data for privacy testing."""
    return pd.DataFrame({
        '_id': [1, 2, 3, 4, 5],
        'thread_id': [101, 101, 102, 102, 103],
        'from_recipient_id': [2, 3, 2, 4, 2],
        'to_recipient_id': [3, 2, 4, 2, 5],
        'body': [
            "Call me at 555-123-4567",
            "My email is john.doe@example.com",
            "SSN: 123-45-6789",
            "Credit card: 4111 1111 1111 1111",
            "Regular message without PII"
        ],
        'date_sent': [1640995200000 + i*60000 for i in range(5)]
    })


@pytest.fixture
def temp_signal_csv(sample_signal_messages):
    """Create a temporary Signal CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_signal_messages.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_recipients_csv(sample_recipients):
    """Create a temporary recipients CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_recipients.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses for testing."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "This is a test image showing a scenic landscape."
    
    with patch('openai.chat.completions.create', return_value=mock_response):
        yield mock_response


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API responses for testing."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "This is a test image showing people in conversation."
    
    with patch('anthropic.messages.create', return_value=mock_response):
        yield mock_response


@pytest.fixture
def mock_twitter_responses():
    """Mock Twitter/Nitter responses for testing."""
    mock_tweet_html = """
    <div class="tweet-content">
        <div class="tweet-text">This is a sample tweet for testing purposes.</div>
        <div class="tweet-stats">
            <span class="tweet-stat">10 replies</span>
            <span class="tweet-stat">25 retweets</span>
            <span class="tweet-stat">50 likes</span>
        </div>
    </div>
    """
    
    mock_response = MagicMock()
    mock_response.text = mock_tweet_html
    mock_response.status_code = 200
    
    with patch('requests.get', return_value=mock_response):
        yield mock_response


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    return [
        {
            'instruction': 'Continue this conversation naturally',
            'input': 'Friend: How was your day?\nYou: Pretty good!',
            'output': 'Got a lot done at work today',
            'metadata': {
                'type': 'conversation_window',
                'momentum': 'moderate',
                'response_delay': 45.2,
                'context_size': 2
            }
        },
        {
            'instruction': 'Continue this rapid conversation',
            'input': 'Friend: Quick question\nYou: What\'s up?',
            'output': 'Can you pick up milk?',
            'metadata': {
                'type': 'conversation_window',
                'momentum': 'rapid',
                'response_delay': 15.0,
                'context_size': 2
            }
        }
    ]


@pytest.fixture
def sample_style_analysis():
    """Create sample style analysis results for testing."""
    return {
        'avg_message_length': 45.2,
        'message_length_distribution': {
            'very_short': 10,
            'short': 25,
            'medium': 40,
            'long': 20,
            'very_long': 5
        },
        'burst_patterns': {
            'total_bursts': 156,
            'avg_burst_size': 3.2,
            'burst_frequency': 0.34,
            'max_burst_size': 8
        },
        'preferred_length': 'medium',
        'emoji_usage': {
            'emoji_frequency': 0.23,
            'messages_with_emojis': 450,
            'emoji_usage_rate': '23.0%',
            'top_emojis': ['😂', '👍', '❤️']
        },
        'total_messages': 1956
    }


@pytest.fixture
def sample_conversation_windows():
    """Create sample conversation windows for testing."""
    return [
        {
            'thread_id': 101,
            'context': [
                {'speaker': 'Other', 'text': 'Hey!', 'timestamp': 1234567890},
                {'speaker': 'You', 'text': 'Hi there!', 'timestamp': 1234567891}
            ],
            'response': {
                'text': 'How are you?',
                'timestamp': 1234567892
            },
            'metadata': {
                'momentum': 'rapid',
                'context_size': 2,
                'avg_time_gap': 1.0,
                'response_delay': 1.0
            }
        },
        {
            'thread_id': 102,
            'context': [
                {'speaker': 'You', 'text': 'Working on something interesting', 'timestamp': 1234567900},
                {'speaker': 'Other', 'text': 'What kind of project?', 'timestamp': 1234567960}
            ],
            'response': {
                'text': 'A machine learning project',
                'timestamp': 1234568020
            },
            'metadata': {
                'momentum': 'moderate',
                'context_size': 2,
                'avg_time_gap': 60.0,
                'response_delay': 60.0
            }
        }
    ]


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP pipeline for testing."""
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    
    # Mock named entity recognition
    mock_entity = MagicMock()
    mock_entity.label_ = 'PERSON'
    mock_entity.start_char = 0
    mock_entity.end_char = 4
    mock_entity.text = 'John'
    mock_doc.ents = [mock_entity]
    
    mock_nlp.return_value = mock_doc
    
    with patch('spacy.load', return_value=mock_nlp):
        yield mock_nlp


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure for testing."""
    # Create the Signal data structure
    signal_dir = tmp_path / "raw" / "signal-flatfiles"
    signal_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processed directory
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return tmp_path


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    env_content = """
# Test environment variables
YOUR_RECIPIENT_ID=2
OPENAI_API_KEY=sk-test123456789
ANTHROPIC_API_KEY=sk-ant-test123456789
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_IMAGE_PROCESSING=false
ENABLE_BATCH_PROCESSING=true
MAX_BATCH_SIZE=5
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write(env_content.strip())
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_tweet_data():
    """Create sample tweet data for testing."""
    return {
        'url': 'https://twitter.com/user/status/1234567890',
        'text': 'This is a sample tweet for testing purposes.',
        'author': '@testuser',
        'timestamp': '2024-01-15T10:30:00Z',
        'metrics': {
            'replies': 10,
            'retweets': 25,
            'likes': 50
        },
        'images': [
            'https://pbs.twimg.com/media/test1.jpg',
            'https://pbs.twimg.com/media/test2.jpg'
        ]
    }


@pytest.fixture
def privacy_test_patterns():
    """Create privacy test patterns for validation."""
    return {
        'phones': [
            '555-123-4567',
            '(555) 123-4567',
            '555.123.4567',
            '+1-555-123-4567'
        ],
        'emails': [
            'user@example.com',
            'test.email+tag@domain.co.uk',
            'firstname.lastname@company.org'
        ],
        'ssns': [
            '123-45-6789',
            '987-65-4321'
        ],
        'credit_cards': [
            '4111 1111 1111 1111',
            '4111-1111-1111-1111',
            '4111111111111111'
        ],
        'api_keys': [
            'sk-1234567890abcdef1234567890abcdef',
            'sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz'
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Automatically set up test environment for all tests."""
    # Set test environment variables
    monkeypatch.setenv('TESTING', 'true')
    monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('YOUR_RECIPIENT_ID', '2')
    
    # Disable external API calls during testing
    monkeypatch.setenv('ENABLE_IMAGE_PROCESSING', 'false')
    monkeypatch.setenv('ENABLE_BATCH_PROCESSING', 'false')


@pytest.fixture
def captured_logs(caplog):
    """Capture log output for testing."""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# Test utility functions
def assert_dataframe_structure(df, expected_columns, expected_length=None):
    """Assert that DataFrame has expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert list(df.columns) == expected_columns, f"Columns mismatch: {list(df.columns)} != {expected_columns}"
    if expected_length is not None:
        assert len(df) == expected_length, f"Length mismatch: {len(df)} != {expected_length}"


def assert_no_sensitive_data(text, sensitive_patterns):
    """Assert that text contains no sensitive data patterns."""
    for pattern_type, patterns in sensitive_patterns.items():
        for pattern in patterns:
            assert pattern not in text, f"Sensitive {pattern_type} pattern '{pattern}' found in text"


def assert_training_data_format(training_data):
    """Assert that training data follows expected format."""
    assert isinstance(training_data, list), "Training data should be a list"
    
    for item in training_data:
        assert isinstance(item, dict), "Each training item should be a dictionary"
        assert 'instruction' in item, "Missing 'instruction' field"
        assert 'input' in item, "Missing 'input' field"
        assert 'output' in item, "Missing 'output' field"
        assert isinstance(item['instruction'], str), "'instruction' should be string"
        assert isinstance(item['input'], str), "'input' should be string"
        assert isinstance(item['output'], str), "'output' should be string"


# Pytest markers for test organization
pytest_plugins = []

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "privacy: Privacy filter tests")
    config.addinivalue_line("markers", "twitter: Twitter extraction tests")
    config.addinivalue_line("markers", "llm: LLM training tests")
    config.addinivalue_line("markers", "requires_api: Tests requiring API keys")
    config.addinivalue_line("markers", "requires_signal_data: Tests requiring Signal data")


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Add unit marker to all tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to all tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add privacy marker to privacy-related tests
        if "privacy" in str(item.fspath) or "privacy" in item.name:
            item.add_marker(pytest.mark.privacy)
        
        # Add twitter marker to Twitter-related tests
        if "twitter" in str(item.fspath) or "twitter" in item.name:
            item.add_marker(pytest.mark.twitter)


# Session-level setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Set up test session."""
    # Create test directories if they don't exist
    test_dirs = ['fixtures', 'unit', 'integration', 'performance']
    for dir_name in test_dirs:
        dir_path = Path(__file__).parent / dir_name
        dir_path.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after session (if needed)
    pass