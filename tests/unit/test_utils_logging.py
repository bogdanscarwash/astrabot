"""
Unit tests for the centralized logging module.

Tests focus on:
1. Core logging functionality (singleton, levels, masking)
2. Decorators and context managers for performance tracking
3. Metadata logging for conversation processing framework
4. Structured logging for tweet extraction and enrichment
"""

import pytest
import os
import tempfile
import json
import re
import time
import threading
from unittest.mock import patch, MagicMock, call
from io import StringIO
import logging as stdlib_logging
from datetime import datetime

from src.utils.logging import (
    AstrabotLogger,
    get_logger,
    log_performance,
    log_api_call,
    log_data_processing,
    mask_sensitive_data,
    setup_logging
)


@pytest.mark.unit
class TestAstrabotLogger:
    """Test the AstrabotLogger class core functionality"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Set up and tear down test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        
        # Reset the singleton logger instance before each test
        import src.utils.logging
        src.utils.logging._logger_instance = None
        
        yield
        
        # Clean up after tests - Reset logging
        logger = stdlib_logging.getLogger('astrabot')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Reset the singleton again
        src.utils.logging._logger_instance = None
    
    def test_singleton_pattern(self):
        """Test that logger follows singleton pattern"""
        logger1 = get_logger('test1')
        logger2 = get_logger('test2')
        
        # Should return the same logger instance
        assert logger1 is logger2
        assert isinstance(logger1, AstrabotLogger)
        assert logger1.name == 'astrabot'
    
    def test_log_levels(self):
        """Test different log levels"""
        output = StringIO()
        setup_logging(log_level='DEBUG', stream=output)
        logger = get_logger('test')
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        
        output_value = output.getvalue()
        
        # Check that messages are logged (exact format depends on configuration)
        assert "Debug message" in output_value
        assert "Info message" in output_value
        assert "Warning message" in output_value
        assert "Error message" in output_value
    
    def test_sensitive_data_masking(self):
        """Test that sensitive data is automatically masked"""
        output = StringIO()
        setup_logging(stream=output)
        logger = get_logger('test')
        
        # Log messages containing sensitive data
        logger.info("API key is sk-1234567890abcdef")
        logger.info("Using key: OPENAI_API_KEY=sk-secret123")
        logger.info("Bearer token: Bearer sk-1234567890")
        
        output_value = output.getvalue()
        
        # Sensitive data should be masked
        assert "sk-1234567890abcdef" not in output_value
        assert "sk-secret123" not in output_value
        assert "sk-****" in output_value
    
    def test_structured_logging_with_metadata(self):
        """Test structured logging with conversation metadata"""
        logger = get_logger('test')
        
        # Test logging with conversation context
        logger.info("Processing message", extra={
            'conversation_id': 'conv_123',
            'message_id': 'msg_456',
            'user_id': 'user_789',
            'thread_id': 'thread_001',
            'has_tweet': True,
            'has_images': False
        })
        
        # Should not raise any exceptions
        assert True
    
    def test_file_handler(self):
        """Test file logging"""
        # Setup logging with file handler
        setup_logging(log_file=self.log_file, log_level='DEBUG')
        logger = get_logger('test')
        
        logger.info("Test file logging")
        
        # Check that log file was created and contains the message
        assert os.path.exists(self.log_file)
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            assert "Test file logging" in content
    
    def test_log_rotation(self):
        """Test log file rotation"""
        # Setup with small max size to trigger rotation
        setup_logging(
            log_file=self.log_file,
            max_bytes=100,  # Very small to force rotation
            backup_count=2
        )
        logger = get_logger('test')
        
        # Write enough to trigger rotation
        for i in range(10):
            logger.info(f"This is a long message to trigger rotation {i}" * 5)
        
        # Check that main log file exists
        assert os.path.exists(self.log_file)
    
    def test_thread_safety(self):
        """Test that logger is thread-safe"""
        logger = get_logger('test')
        errors = []
        
        def log_from_thread(thread_id):
            try:
                for i in range(10):
                    logger.info(f"Thread {thread_id} message {i}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_from_thread, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # No errors should occur
        assert len(errors) == 0
    
    def test_context_manager(self):
        """Test logger context manager for adding metadata"""
        logger = get_logger('test')
        
        with logger.context(conversation_id='conv_123', user_id='user_456'):
            logger.info("Message with context")
            
            # Nested context
            with logger.context(message_id='msg_789', tweet_url='https://twitter.com/user/status/123'):
                logger.info("Nested context message")
        
        logger.info("Message without context")
        
        # Should complete without errors
        assert True


@pytest.mark.unit
class TestLoggingDecorators:
    """Test logging decorators for performance and API tracking"""
    
    def test_log_performance_decorator(self):
        """Test performance logging decorator"""
        @log_performance("test_function")
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        with patch('src.utils.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = slow_function()
            
            # Check that performance was logged
            assert mock_logger.info.called
            call_args = mock_logger.info.call_args[0][0]
            assert "test_function" in call_args
            assert "completed in" in call_args
            
            # Check function still returns correct result
            assert result == "result"
    
    def test_log_api_call_decorator(self):
        """Test API call logging decorator"""
        @log_api_call("OpenAI")
        def make_api_call(url, api_key=None):
            return {"status": "success"}
        
        with patch('src.utils.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            result = make_api_call("https://api.openai.com/v1/test", api_key="sk-secret")
            
            # Check that API call was logged
            assert mock_logger.info.call_count >= 2  # Start and success
            
            # Check sensitive data was not logged
            call_args = str(mock_logger.info.call_args_list)
            assert "sk-secret" not in call_args
    
    def test_log_data_processing(self):
        """Test data processing logging context manager"""
        with patch('src.utils.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with log_data_processing("tweets", count=100) as ctx:
                # Simulate processing
                ctx['processed'] = 50
                ctx['processed'] = 100
            
            # Check that start and completion were logged
            assert mock_logger.info.call_count >= 2
            
            # Check start message
            start_call = mock_logger.info.call_args_list[0]
            assert "Starting tweets processing" in start_call[0][0]
            
            # Check completion message
            end_call = mock_logger.info.call_args_list[-1]
            assert "Completed tweets processing" in end_call[0][0]
            assert "100 items" in end_call[0][0]


@pytest.mark.unit
class TestLoggingUtilities:
    """Test logging utility functions"""
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking function"""
        # Test API key masking
        text = "My API key is sk-1234567890abcdef"
        masked = mask_sensitive_data(text)
        assert "sk-1234567890abcdef" not in masked
        assert "sk-****" in masked
        
        # Test multiple patterns
        text = "Bearer sk-abc123 and key=AIza123456"
        masked = mask_sensitive_data(text)
        assert "sk-abc123" not in masked
        assert "AIza123456" not in masked
        
        # Test that non-sensitive data is preserved
        text = "This is a normal message"
        masked = mask_sensitive_data(text)
        assert text == masked
    
    def test_setup_logging_configuration(self):
        """Test logging setup with different configurations"""
        # Test with environment variables
        with patch.dict(os.environ, {'LOG_LEVEL': 'WARNING', 'LOG_FORMAT': 'json'}):
            setup_logging()
            logger = get_logger('test')
            
            # Logger should respect environment settings
            assert logger.level == stdlib_logging.WARNING


@pytest.mark.unit
class TestConversationMetadataLogging:
    """Test logging of conversation processing metadata"""
    
    def test_conversation_processing_metadata(self):
        """Test logging conversation processing metadata"""
        output = StringIO()
        setup_logging(stream=output)
        logger = get_logger('conversation')
        
        # Log conversation processing start
        logger.info("Starting conversation processing", extra={
            'thread_id': 'thread_123',
            'message_count': 150,
            'date_range': '2024-01-01 to 2024-01-31',
            'has_blocked_contact': False
        })
        
        # Log individual message processing
        logger.info("Processing message", extra={
            'message_id': 'msg_001',
            'from_recipient_id': 2,
            'has_urls': True,
            'url_count': 2,
            'has_twitter': True,
            'message_length': 280
        })
        
        # Log tweet extraction
        logger.info("Extracting tweet content", extra={
            'tweet_url': 'https://twitter.com/user/status/123',
            'tweet_id': '123',
            'extraction_method': 'nitter',
            'has_images': True,
            'image_count': 3
        })
        
        # Log image description
        logger.info("Describing images", extra={
            'image_urls': ['url1', 'url2', 'url3'],
            'vision_api': 'openai',
            'model': 'gpt-4o-mini',
            'total_tokens': 500
        })
        
        output_value = output.getvalue()
        
        # Verify metadata is being logged
        assert "Starting conversation processing" in output_value
        assert "Processing message" in output_value
        assert "Extracting tweet content" in output_value
        assert "Describing images" in output_value
    
    def test_conversation_window_metadata(self):
        """Test logging conversation window creation metadata"""
        logger = get_logger('conversation')
        
        # Log conversation window creation
        logger.info("Creating conversation window", extra={
            'window_id': 'window_001',
            'context_messages': 5,
            'your_response_id': 'msg_006',
            'conversation_momentum': 'rapid',
            'avg_time_gap_seconds': 45,
            'has_media': True,
            'emoji_density': 0.15
        })
        
        # Should not raise exceptions
        assert True
    
    def test_training_data_creation_metadata(self):
        """Test logging training data creation metadata"""
        logger = get_logger('training')
        
        # Log training data creation
        logger.info("Creating training examples", extra={
            'format': 'conversational',
            'total_examples': 1000,
            'filtered_examples': 850,
            'filter_reasons': {
                'too_short': 100,
                'no_context': 50
            },
            'avg_context_length': 350,
            'avg_response_length': 75
        })
        
        # Log batch processing
        logger.info("Processing batch", extra={
            'batch_number': 5,
            'batch_size': 100,
            'processing_time_seconds': 2.5,
            'examples_per_second': 40
        })
        
        assert True
    
    def test_style_analysis_metadata(self):
        """Test logging style analysis metadata"""
        logger = get_logger('style')
        
        # Log style analysis results
        logger.info("Analyzed communication style", extra={
            'user_id': 'user_123',
            'total_messages': 5000,
            'avg_message_length': 45.2,
            'emoji_usage_rate': 0.23,
            'url_sharing_rate': 0.08,
            'burst_pattern_frequency': 0.35,
            'dominant_conversation_times': ['evening', 'late_night'],
            'personality_traits': {
                'humor': 0.8,
                'formality': 0.3,
                'enthusiasm': 0.7
            }
        })
        
        assert True


@pytest.mark.unit
class TestJSONStructuredLogging:
    """Test JSON structured logging output"""
    
    def test_json_output_format(self):
        """Test JSON structured logging output"""
        output = StringIO()
        
        # Setup JSON logging
        setup_logging(log_format='json', stream=output)
        logger = get_logger('test')
        
        # Log with metadata
        logger.info("Test message", extra={
            'conversation_id': 'conv_123',
            'processing_stage': 'tweet_extraction',
            'success': True
        })
        
        # Parse JSON output
        output.seek(0)
        log_line = output.getvalue().strip()
        
        # Should be valid JSON
        log_data = json.loads(log_line)
        assert log_data['message'] == "Test message"
        assert log_data['conversation_id'] == 'conv_123'
        assert log_data['processing_stage'] == 'tweet_extraction'
        assert log_data['success'] is True
        assert 'timestamp' in log_data
        assert 'level' in log_data
    
    def test_json_with_nested_metadata(self):
        """Test JSON logging with nested metadata structures"""
        output = StringIO()
        
        setup_logging(log_format='json', stream=output)
        logger = get_logger('test')
        
        # Log with nested metadata
        logger.info("Processing complete", extra={
            'results': {
                'messages_processed': 100,
                'tweets_extracted': 25,
                'images_described': 10
            },
            'performance': {
                'total_time': 5.2,
                'avg_per_message': 0.052
            },
            'errors': []
        })
        
        output.seek(0)
        log_data = json.loads(output.getvalue().strip())
        
        assert log_data['results']['messages_processed'] == 100
        assert log_data['performance']['total_time'] == 5.2
        assert log_data['errors'] == []


@pytest.mark.unit
class TestErrorHandlingAndRecovery:
    """Test error handling and recovery in logging"""
    
    def test_logging_with_unicode_and_special_chars(self):
        """Test logging handles unicode and special characters"""
        logger = get_logger('test')
        
        # Various unicode and special characters
        messages = [
            "Unicode emoji: 🚀 🎉 ❤️",
            "Special chars: <>&\"'",
            "Newlines:\nMultiple\nLines",
            "Tabs:\t\tIndented",
            "Mixed: 你好 مرحبا Здравствуйте"
        ]
        
        for msg in messages:
            # Should not raise exceptions
            logger.info(msg)
        
        assert True
    
    def test_logging_with_large_metadata(self):
        """Test logging with large metadata objects"""
        logger = get_logger('test')
        
        # Create large metadata
        large_list = ['item'] * 1000
        large_dict = {f'key_{i}': f'value_{i}' for i in range(100)}
        
        logger.info("Large metadata test", extra={
            'large_list_sample': large_list[:10],  # Only log sample
            'large_dict_keys': list(large_dict.keys())[:10],
            'total_items': len(large_list),
            'dict_size': len(large_dict)
        })
        
        assert True
    
    def test_concurrent_logging(self):
        """Test concurrent logging from multiple sources"""
        logger = get_logger('test')
        results = []
        
        def log_batch(source_id):
            try:
                for i in range(20):
                    logger.info(f"Message from source {source_id}", extra={
                        'source': source_id,
                        'message_num': i,
                        'timestamp': time.time()
                    })
                results.append(f"source_{source_id}_complete")
            except Exception as e:
                results.append(f"source_{source_id}_error: {e}")
        
        # Start multiple concurrent loggers
        threads = []
        for i in range(10):
            t = threading.Thread(target=log_batch, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All should complete successfully
        assert len(results) == 10
        assert all('complete' in r for r in results)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])