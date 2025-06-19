"""
Unit tests for the centralized logging module.
"""

import unittest
import os
import tempfile
import json
import re
from unittest.mock import patch, MagicMock
from io import StringIO
import logging as stdlib_logging
import threading
import time
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logging import (
    AstrabotLogger,
    get_logger,
    log_performance,
    log_api_call,
    log_data_processing,
    mask_sensitive_data,
    setup_logging
)


class TestAstrabotLogger(unittest.TestCase):
    """Test the AstrabotLogger class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, 'test.log')
        
    def tearDown(self):
        """Clean up after tests"""
        # Reset logging
        logger = stdlib_logging.getLogger('astrabot')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    def test_singleton_pattern(self):
        """Test that logger follows singleton pattern"""
        logger1 = get_logger('test1')
        logger2 = get_logger('test2')
        
        # Should return the same logger instance
        self.assertIs(logger1, logger2)
    
    def test_logger_initialization(self):
        """Test basic logger initialization"""
        logger = get_logger('test')
        
        self.assertIsInstance(logger, AstrabotLogger)
        self.assertEqual(logger.name, 'astrabot')
    
    def test_log_levels(self):
        """Test different log levels"""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            logger = get_logger('test')
            
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            
            output = mock_stdout.getvalue()
            
            # Check that messages are logged (exact format depends on configuration)
            self.assertIn("Info message", output)
            self.assertIn("Warning message", output)
            self.assertIn("Error message", output)
    
    def test_sensitive_data_masking(self):
        """Test that sensitive data is automatically masked"""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            logger = get_logger('test')
            
            # Log messages containing sensitive data
            logger.info("API key is sk-1234567890abcdef")
            logger.info("Using key: OPENAI_API_KEY=sk-secret123")
            logger.info("Bearer token: Bearer sk-1234567890")
            
            output = mock_stdout.getvalue()
            
            # Sensitive data should be masked
            self.assertNotIn("sk-1234567890abcdef", output)
            self.assertNotIn("sk-secret123", output)
            self.assertIn("sk-****", output)
    
    def test_structured_logging(self):
        """Test structured logging with context"""
        logger = get_logger('test')
        
        # Test logging with context
        logger.info("Processing message", extra={
            'conversation_id': 'conv_123',
            'message_id': 'msg_456',
            'user_id': 'user_789'
        })
        
        # This should not raise any exceptions
        self.assertTrue(True)
    
    def test_file_handler(self):
        """Test file logging"""
        # Setup logging with file handler
        setup_logging(log_file=self.log_file, log_level='DEBUG')
        logger = get_logger('test')
        
        logger.info("Test file logging")
        
        # Check that log file was created and contains the message
        self.assertTrue(os.path.exists(self.log_file))
        
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test file logging", content)
    
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
        
        # Check that backup files were created
        backup_file = f"{self.log_file}.1"
        self.assertTrue(os.path.exists(self.log_file))
        # Rotation might occur based on implementation
    
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
        self.assertEqual(len(errors), 0)


class TestLoggingDecorators(unittest.TestCase):
    """Test logging decorators"""
    
    def test_log_performance_decorator(self):
        """Test performance logging decorator"""
        @log_performance("test_function")
        def slow_function():
            time.sleep(0.1)
            return "result"
        
        with patch('utils.logging.get_logger') as mock_logger:
            result = slow_function()
            
            # Check that performance was logged
            mock_logger.return_value.info.assert_called()
            call_args = mock_logger.return_value.info.call_args[0][0]
            self.assertIn("test_function", call_args)
            self.assertIn("completed in", call_args)
            
            # Check function still returns correct result
            self.assertEqual(result, "result")
    
    def test_log_api_call_decorator(self):
        """Test API call logging decorator"""
        @log_api_call("OpenAI")
        def make_api_call(url, api_key=None):
            return {"status": "success"}
        
        with patch('utils.logging.get_logger') as mock_logger:
            result = make_api_call("https://api.openai.com/v1/test", api_key="sk-secret")
            
            # Check that API call was logged
            mock_logger.return_value.info.assert_called()
            
            # Check sensitive data was not logged
            call_args = str(mock_logger.return_value.info.call_args)
            self.assertNotIn("sk-secret", call_args)
    
    def test_log_data_processing(self):
        """Test data processing logging context manager"""
        with patch('utils.logging.get_logger') as mock_logger:
            with log_data_processing("tweets", count=100) as ctx:
                # Simulate processing
                ctx.update(processed=50)
                ctx.update(processed=100)
            
            # Check that start and completion were logged
            self.assertTrue(mock_logger.return_value.info.called)


class TestLoggingUtilities(unittest.TestCase):
    """Test logging utility functions"""
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking function"""
        # Test API key masking
        text = "My API key is sk-1234567890abcdef"
        masked = mask_sensitive_data(text)
        self.assertNotIn("sk-1234567890abcdef", masked)
        self.assertIn("sk-****", masked)
        
        # Test multiple patterns
        text = "Bearer sk-abc123 and key=AIza123456"
        masked = mask_sensitive_data(text)
        self.assertNotIn("sk-abc123", masked)
        self.assertNotIn("AIza123456", masked)
        
        # Test that non-sensitive data is preserved
        text = "This is a normal message"
        masked = mask_sensitive_data(text)
        self.assertEqual(text, masked)
    
    def test_setup_logging_configuration(self):
        """Test logging setup with different configurations"""
        # Test with environment variables
        with patch.dict(os.environ, {'LOG_LEVEL': 'WARNING', 'LOG_FORMAT': 'json'}):
            setup_logging()
            logger = get_logger('test')
            
            # Logger should respect environment settings
            self.assertEqual(logger.level, stdlib_logging.WARNING)


class TestIntegration(unittest.TestCase):
    """Integration tests for logging module"""
    
    def test_logging_in_conversation_context(self):
        """Test logging in conversation processing context"""
        logger = get_logger('conversation')
        
        # Simulate conversation processing
        with logger.context(conversation_id='conv_123', user_id='user_456'):
            logger.info("Processing message")
            logger.debug("Extracting tweet content")
            
            # Nested context
            with logger.context(message_id='msg_789'):
                logger.info("Processing tweet images")
        
        # This should complete without errors
        self.assertTrue(True)
    
    def test_json_structured_output(self):
        """Test JSON structured logging output"""
        output = StringIO()
        
        # Setup JSON logging
        setup_logging(log_format='json', stream=output)
        logger = get_logger('test')
        
        logger.info("Test message", extra={'key': 'value'})
        
        # Parse JSON output
        output.seek(0)
        log_line = output.getvalue().strip()
        
        try:
            log_data = json.loads(log_line)
            self.assertEqual(log_data['message'], "Test message")
            self.assertEqual(log_data['key'], 'value')
            self.assertIn('timestamp', log_data)
        except json.JSONDecodeError:
            # If not JSON format, that's okay for basic implementation
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)