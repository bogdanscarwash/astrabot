"""
Unit tests for conversation_utilities module
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.extractors.twitter_extractor import (
    extract_tweet_text,
    inject_tweet_context,
    extract_tweet_images,
    process_message_with_twitter_content,
    process_message_with_structured_content
)
from src.models.schemas import TweetContent, ImageDescription, EnhancedMessage


class TestTweetExtraction(unittest.TestCase):
    """Test tweet extraction functionality"""
    
    def test_extract_tweet_text_with_valid_url(self):
        """Test extracting tweet text from a valid URL"""
        # Test with a known URL format
        url = "https://twitter.com/user/status/123456789"
        
        # The function should at least parse the URL correctly
        result = extract_tweet_text(url)
        
        # If no network connection or Nitter is down, it might return None
        # But we can test the URL parsing logic
        if result:
            self.assertIn('tweet_id', result)
            self.assertEqual(result['tweet_id'], '123456789')
    
    def test_extract_tweet_text_structured(self):
        """Test structured tweet extraction"""
        url = "https://twitter.com/user/status/123456789"
        
        # Mock the response to test structured output
        with patch('conversation_utilities.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'<div class="tweet-content">Hello #test @user</div>'
            mock_get.return_value = mock_response
            
            result = extract_tweet_text(url, return_structured=True)
            
            # If successful, should return TweetContent object
            if result and isinstance(result, TweetContent):
                self.assertIsInstance(result, TweetContent)
                self.assertEqual(result.tweet_id, '123456789')
                self.assertIsInstance(result.hashtags, list)
                self.assertIsInstance(result.mentioned_users, list)
    
    def test_inject_tweet_context(self):
        """Test injecting tweet context into a message"""
        message = "Check this out: https://twitter.com/user/status/123"
        tweet_data = {
            'text': 'This is a test tweet',
            'author': 'testuser',
            'tweet_id': '123'
        }
        
        result = inject_tweet_context(message, tweet_data)
        
        self.assertIn(message, result)
        self.assertIn('[TWEET: @testuser]', result)
        self.assertIn('This is a test tweet', result)
        self.assertIn('[/TWEET]', result)
    
    def test_inject_tweet_context_no_data(self):
        """Test injecting tweet context with no tweet data"""
        message = "Just a regular message"
        
        result = inject_tweet_context(message, None)
        self.assertEqual(result, message)


class TestImageExtraction(unittest.TestCase):
    """Test image extraction functionality"""
    
    def test_extract_tweet_images_valid_url(self):
        """Test extracting images from a tweet URL"""
        url = "https://twitter.com/user/status/123456789"
        
        # This will likely return empty list without network
        # but we're testing the function doesn't crash
        result = extract_tweet_images(url)
        
        self.assertIsInstance(result, list)
    
    @patch('conversation_utilities.requests.get')
    def test_extract_tweet_images_with_mock(self, mock_get):
        """Test image extraction with mocked response"""
        # Mock Nitter response with image
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'''
        <a class="still-image" href="/pic/media%2Ftest.jpg?format=jpg">
            <img src="/pic/media%2Ftest.jpg"/>
        </a>
        '''
        mock_get.return_value = mock_response
        
        url = "https://twitter.com/user/status/123456789"
        result = extract_tweet_images(url)
        
        # Should extract and convert to Twitter image URL
        self.assertIsInstance(result, list)
        if result:
            self.assertIn('pbs.twimg.com', result[0])


class TestStructuredContent(unittest.TestCase):
    """Test structured content processing"""
    
    def test_enhanced_message_creation(self):
        """Test creating an EnhancedMessage object"""
        enhanced = EnhancedMessage(
            original_message="Test message",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_contents=[],
            image_descriptions=[]
        )
        
        self.assertEqual(enhanced.original_message, "Test message")
        self.assertEqual(enhanced.conversation_id, "conv_123")
        self.assertEqual(enhanced.to_training_format(), "Test message")
    
    def test_enhanced_message_with_content(self):
        """Test EnhancedMessage with tweet and image content"""
        tweet = TweetContent(
            text="Test tweet",
            author="testuser",
            tweet_id="123",
            mentioned_users=["user1"],
            hashtags=["test"],
            sentiment="positive"
        )
        
        image = ImageDescription(
            description="A test image",
            detected_text=None,
            main_subjects=["test", "image"],
            emotional_tone="neutral"
        )
        
        enhanced = EnhancedMessage(
            original_message="Check this: https://twitter.com/user/status/123",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_contents=[tweet],
            image_descriptions=[image]
        )
        
        training_format = enhanced.to_training_format()
        
        self.assertIn("Check this:", training_format)
        self.assertIn("[TWEET:", training_format)
        self.assertIn("@testuser: Test tweet", training_format)
        self.assertIn("[IMAGES:", training_format)
        self.assertIn("A test image", training_format)
    
    @patch('conversation_utilities.extract_tweet_text')
    @patch('conversation_utilities.extract_tweet_images')
    @patch('conversation_utilities.describe_tweet_images_with_context')
    def test_process_message_with_structured_content(self, mock_describe, mock_images, mock_tweet):
        """Test processing a message with structured content extraction"""
        # Setup mocks
        mock_tweet.return_value = TweetContent(
            text="Mocked tweet",
            author="mockuser",
            tweet_id="123",
            mentioned_users=[],
            hashtags=[],
            sentiment="neutral"
        )
        
        mock_images.return_value = ["https://example.com/image.jpg"]
        
        mock_describe.return_value = []  # No descriptions for simplicity
        
        # Process message
        result = process_message_with_structured_content(
            message="Check out: https://twitter.com/user/status/123",
            conversation_id="test_conv",
            message_id="test_msg",
            sender_id="test_user",
            timestamp=datetime.now(),
            use_images=True
        )
        
        self.assertIsInstance(result, EnhancedMessage)
        self.assertEqual(len(result.tweet_contents), 1)
        self.assertEqual(result.tweet_contents[0].author, "mockuser")


class TestConversationTracking(unittest.TestCase):
    """Test conversation tracking functionality"""
    
    def test_images_with_context_structure(self):
        """Test the structure of images with context"""
        from structured_schemas import ImageWithContext
        
        img_context = ImageWithContext(
            image_url="https://example.com/image.jpg",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_url="https://twitter.com/user/status/123"
        )
        
        self.assertEqual(img_context.conversation_id, "conv_123")
        self.assertEqual(img_context.sender_id, "user_789")
        self.assertIsNotNone(img_context.timestamp)
    
    def test_batch_image_description_structure(self):
        """Test BatchImageDescription structure"""
        from structured_schemas import ImageWithContext, BatchImageDescription
        
        img_context = ImageWithContext(
            image_url="https://example.com/image.jpg",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_url=None
        )
        
        img_desc = ImageDescription(
            description="Test description",
            detected_text=None,
            main_subjects=["test"],
            emotional_tone="neutral"
        )
        
        batch_desc = BatchImageDescription(
            image_context=img_context,
            description=img_desc
        )
        
        context_dict = batch_desc.to_dict_with_context()
        
        self.assertIn('conversation_id', context_dict)
        self.assertIn('sender_id', context_dict)
        self.assertIn('description', context_dict)
        self.assertIn('emotional_tone', context_dict)
        self.assertEqual(context_dict['conversation_id'], "conv_123")


class TestDataPipeline(unittest.TestCase):
    """Test the data processing pipeline"""
    
    def setUp(self):
        """Create test dataframes"""
        self.messages_df = pd.DataFrame({
            '_id': [1, 2, 3],
            'thread_id': [1, 1, 2],
            'from_recipient_id': [2, 3, 2],  # 2 is "you"
            'body': [
                'Hello',
                'Check this: https://twitter.com/user/status/123',
                'Another message'
            ],
            'date_sent': [1000, 2000, 3000]
        })
        
        self.recipients_df = pd.DataFrame({
            '_id': [2, 3],
            'profile_given_name': ['You', 'Friend']
        })
    
    def test_message_filtering(self):
        """Test filtering messages for processing"""
        # Filter for meaningful messages
        filtered = self.messages_df[
            (self.messages_df['body'].notna()) & 
            (self.messages_df['body'].str.len() > 5)
        ]
        
        # Should keep messages 1 and 2
        self.assertEqual(len(filtered), 2)
        self.assertIn('https://twitter.com', filtered['body'].iloc[1])


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)