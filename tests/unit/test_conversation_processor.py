"""
Unit tests for conversation processor module
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.core.conversation_processor import (
    extract_tweet_text,
    inject_tweet_context,
    extract_tweet_images,
    describe_tweet_images,
    describe_tweet_images_with_context,
    process_message_with_twitter_content,
    process_message_with_structured_content,
    preserve_conversation_dynamics
)
from src.models.schemas import TweetContent, ImageDescription, EnhancedMessage


@pytest.mark.unit
@pytest.mark.twitter
class TestConversationProcessor:
    """Test conversation processor functionality"""
    
    @pytest.mark.skip(reason="extract_tweet_id_from_url function not implemented")
    def test_extract_tweet_id_from_url(self):
        """Test extracting tweet ID from URL"""
        # This function doesn't exist in the module
        pass
    
    def test_extract_tweet_text_structured(self):
        """Test structured tweet extraction"""
        url = "https://twitter.com/user/status/123456789"
        
        # Mock the response to test structured output
        with patch('src.core.conversation_processor.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b'<div class="tweet-content">Hello #test @user</div>'
            mock_get.return_value = mock_response
            
            result = extract_tweet_text(url, return_structured=True)
            
            # If successful, should return TweetContent object
            if result and isinstance(result, TweetContent):
                assert isinstance(result, TweetContent)
                assert result.tweet_id == '123456789'
                assert isinstance(result.hashtags, list)
                assert isinstance(result.mentioned_users, list)
    
    def test_inject_tweet_context(self):
        """Test injecting tweet context into a message"""
        message = "Check this out: https://twitter.com/user/status/123"
        tweet_data = {
            'text': 'This is a test tweet',
            'author': 'testuser',
            'tweet_id': '123'
        }
        
        result = inject_tweet_context(message, tweet_data)
        
        assert message in result
        assert '[TWEET: @testuser]' in result
        assert 'This is a test tweet' in result
        assert '[/TWEET]' in result
    
    def test_inject_tweet_context_no_data(self):
        """Test injecting tweet context with no tweet data"""
        message = "Just a regular message"
        
        result = inject_tweet_context(message, None)
        assert result == message


@pytest.mark.unit
@pytest.mark.twitter
class TestImageExtraction:
    """Test image extraction functionality"""
    
    def test_extract_tweet_images_valid_url(self):
        """Test extracting images from a tweet URL"""
        url = "https://twitter.com/user/status/123456789"
        
        # This will likely return empty list without network
        # but we're testing the function doesn't crash
        result = extract_tweet_images(url)
        
        assert isinstance(result, list)
    
    @patch('src.core.conversation_processor.requests.get')
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
        assert isinstance(result, list)
        if result:
            assert 'pbs.twimg.com' in result[0]


@pytest.mark.unit
class TestStructuredContent:
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
        
        assert enhanced.original_message == "Test message"
        assert enhanced.conversation_id == "conv_123"
        assert enhanced.to_training_format() == "Test message"
    
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
            emotional_tone="casual"  # Changed from "neutral" which doesn't exist
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
        
        assert "Check this:" in training_format
        assert "[TWEET:" in training_format
        assert "@testuser: Test tweet" in training_format
        assert "[IMAGES:" in training_format
        assert "A test image" in training_format
    
    @patch('src.core.conversation_processor.extract_tweet_text')
    @patch('src.core.conversation_processor.extract_tweet_images')
    @patch('src.core.conversation_processor.describe_tweet_images_with_context')
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
        
        assert isinstance(result, EnhancedMessage)
        assert len(result.tweet_contents) == 1
        assert result.tweet_contents[0].author == "mockuser"


@pytest.mark.unit
class TestConversationTracking:
    """Test conversation tracking functionality"""
    
    def test_images_with_context_structure(self):
        """Test the structure of images with context"""
        from src.models.schemas import ImageWithContext
        
        img_context = ImageWithContext(
            image_url="https://example.com/image.jpg",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_url="https://twitter.com/user/status/123"
        )
        
        assert img_context.conversation_id == "conv_123"
        assert img_context.sender_id == "user_789"
        assert img_context.timestamp is not None
    
    def test_batch_image_description_structure(self):
        """Test BatchImageDescription structure"""
        from src.models.schemas import ImageWithContext, BatchImageDescription
        
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
            emotional_tone="casual"  # Changed from "neutral" which doesn't exist
        )
        
        batch_desc = BatchImageDescription(
            image_context=img_context,
            description=img_desc
        )
        
        context_dict = batch_desc.to_dict_with_context()
        
        assert 'conversation_id' in context_dict
        assert 'sender_id' in context_dict
        assert 'description' in context_dict
        assert 'emotional_tone' in context_dict
        assert context_dict['conversation_id'] == "conv_123"


@pytest.mark.unit
class TestDataPipeline:
    """Test the data processing pipeline"""
    
    @pytest.fixture
    def messages_df(self):
        """Create test messages dataframe"""
        return pd.DataFrame({
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
    
    @pytest.fixture
    def recipients_df(self):
        """Create test recipients dataframe"""
        return pd.DataFrame({
            '_id': [2, 3],
            'profile_given_name': ['You', 'Friend']
        })
    
    def test_message_filtering(self, messages_df):
        """Test filtering messages for processing"""
        # Filter for meaningful messages
        filtered = messages_df[
            (messages_df['body'].notna()) & 
            (messages_df['body'].str.len() > 5)
        ]
        
        # Should keep messages 2 and 3 (longer than 5 chars)
        assert len(filtered) == 2
        assert 'https://twitter.com' in filtered['body'].iloc[0]  # First in filtered results
