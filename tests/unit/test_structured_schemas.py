"""
Unit tests for structured_schemas module
"""

import unittest
from datetime import datetime
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.schemas import (
    EmotionalTone, Sentiment, ImageDescription, TweetContent,
    ImageWithContext, BatchImageDescription, EnhancedMessage,
    generate_json_schema
)


class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_emotional_tone_values(self):
        """Test EmotionalTone enum values"""
        self.assertEqual(EmotionalTone.POSITIVE, "positive")
        self.assertEqual(EmotionalTone.NEGATIVE, "negative")
        self.assertEqual(EmotionalTone.NEUTRAL, "neutral")
        self.assertEqual(EmotionalTone.HUMOROUS, "humorous")
        
        # Test all values are strings
        for tone in EmotionalTone:
            self.assertIsInstance(tone.value, str)
    
    def test_sentiment_values(self):
        """Test Sentiment enum values"""
        self.assertEqual(Sentiment.POSITIVE, "positive")
        self.assertEqual(Sentiment.NEGATIVE, "negative")
        self.assertEqual(Sentiment.NEUTRAL, "neutral")
        self.assertEqual(Sentiment.MIXED, "mixed")


class TestImageDescription(unittest.TestCase):
    """Test ImageDescription model"""
    
    def test_image_description_creation(self):
        """Test creating an ImageDescription"""
        desc = ImageDescription(
            description="A beautiful sunset over the ocean",
            detected_text="Sunset 2024",
            main_subjects=["sunset", "ocean", "sky"],
            emotional_tone=EmotionalTone.POSITIVE
        )
        
        self.assertEqual(desc.description, "A beautiful sunset over the ocean")
        self.assertEqual(desc.detected_text, "Sunset 2024")
        self.assertEqual(len(desc.main_subjects), 3)
        self.assertEqual(desc.emotional_tone, EmotionalTone.POSITIVE)
    
    def test_image_description_minimal(self):
        """Test creating ImageDescription with minimal data"""
        desc = ImageDescription(
            description="Test image",
            main_subjects=[],
            emotional_tone=EmotionalTone.NEUTRAL
        )
        
        self.assertIsNone(desc.detected_text)
        self.assertEqual(len(desc.main_subjects), 0)
    
    def test_to_training_format(self):
        """Test converting to training format"""
        desc = ImageDescription(
            description="A cat sleeping",
            detected_text="Sweet dreams",
            main_subjects=["cat", "bed"],
            emotional_tone=EmotionalTone.POSITIVE
        )
        
        training_format = desc.to_training_format()
        
        self.assertIn("A cat sleeping", training_format)
        self.assertIn("Text in image: Sweet dreams", training_format)
        self.assertIn("Shows: cat, bed", training_format)


class TestTweetContent(unittest.TestCase):
    """Test TweetContent model"""
    
    def test_tweet_content_creation(self):
        """Test creating TweetContent"""
        tweet = TweetContent(
            text="Just shipped a new feature! #coding #python @teammate",
            author="developer",
            tweet_id="123456789",
            mentioned_users=["teammate"],
            hashtags=["coding", "python"],
            sentiment=Sentiment.POSITIVE
        )
        
        self.assertEqual(tweet.author, "developer")
        self.assertEqual(len(tweet.hashtags), 2)
        self.assertEqual(tweet.mentioned_users[0], "teammate")
    
    def test_tweet_content_defaults(self):
        """Test TweetContent with default values"""
        tweet = TweetContent(
            text="Simple tweet",
            author="user",
            tweet_id="123"
        )
        
        self.assertEqual(tweet.sentiment, Sentiment.NEUTRAL)
        self.assertEqual(len(tweet.mentioned_users), 0)
        self.assertEqual(len(tweet.hashtags), 0)
    
    def test_tweet_to_training_format(self):
        """Test converting tweet to training format"""
        tweet = TweetContent(
            text="Check out this article",
            author="techwriter",
            tweet_id="789",
            hashtags=["tech", "AI"]
        )
        
        training_format = tweet.to_training_format()
        
        self.assertIn("@techwriter:", training_format)
        self.assertIn("Check out this article", training_format)
        self.assertIn("(tags: tech, AI)", training_format)


class TestImageWithContext(unittest.TestCase):
    """Test ImageWithContext model"""
    
    def test_image_with_context_creation(self):
        """Test creating ImageWithContext"""
        timestamp = datetime.now()
        
        img_context = ImageWithContext(
            image_url="https://example.com/image.jpg",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=timestamp,
            tweet_url="https://twitter.com/user/status/123"
        )
        
        self.assertEqual(img_context.conversation_id, "conv_123")
        self.assertEqual(img_context.sender_id, "user_789")
        self.assertEqual(img_context.timestamp, timestamp)
        self.assertEqual(img_context.tweet_url, "https://twitter.com/user/status/123")
    
    def test_image_with_context_no_tweet(self):
        """Test ImageWithContext without tweet URL"""
        img_context = ImageWithContext(
            image_url="https://example.com/image.jpg",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now()
        )
        
        self.assertIsNone(img_context.tweet_url)


class TestBatchImageDescription(unittest.TestCase):
    """Test BatchImageDescription model"""
    
    def test_batch_image_description(self):
        """Test creating BatchImageDescription"""
        img_context = ImageWithContext(
            image_url="https://example.com/test.jpg",
            conversation_id="conv_001",
            message_id="msg_001",
            sender_id="alice",
            timestamp=datetime.now()
        )
        
        img_desc = ImageDescription(
            description="A test image",
            detected_text="Test",
            main_subjects=["test"],
            emotional_tone=EmotionalTone.NEUTRAL
        )
        
        batch = BatchImageDescription(
            image_context=img_context,
            description=img_desc
        )
        
        self.assertEqual(batch.image_context.conversation_id, "conv_001")
        self.assertEqual(batch.description.description, "A test image")
    
    def test_to_dict_with_context(self):
        """Test converting to dictionary with full context"""
        timestamp = datetime.now()
        
        img_context = ImageWithContext(
            image_url="https://example.com/test.jpg",
            conversation_id="conv_001",
            message_id="msg_001",
            sender_id="alice",
            timestamp=timestamp,
            tweet_url="https://twitter.com/user/status/123"
        )
        
        img_desc = ImageDescription(
            description="A beautiful landscape",
            detected_text=None,
            main_subjects=["mountains", "lake"],
            emotional_tone=EmotionalTone.POSITIVE
        )
        
        batch = BatchImageDescription(
            image_context=img_context,
            description=img_desc
        )
        
        context_dict = batch.to_dict_with_context()
        
        # Check all required fields are present
        self.assertIn('conversation_id', context_dict)
        self.assertIn('message_id', context_dict)
        self.assertIn('sender_id', context_dict)
        self.assertIn('timestamp', context_dict)
        self.assertIn('image_url', context_dict)
        self.assertIn('description', context_dict)
        self.assertIn('main_subjects', context_dict)
        self.assertIn('emotional_tone', context_dict)
        self.assertIn('formatted_description', context_dict)
        
        # Check values
        self.assertEqual(context_dict['conversation_id'], "conv_001")
        self.assertEqual(context_dict['sender_id'], "alice")
        self.assertEqual(len(context_dict['main_subjects']), 2)


class TestEnhancedMessage(unittest.TestCase):
    """Test EnhancedMessage model"""
    
    def test_enhanced_message_creation(self):
        """Test creating EnhancedMessage"""
        timestamp = datetime.now()
        
        tweet = TweetContent(
            text="Test tweet",
            author="user",
            tweet_id="123"
        )
        
        img_desc = ImageDescription(
            description="Test image",
            main_subjects=["test"],
            emotional_tone=EmotionalTone.NEUTRAL
        )
        
        enhanced = EnhancedMessage(
            original_message="Check this out: https://twitter.com/user/status/123",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=timestamp,
            tweet_contents=[tweet],
            image_descriptions=[img_desc]
        )
        
        self.assertEqual(len(enhanced.tweet_contents), 1)
        self.assertEqual(len(enhanced.image_descriptions), 1)
    
    def test_enhanced_message_to_training_format(self):
        """Test converting enhanced message to training format"""
        enhanced = EnhancedMessage(
            original_message="Look at this",
            conversation_id="conv_123",
            message_id="msg_456",
            sender_id="user_789",
            timestamp=datetime.now(),
            tweet_contents=[
                TweetContent(
                    text="Amazing news!",
                    author="newsbot",
                    tweet_id="123"
                )
            ],
            image_descriptions=[
                ImageDescription(
                    description="Breaking news graphic",
                    main_subjects=["news"],
                    emotional_tone=EmotionalTone.SERIOUS
                )
            ]
        )
        
        training_format = enhanced.to_training_format()
        
        self.assertIn("Look at this", training_format)
        self.assertIn("[TWEET:", training_format)
        self.assertIn("@newsbot: Amazing news!", training_format)
        self.assertIn("[IMAGES:", training_format)
        self.assertIn("Breaking news graphic", training_format)


class TestJSONSchemaGeneration(unittest.TestCase):
    """Test JSON schema generation"""
    
    def test_generate_json_schema(self):
        """Test generating JSON schema from model"""
        schema = generate_json_schema(ImageDescription)
        
        self.assertIn('type', schema)
        self.assertIn('json_schema', schema)
        self.assertEqual(schema['type'], 'json_schema')
        
        json_schema = schema['json_schema']
        self.assertIn('name', json_schema)
        self.assertIn('schema', json_schema)
        self.assertTrue(json_schema.get('strict', False))
        
        # Check schema structure
        inner_schema = json_schema['schema']
        self.assertIn('properties', inner_schema)
        self.assertIn('required', inner_schema)
    
    def test_schema_additional_properties(self):
        """Test that additionalProperties is set to false"""
        schema = generate_json_schema(TweetContent)
        inner_schema = schema['json_schema']['schema']
        
        # Should have additionalProperties: false for strict mode
        self.assertEqual(inner_schema.get('additionalProperties', True), False)


if __name__ == '__main__':
    unittest.main(verbosity=2)