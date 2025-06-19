#!/usr/bin/env python3
"""
Test script for structured outputs with conversation tracking
"""

import sys
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from conversation_utilities import (
    extract_tweet_text,
    describe_tweet_images_with_context,
    process_message_with_structured_content
)
from structured_schemas import TweetContent, ImageDescription, EnhancedMessage
from datetime import datetime
import json

def test_structured_tweet_extraction():
    """Test structured tweet extraction"""
    print("Testing Structured Tweet Extraction")
    print("=" * 50)
    
    test_urls = [
        "https://twitter.com/greenTetra_/status/1778114292983710193",
        "https://x.com/example/status/123456789"
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        
        # Get structured data
        structured = extract_tweet_text(url, return_structured=True)
        
        if structured and isinstance(structured, TweetContent):
            print(f"  ✓ Author: @{structured.author}")
            print(f"  ✓ Tweet ID: {structured.tweet_id}")
            print(f"  ✓ Text: {structured.text[:100]}...")
            print(f"  ✓ Mentions: {structured.mentioned_users}")
            print(f"  ✓ Hashtags: {structured.hashtags}")
            print(f"  ✓ Sentiment: {structured.sentiment}")
            print(f"  ✓ Training format: {structured.to_training_format()[:100]}...")
        else:
            print(f"  ✗ Failed to extract structured data")


def test_batch_image_processing():
    """Test batch image processing with context"""
    print("\n\nTesting Batch Image Processing with Context")
    print("=" * 50)
    
    # Create test images with context
    test_images = [
        {
            'image_url': 'https://pbs.twimg.com/media/test1.jpg',
            'conversation_id': 'conv_001',
            'message_id': 'msg_001',
            'sender_id': 'user_alice',
            'timestamp': datetime.now(),
            'tweet_url': 'https://twitter.com/example/status/111'
        },
        {
            'image_url': 'https://pbs.twimg.com/media/test2.jpg',
            'conversation_id': 'conv_001',
            'message_id': 'msg_002',
            'sender_id': 'user_alice',
            'timestamp': datetime.now(),
            'tweet_url': 'https://twitter.com/example/status/111'
        },
        {
            'image_url': 'https://pbs.twimg.com/media/test3.jpg',
            'conversation_id': 'conv_002',
            'message_id': 'msg_003',
            'sender_id': 'user_bob',
            'timestamp': datetime.now(),
            'tweet_url': 'https://twitter.com/example/status/222'
        }
    ]
    
    print(f"Test scenario:")
    print(f"  - 3 images total")
    print(f"  - 2 images from conversation 'conv_001' (user_alice)")
    print(f"  - 1 image from conversation 'conv_002' (user_bob)")
    print(f"  - Images from 2 different tweets")
    
    # Note: This would actually call the API if OPENAI_API_KEY is set
    # For testing, we'll just show the structure
    print("\nImage context structure:")
    for img in test_images:
        print(f"\n  Image: {img['image_url']}")
        print(f"    Conversation: {img['conversation_id']}")
        print(f"    Sender: {img['sender_id']}")
        print(f"    Tweet: {img['tweet_url']}")


def test_enhanced_message_processing():
    """Test processing a complete message with structured content"""
    print("\n\nTesting Enhanced Message Processing")
    print("=" * 50)
    
    test_message = "Hey, check out this thread: https://twitter.com/example/status/123456 Really interesting discussion!"
    
    print(f"Original message: {test_message}")
    
    # Create enhanced message
    enhanced = process_message_with_structured_content(
        message=test_message,
        conversation_id="test_conv_003",
        message_id="test_msg_001",
        sender_id="test_user_123",
        timestamp=datetime.now(),
        use_images=False  # Skip image processing for test
    )
    
    print(f"\nEnhanced Message Structure:")
    print(f"  Original: {enhanced.original_message}")
    print(f"  Conversation ID: {enhanced.conversation_id}")
    print(f"  Message ID: {enhanced.message_id}")
    print(f"  Sender ID: {enhanced.sender_id}")
    print(f"  Tweet contents: {len(enhanced.tweet_contents)} tweets")
    print(f"  Image descriptions: {len(enhanced.image_descriptions)} images")
    
    if enhanced.tweet_contents:
        print(f"\n  Tweet Details:")
        for tweet in enhanced.tweet_contents:
            print(f"    - @{tweet.author}: {tweet.text[:50]}...")
    
    print(f"\nTraining format output:")
    print(enhanced.to_training_format())


def test_conversation_tracking():
    """Demonstrate conversation tracking benefits"""
    print("\n\nDemonstrating Conversation Tracking Benefits")
    print("=" * 50)
    
    print("\nScenario: Multiple people share the same tweet in different conversations")
    print("\nWithout conversation tracking:")
    print("  - All images processed together")
    print("  - Lost context of who shared what")
    print("  - Can't maintain conversation boundaries")
    
    print("\nWith conversation tracking:")
    print("  - Each image maintains its conversation context")
    print("  - Know exactly who shared each image")
    print("  - Can group training data by conversation")
    print("  - Preserve temporal ordering within conversations")
    
    # Example tracking data
    example_tracking = {
        'image_1': {
            'conversation_id': 'family_chat',
            'sender': 'mom',
            'timestamp': '2024-01-15 10:30:00',
            'description': 'A photo of a sunset'
        },
        'image_2': {
            'conversation_id': 'work_chat',
            'sender': 'colleague',
            'timestamp': '2024-01-15 14:45:00',
            'description': 'The same sunset photo'
        }
    }
    
    print("\nExample tracking data:")
    print(json.dumps(example_tracking, indent=2))
    
    print("\nThis allows training data like:")
    print("  'In family chat, mom shares sunset photos'")
    print("  'In work chat, colleagues share nature photography'")


if __name__ == "__main__":
    print("Structured Output Test Suite")
    print("================================\n")
    
    test_structured_tweet_extraction()
    test_batch_image_processing()
    test_enhanced_message_processing()
    test_conversation_tracking()
    
    print("\n\nAll tests completed!")
    print("\nNote: Some tests require API keys to fully function:")
    print("- Set OPENAI_API_KEY for image processing")
    print("- Twitter extraction works without API keys using Nitter")