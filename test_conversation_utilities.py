#!/usr/bin/env python3
"""
Test script for conversation utilities
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from conversation_utilities import (
    extract_tweet_text,
    extract_tweet_images,
    inject_tweet_context,
    process_message_with_twitter_content
)

def test_tweet_extraction():
    """Test tweet text extraction"""
    print("Testing Tweet Text Extraction")
    print("=" * 50)
    
    # Test URLs
    test_urls = [
        "https://twitter.com/greenTetra_/status/1778114292983710193?t=fu4RSzGOZgxn24VL9HbmMw",
        "https://t.co/abc123",  # Short URL
        "not a twitter url"
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        result = extract_tweet_text(url)
        if result:
            print(f"  ✓ Author: @{result['author']}")
            print(f"  ✓ Tweet ID: {result['tweet_id']}")
            print(f"  ✓ Text: {result['text'][:100]}...")
        else:
            print(f"  ✗ Failed to extract")


def test_tweet_injection():
    """Test tweet content injection"""
    print("\n\nTesting Tweet Content Injection")
    print("=" * 50)
    
    # Test message
    message = "Have you seen this? https://twitter.com/example/status/123456"
    
    # Mock tweet data
    mock_tweet = {
        'text': 'This is an example tweet about AI and machine learning!',
        'author': 'example_user',
        'tweet_id': '123456'
    }
    
    enhanced = inject_tweet_context(message, mock_tweet)
    print(f"Original: {message}")
    print(f"Enhanced: {enhanced}")


def test_full_processing():
    """Test full message processing"""
    print("\n\nTesting Full Message Processing")
    print("=" * 50)
    
    # Test with multiple URLs
    message = """
    Check out these interesting tweets:
    https://twitter.com/user1/status/111111
    and also this one: https://x.com/user2/status/222222
    What do you think?
    """
    
    print("Original message:")
    print(message)
    
    print("\nProcessed message (without images):")
    processed = process_message_with_twitter_content(message, use_images=False)
    print(processed)


def test_image_extraction():
    """Test image URL extraction"""
    print("\n\nTesting Image URL Extraction")
    print("=" * 50)
    
    test_url = "https://twitter.com/example/status/123456789"
    print(f"Testing: {test_url}")
    
    images = extract_tweet_images(test_url)
    if images:
        print(f"Found {len(images)} images:")
        for i, img in enumerate(images):
            print(f"  {i+1}. {img}")
    else:
        print("No images found (this is normal if the tweet has no images)")


if __name__ == "__main__":
    print("Conversation Utilities Test Suite")
    print("================================\n")
    
    test_tweet_extraction()
    test_tweet_injection()
    test_full_processing()
    test_image_extraction()
    
    print("\n\nAll tests completed!")
    print("\nNote: Some tests may fail if:")
    print("- Nitter instances are down")
    print("- Twitter URLs are invalid or deleted")
    print("- Network connectivity issues")