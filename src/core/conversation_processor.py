"""
Conversation Utilities for Astrabot

This module provides utilities for enhancing conversation data with external content,
particularly Twitter/X posts. It includes functions for:

1. Extracting tweet text and metadata from Twitter/X URLs
2. Extracting image URLs from tweets
3. Describing images using vision AI APIs (OpenAI, Anthropic)
4. Injecting tweet content into messages with clear formatting
5. Processing messages to automatically enhance Twitter links

The module is designed to work with Signal chat exports and improve the quality
of training data for fine-tuning language models.
"""

import json
import os
import re
import tempfile
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.models.conversation_schemas import (
    ConversationDynamics,
    ConversationThread,
    ConversationWindow,
    MessageTiming,
    RelationshipDynamic,
    SignalMessage,
)
from src.models.schemas import (
    IMAGE_DESCRIPTION_SCHEMA,
    BatchImageDescription,
    BurstSequence,
    ConversationMood,
    EmotionalTone,
    EnhancedMessage,
    ImageDescription,
    ImageWithContext,
    MessageType,
    Sentiment,
    TopicCategory,
    TopicTransition,
    TweetContent,
)
from src.utils.logging import get_logger

# Import analysis modules
try:
    from src.core.emoji_analyzer import EmojiAnalyzer
    from src.core.personality_profiler import PersonalityProfiler
    from src.core.topic_tracker import TopicTracker
except ImportError:
    # Provide stubs if modules not available
    class EmojiAnalyzer:
        def analyze_message_emoji_patterns(self, *args, **kwargs):
            return {"has_emojis": False, "emojis_list": [], "dominant_emotion": "neutral"}

    class TopicTracker:
        def detect_message_topics(self, *args, **kwargs):
            return {"primary_topic": TopicCategory.OTHER}

    class PersonalityProfiler:
        def analyze_relationship_dynamics(self, *args, **kwargs):
            return RelationshipDynamic.CLOSE_FRIENDS

        def generate_personality_profile(self, *args, **kwargs):
            return None


logger = get_logger(__name__)

# Import configuration
try:
    from src.utils.config import config
except ImportError:
    # Fallback if config.py is not available
    class Config:
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
        ENABLE_IMAGE_PROCESSING = True
        ENABLE_BATCH_PROCESSING = True
        MAX_BATCH_SIZE = 10

    config = Config()


class TwitterExtractor:
    """Handles extraction of Twitter/X content from URLs."""

    def __init__(self):
        """Initialize the Twitter extractor."""
        self.logger = get_logger(__name__)
        self.nitter_instances = [
            "nitter.privacydev.net",
            "nitter.poast.org",
            "nitter.net",
            "nitter.it",
            "nitter.unixfox.eu",
        ]
        self._response_cache = {}

    def extract_tweet_id(self, url: str) -> Optional[str]:
        """
        Extract tweet ID from a Twitter/X URL.

        Args:
            url: Twitter/X URL

        Returns:
            Tweet ID or None if not found
        """
        # Handle t.co redirects
        if "t.co" in url:
            try:
                response = requests.head(url, allow_redirects=True, timeout=10)
                url = response.url
            except:
                return None

        # Extract tweet ID
        tweet_id_match = re.search(r"/status/(\d+)", url)
        return tweet_id_match.group(1) if tweet_id_match else None

    def is_valid_twitter_url(self, url: str) -> bool:
        """
        Check if URL is a valid Twitter/X URL.

        Args:
            url: URL to check

        Returns:
            True if valid Twitter/X URL
        """
        if not url:
            return False
        return any(domain in url for domain in ["twitter.com", "x.com", "t.co"])

    def extract_tweet_content(self, url: str) -> Optional[TweetContent]:
        """
        Extract tweet content as structured data.

        Args:
            url: Twitter/X URL

        Returns:
            TweetContent object or None if extraction fails
        """
        # Check cache first
        if url in self._response_cache:
            cached_response = self._response_cache[url]
            if isinstance(cached_response, TweetContent):
                return cached_response

        # Extract tweet ID and username
        tweet_id = self.extract_tweet_id(url)
        if not tweet_id:
            return None

        username_match = re.search(r"(?:twitter\.com|x\.com)/([^/]+)/", url)
        username = username_match.group(1) if username_match else "unknown"

        # Try Nitter instances
        for instance in self.nitter_instances:
            try:
                nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(nitter_url, timeout=10, headers=headers)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Find the main tweet content
                    tweet_content = soup.find("div", class_="tweet-content")
                    if tweet_content:
                        # Extract text, removing extra whitespace
                        tweet_text = tweet_content.get_text().strip()
                        tweet_text = " ".join(tweet_text.split())

                        # Extract hashtags and mentions
                        hashtags = self.parse_hashtags(tweet_text)
                        mentions = self.parse_mentions(tweet_text)

                        # Detect thread/retweet
                        is_thread = bool(soup.find(class_="show-this-thread"))
                        is_retweet = bool(soup.find(class_="retweet-header"))

                        tweet_obj = TweetContent(
                            text=tweet_text,
                            author=username,
                            tweet_id=tweet_id,
                            mentioned_users=mentions,
                            hashtags=hashtags,
                            sentiment=Sentiment.NEUTRAL,  # Default
                            is_thread=is_thread,
                            is_retweet=is_retweet,
                        )

                        # Cache the result
                        self._response_cache[url] = tweet_obj
                        return tweet_obj
            except Exception as e:
                self.logger.debug(f"Failed to fetch from {instance}: {e}")
                continue

        return None

    def extract_tweet_images(self, url: str) -> list[str]:
        """
        Extract image URLs from a tweet.

        Args:
            url: Twitter/X URL

        Returns:
            List of image URLs
        """
        # Extract tweet ID and username
        tweet_id = self.extract_tweet_id(url)
        if not tweet_id:
            return []

        username_match = re.search(r"(?:twitter\.com|x\.com)/([^/]+)/", url)
        username = username_match.group(1) if username_match else "unknown"

        image_urls = []

        # Try Nitter instances
        for instance in self.nitter_instances:
            try:
                nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(nitter_url, timeout=10, headers=headers)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Find images in the main tweet
                    for img_link in soup.find_all("a", class_="still-image"):
                        img_url = img_link.get("href")
                        if img_url:
                            # Convert Nitter image URL to direct Twitter image URL
                            twitter_url = self.convert_nitter_to_twitter_image_url(img_url)
                            image_urls.append(twitter_url)

                    # Also check for img tags within tweet content
                    tweet_content = soup.find("div", class_="tweet-content")
                    if tweet_content:
                        for img in tweet_content.find_all("img"):
                            src = img.get("src")
                            if src and "twimg.com" in src:
                                image_urls.append(src)

                    if image_urls:
                        return image_urls
            except Exception as e:
                self.logger.debug(f"Failed to fetch from {instance}: {e}")
                continue

        return image_urls

    def parse_hashtags(self, text: str) -> list[str]:
        """
        Extract hashtags from text.

        Args:
            text: Text to parse

        Returns:
            List of hashtags (without # symbol)
        """
        hashtags = re.findall(r"#(\w+)", text)
        return list(set(hashtags))

    def parse_mentions(self, text: str) -> list[str]:
        """
        Extract mentions from text.

        Args:
            text: Text to parse

        Returns:
            List of usernames (without @ symbol)
        """
        mentions = re.findall(r"@(\w+)", text)
        return list(set(mentions))

    def analyze_sentiment(self, text: str) -> str:
        """
        Basic sentiment analysis.

        Args:
            text: Text to analyze

        Returns:
            'positive', 'negative', or 'neutral'
        """
        # Basic sentiment keywords
        positive_indicators = [
            "love",
            "great",
            "awesome",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "brilliant",
            "perfect",
            "😊",
            "😄",
            "❤️",
            "👍",
            "🎉",
        ]
        negative_indicators = [
            "hate",
            "terrible",
            "awful",
            "horrible",
            "bad",
            "worst",
            "disappointed",
            "angry",
            "sad",
            "😢",
            "😡",
            "👎",
            "😠",
        ]

        text_lower = text.lower()
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)

        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"

    def convert_nitter_to_twitter_image_url(self, nitter_url: str) -> str:
        """
        Convert Nitter image URL to Twitter image URL.

        Args:
            nitter_url: Nitter image URL

        Returns:
            Twitter image URL
        """
        # Extract image filename from Nitter URL
        img_match = re.search(r"/pic/media%2F([^?]+)", nitter_url)
        if img_match:
            img_filename = img_match.group(1)
            return f"https://pbs.twimg.com/media/{img_filename}"
        return nitter_url


def extract_tweet_text(
    url: str, return_structured: bool = False
) -> Optional[dict[str, str] | TweetContent]:
    """
    Extract only the main tweet text from a Twitter/X URL.

    Args:
        url: Twitter/X URL (supports twitter.com, x.com, and t.co)
        return_structured: If True, return TweetContent object instead of dict

    Returns:
        Dict with 'text', 'author', 'tweet_id' or TweetContent object, or None if extraction fails
    """
    # Clean and normalize URL
    if "t.co" in url:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            url = response.url
        except:
            return None

    # Convert x.com to twitter.com for consistency
    url = url.replace("x.com", "twitter.com")

    # Extract tweet ID
    tweet_id_match = re.search(r"/status/(\d+)", url)
    if not tweet_id_match:
        return None

    tweet_id = tweet_id_match.group(1)
    username_match = re.search(r"twitter\.com/([^/]+)/", url)
    username = username_match.group(1) if username_match else "unknown"

    # Try Nitter instances first (privacy-friendly, no API needed)
    nitter_instances = [
        "nitter.privacydev.net",
        "nitter.poast.org",
        "nitter.net",
        "nitter.it",
        "nitter.unixfox.eu",
    ]

    for instance in nitter_instances:
        try:
            nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(nitter_url, timeout=10, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # Find the main tweet content
                tweet_content = soup.find("div", class_="tweet-content")
                if tweet_content:
                    # Extract text, removing extra whitespace
                    tweet_text = tweet_content.get_text().strip()
                    tweet_text = " ".join(tweet_text.split())

                    # Extract hashtags and mentions if structured output requested
                    if return_structured:
                        hashtags = re.findall(r"#\w+", tweet_text)
                        mentions = re.findall(r"@\w+", tweet_text)

                        return TweetContent(
                            text=tweet_text,
                            author=username,
                            tweet_id=tweet_id,
                            mentioned_users=[m[1:] for m in mentions],  # Remove @ symbol
                            hashtags=[h[1:] for h in hashtags],  # Remove # symbol
                            sentiment=Sentiment.NEUTRAL,  # Default, could be enhanced
                        )
                    else:
                        return {"text": tweet_text, "author": username, "tweet_id": tweet_id}
        except Exception:
            continue

    # Fallback: Try direct Twitter scraping (less reliable)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; bot)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Look for tweet text in meta tags (often available)
            soup = BeautifulSoup(response.content, "html.parser")
            meta_desc = soup.find("meta", {"property": "og:description"})
            if meta_desc and meta_desc.get("content"):
                tweet_text = meta_desc["content"]

                if return_structured:
                    hashtags = re.findall(r"#\w+", tweet_text)
                    mentions = re.findall(r"@\w+", tweet_text)

                    return TweetContent(
                        text=tweet_text,
                        author=username,
                        tweet_id=tweet_id,
                        mentioned_users=[m[1:] for m in mentions],
                        hashtags=[h[1:] for h in hashtags],
                        sentiment=Sentiment.NEUTRAL,
                    )
                else:
                    return {"text": tweet_text, "author": username, "tweet_id": tweet_id}
    except:
        pass

    return None


def inject_tweet_context(message_with_url: str, tweet_data: Optional[dict[str, str]]) -> str:
    """
    Inject tweet text into the message with clear markers.

    Args:
        message_with_url: Original message containing the Twitter URL
        tweet_data: Extracted tweet data from extract_tweet_text()

    Returns:
        Message with tweet content injected, or original message if no data
    """
    if not tweet_data or not tweet_data.get("text"):
        return message_with_url

    # Format: Keep original message but add tweet content clearly marked
    tweet_section = f"\n\n[TWEET: @{tweet_data['author']}]\n{tweet_data['text']}\n[/TWEET]"

    return message_with_url + tweet_section


def extract_tweet_images(url: str) -> list[str]:
    """
    Extract image URLs from the main tweet only.

    Args:
        url: Twitter/X URL

    Returns:
        List of direct image URLs (not thumbnails)
    """
    image_urls = []

    # Normalize URL
    if "t.co" in url:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            url = response.url
        except:
            return []

    url = url.replace("x.com", "twitter.com")

    # Extract tweet ID and username
    tweet_id_match = re.search(r"/status/(\d+)", url)
    if not tweet_id_match:
        return []

    tweet_id = tweet_id_match.group(1)
    username_match = re.search(r"twitter\.com/([^/]+)/", url)
    username = username_match.group(1) if username_match else "unknown"

    # Try Nitter instances
    nitter_instances = ["nitter.privacydev.net", "nitter.poast.org", "nitter.net", "nitter.it"]

    for instance in nitter_instances:
        try:
            nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(nitter_url, timeout=10, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # Find images in the main tweet
                # Nitter usually puts images in <a> tags with class 'still-image'
                for img_link in soup.find_all("a", class_="still-image"):
                    img_url = img_link.get("href")
                    if img_url:
                        # Convert Nitter image URL to direct Twitter image URL
                        if "/pic/" in img_url:
                            # Extract the image filename
                            img_match = re.search(r"/pic/([^?]+)", img_url)
                            if img_match:
                                img_filename = img_match.group(1)
                                # Construct direct Twitter image URL
                                direct_url = f"https://pbs.twimg.com/media/{img_filename}"
                                image_urls.append(direct_url)

                # Also check for img tags within tweet content
                tweet_content = soup.find("div", class_="tweet-content")
                if tweet_content:
                    for img in tweet_content.find_all("img"):
                        src = img.get("src")
                        if src and "twimg.com" in src:
                            image_urls.append(src)

                if image_urls:
                    return image_urls
        except:
            continue

    return image_urls


def describe_tweet_images(
    image_urls: list[str],
    api_endpoint: str,
    api_key: Optional[str] = None,
    batch_process: bool = True,
) -> list[str]:
    """
    Send images to AI vision API for description.

    Args:
        image_urls: List of image URLs to describe
        api_endpoint: API endpoint type ('openai', 'anthropic', or full URL for custom)
        api_key: API key (will try to get from environment if not provided)
        batch_process: If True and using OpenAI, process multiple images in one API call for cost savings

    Returns:
        List of image descriptions
    """
    descriptions = []

    if not image_urls:
        return descriptions

    # Get API key from config if not provided
    if not api_key:
        if api_endpoint == "openai":
            api_key = config.OPENAI_API_KEY
        elif api_endpoint == "anthropic":
            api_key = config.ANTHROPIC_API_KEY

    # Batch process with OpenAI if enabled and multiple images
    if batch_process and api_endpoint == "openai" and len(image_urls) > 1 and api_key:
        return _batch_describe_with_openai(image_urls, api_key)

    # Process individually (original behavior)
    for img_url in image_urls:
        try:
            # Download image temporarily
            img_response = requests.get(img_url, timeout=10)
            if img_response.status_code != 200:
                descriptions.append("Failed to download image")
                continue

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_file.write(img_response.content)
                tmp_path = tmp_file.name

            try:
                # Send to appropriate API
                if api_endpoint == "openai" and api_key:
                    desc_result = _describe_with_openai(tmp_path, api_key, use_structured=True)
                    # Convert ImageDescription to string for backward compatibility
                    if isinstance(desc_result, ImageDescription):
                        description = desc_result.to_training_format()
                    else:
                        description = desc_result
                elif api_endpoint == "anthropic" and api_key:
                    description = _describe_with_anthropic(tmp_path, api_key)
                else:
                    # Custom endpoint - implement as needed
                    description = "Custom API endpoint not implemented"

                descriptions.append(description)
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            descriptions.append(f"Error processing image: {str(e)}")

    return descriptions


def describe_tweet_images_with_context(
    images_with_context: list[dict[str, any]], api_key: Optional[str] = None
) -> list[BatchImageDescription]:
    """
    Process images while maintaining conversation context.

    Args:
        images_with_context: List of dicts with:
            - image_url: str
            - conversation_id: str
            - message_id: str
            - sender_id: str
            - timestamp: datetime
            - tweet_url: Optional[str]
        api_key: OpenAI API key (will try to get from environment if not provided)

    Returns:
        List of BatchImageDescription objects with context preserved
    """
    if not images_with_context:
        return []

    # Get API key from config if not provided
    if not api_key:
        api_key = config.OPENAI_API_KEY

    if not api_key:
        raise ValueError("OpenAI API key required for image description")

    import base64

    # Build content array with all images and track their indices
    content = [
        {
            "type": "text",
            "text": f"Please describe each of these {len(images_with_context)} images concisely. "
            f"For each image, provide a JSON object with description, detected_text, "
            f'main_subjects (array), and emotional_tone. Number each as "Image 1:", "Image 2:", etc.',
        }
    ]

    # Track mapping: index -> context
    index_to_context = {}
    valid_indices = []

    # Download and encode all images
    for i, img_context in enumerate(images_with_context):
        try:
            img_url = img_context["image_url"]
            img_response = requests.get(img_url, timeout=10)
            if img_response.status_code == 200:
                image_data = base64.b64encode(img_response.content).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low",  # Use low detail for cost optimization
                        },
                    }
                )

                # Create ImageWithContext object
                index_to_context[i] = ImageWithContext(
                    image_url=img_url,
                    conversation_id=img_context["conversation_id"],
                    message_id=img_context["message_id"],
                    sender_id=img_context["sender_id"],
                    timestamp=img_context["timestamp"],
                    tweet_url=img_context.get("tweet_url"),
                )
                valid_indices.append(i)
        except Exception as e:
            print(f"Error downloading image {i}: {str(e)}")
            continue

    if not valid_indices:
        return []

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "response_format": IMAGE_DESCRIPTION_SCHEMA,
        "max_tokens": 500 * len(valid_indices),  # More tokens for multiple images
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        if response.status_code == 200:
            results = []
            full_response = response.json()["choices"][0]["message"]["content"]

            # Parse the structured JSON response
            # The response should be a JSON array of ImageDescription objects
            try:
                descriptions_data = json.loads(full_response)
                if isinstance(descriptions_data, dict):
                    # Single image case
                    descriptions_data = [descriptions_data]

                # Map descriptions back to their contexts
                for i, desc_data in enumerate(descriptions_data):
                    if i < len(valid_indices):
                        original_index = valid_indices[i]
                        context = index_to_context[original_index]

                        # Create ImageDescription from the structured data
                        description = ImageDescription(**desc_data)

                        # Create BatchImageDescription
                        batch_desc = BatchImageDescription(
                            image_context=context, description=description
                        )
                        results.append(batch_desc)

            except json.JSONDecodeError:
                # Fallback: try to parse numbered descriptions
                results = _parse_numbered_descriptions(
                    full_response, index_to_context, valid_indices
                )

            return results
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        return []


def _parse_numbered_descriptions(
    response_text: str, index_to_context: dict[int, ImageWithContext], valid_indices: list[int]
) -> list[BatchImageDescription]:
    """Fallback parser for numbered descriptions if JSON parsing fails."""
    results = []

    # Split by "Image N:" pattern
    image_blocks = re.split(r"Image \d+:", response_text)

    for i, block in enumerate(image_blocks[1:]):  # Skip first empty split
        if i < len(valid_indices):
            original_index = valid_indices[i]
            context = index_to_context[original_index]

            # Try to extract structured data from the block
            description_match = re.search(r'"description":\s*"([^"]+)"', block)
            subjects_match = re.search(r'"main_subjects":\s*\[([^\]]+)\]', block)
            tone_match = re.search(r'"emotional_tone":\s*"([^"]+)"', block)

            description_text = description_match.group(1) if description_match else block.strip()

            subjects = []
            if subjects_match:
                subjects_str = subjects_match.group(1)
                subjects = [s.strip().strip('"') for s in subjects_str.split(",")]

            tone = tone_match.group(1) if tone_match else "neutral"

            # Create structured objects
            description = ImageDescription(
                description=description_text,
                detected_text=None,
                main_subjects=subjects,
                emotional_tone=tone,
            )

            batch_desc = BatchImageDescription(image_context=context, description=description)
            results.append(batch_desc)

    return results


def _batch_describe_with_openai(image_urls: list[str], api_key: str) -> list[str]:
    """
    Batch process multiple images in a single OpenAI API call for cost efficiency.
    Uses GPT-4o-mini with low detail setting for optimal cost/performance.

    Args:
        image_urls: List of image URLs to describe
        api_key: OpenAI API key

    Returns:
        List of image descriptions in the same order as input URLs
    """
    import base64

    # Build content array with all images
    content = [
        {
            "type": "text",
            "text": f"Please describe each of these {len(image_urls)} images concisely in 1-2 sentences. Number each description.",
        }
    ]

    # Download and encode all images
    for i, img_url in enumerate(image_urls):
        try:
            img_response = requests.get(img_url, timeout=10)
            if img_response.status_code == 200:
                image_data = base64.b64encode(img_response.content).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": "low",  # Use low detail for cost optimization
                        },
                    }
                )
            else:
                content.append({"type": "text", "text": f"Image {i+1}: Failed to download"})
        except Exception as e:
            content.append({"type": "text", "text": f"Image {i+1}: Error - {str(e)}"})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,  # Allow more tokens for multiple descriptions
    }

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )

        if response.status_code == 200:
            # Parse numbered descriptions from response
            full_response = response.json()["choices"][0]["message"]["content"]
            descriptions = []

            # Split by numbers and extract descriptions
            lines = full_response.split("\n")
            current_desc = []

            for line in lines:
                if re.match(r"^\d+\.", line.strip()):
                    if current_desc:
                        descriptions.append(" ".join(current_desc).strip())
                    current_desc = [re.sub(r"^\d+\.\s*", "", line.strip())]
                elif line.strip():
                    current_desc.append(line.strip())

            if current_desc:
                descriptions.append(" ".join(current_desc).strip())

            # Ensure we have the right number of descriptions
            while len(descriptions) < len(image_urls):
                descriptions.append("Description unavailable")

            return descriptions[: len(image_urls)]
        else:
            return [f"Batch API error: {response.status_code}"] * len(image_urls)

    except Exception as e:
        return [f"Batch processing error: {str(e)}"] * len(image_urls)


def _describe_with_openai(
    image_path: str, api_key: str, use_structured: bool = True
) -> str | ImageDescription:
    """
    Helper function for OpenAI Vision API using GPT-4o-mini.

    Args:
        image_path: Path to the image file
        api_key: OpenAI API key
        use_structured: If True, return ImageDescription object; if False, return string

    Returns:
        ImageDescription object or string description
    """
    import base64

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    content = [
        {
            "type": "text",
            "text": "Describe this image concisely. Include any visible text, identify main subjects, and note the emotional tone.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data}",
                "detail": "low",  # Use low detail for cost optimization
            },
        },
    ]

    payload = {
        "model": "gpt-4o-mini",  # Using GPT-4o-mini for cost-effective vision processing
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 150,
    }

    # Add structured output format if requested
    if use_structured:
        payload["response_format"] = IMAGE_DESCRIPTION_SCHEMA

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]

        if use_structured:
            # Parse the structured JSON response
            try:
                desc_data = json.loads(content)
                return ImageDescription(**desc_data)
            except (json.JSONDecodeError, ValueError):
                # Fallback to string if parsing fails
                return content
        else:
            return content
    else:
        error_msg = f"OpenAI API error: {response.status_code}"
        if use_structured:
            return ImageDescription(
                description=error_msg,
                detected_text=None,
                main_subjects=[],
                emotional_tone="neutral",
            )
        else:
            return error_msg


def _describe_with_anthropic(image_path: str, api_key: str) -> str:
    """Helper function for Anthropic Claude Vision API"""
    import base64

    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 150,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely in 1-2 sentences."},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                ],
            }
        ],
    }

    response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["content"][0]["text"]
    else:
        return f"Anthropic API error: {response.status_code}"


# Example usage function
def process_message_with_twitter_content(
    message: str, use_images: bool = True, image_api: str = "openai"
) -> str:
    """
    Process a message containing Twitter URLs to inject tweet content.

    Args:
        message: Message potentially containing Twitter URLs
        use_images: Whether to process and describe images
        image_api: Which API to use for image description

    Returns:
        Enhanced message with tweet content injected
    """
    # Find all Twitter URLs in the message
    twitter_patterns = [
        r"https?://(?:www\.)?twitter\.com/\S+/status/\d+",
        r"https?://(?:www\.)?x\.com/\S+/status/\d+",
        r"https?://t\.co/\S+",
    ]

    enhanced_message = message

    for pattern in twitter_patterns:
        urls = re.findall(pattern, message)

        for url in urls:
            # Extract tweet text
            tweet_data = extract_tweet_text(url)

            if tweet_data:
                # Start building the enhanced content
                enhanced_content = f"\n\n[TWEET: @{tweet_data['author']}]\n{tweet_data['text']}"

                # Process images if requested
                if use_images:
                    image_urls = extract_tweet_images(url)
                    if image_urls:
                        descriptions = describe_tweet_images(image_urls, image_api)
                        if descriptions:
                            enhanced_content += "\n[IMAGES: "
                            enhanced_content += " | ".join(descriptions)
                            enhanced_content += "]"

                enhanced_content += "\n[/TWEET]"

                # Replace the URL with URL + enhanced content
                enhanced_message = enhanced_message.replace(url, url + enhanced_content)

    return enhanced_message


def process_message_with_structured_content(
    message: str,
    conversation_id: str,
    message_id: str,
    sender_id: str,
    timestamp: datetime,
    use_images: bool = True,
    image_api: str = "openai",
) -> EnhancedMessage:
    """
    Process a message containing Twitter URLs to extract structured content.

    Args:
        message: Message potentially containing Twitter URLs
        conversation_id: ID of the conversation/thread
        message_id: Unique message ID
        sender_id: ID of the message sender
        timestamp: Message timestamp
        use_images: Whether to process and describe images
        image_api: Which API to use for image description

    Returns:
        EnhancedMessage object with structured tweet and image data
    """
    # Find all Twitter URLs in the message
    twitter_patterns = [
        r"https?://(?:www\.)?twitter\.com/\S+/status/\d+",
        r"https?://(?:www\.)?x\.com/\S+/status/\d+",
        r"https?://t\.co/\S+",
    ]

    tweet_contents = []
    image_descriptions = []

    for pattern in twitter_patterns:
        urls = re.findall(pattern, message)

        for url in urls:
            # Extract tweet content as structured data
            tweet_data = extract_tweet_text(url, return_structured=True)

            if tweet_data and isinstance(tweet_data, TweetContent):
                tweet_contents.append(tweet_data)

                # Process images if requested
                if use_images:
                    image_urls = extract_tweet_images(url)
                    if image_urls:
                        # Prepare images with context for batch processing
                        images_with_context = []
                        for img_url in image_urls:
                            images_with_context.append(
                                {
                                    "image_url": img_url,
                                    "conversation_id": conversation_id,
                                    "message_id": message_id,
                                    "sender_id": sender_id,
                                    "timestamp": timestamp,
                                    "tweet_url": url,
                                }
                            )

                        # Process images with context
                        batch_results = describe_tweet_images_with_context(images_with_context)

                        # Extract just the descriptions for the EnhancedMessage
                        for batch_result in batch_results:
                            image_descriptions.append(batch_result.description)

    # Create and return the enhanced message
    return EnhancedMessage(
        original_message=message,
        conversation_id=conversation_id,
        message_id=message_id,
        sender_id=sender_id,
        timestamp=timestamp,
        tweet_contents=tweet_contents,
        image_descriptions=image_descriptions,
    )


def preserve_conversation_dynamics(messages_df, your_recipient_id=2):
    """
    Capture and preserve different conversation modes and dynamics.

    Args:
        messages_df: DataFrame of messages
        your_recipient_id: Your recipient ID

    Returns:
        Conversation data with preserved dynamics and style patterns
    """
    conversation_dynamics = []

    # Group by thread
    for thread_id in messages_df["thread_id"].unique():
        thread_messages = messages_df[messages_df["thread_id"] == thread_id].sort_values(
            "date_sent"
        )

        # Identify your message sequences
        i = 0
        while i < len(thread_messages):
            # Find sequences where you're speaking
            if thread_messages.iloc[i]["from_recipient_id"] == your_recipient_id:
                # Collect your burst sequence
                your_sequence = [thread_messages.iloc[i].to_dict()]
                j = i + 1

                # Keep collecting while you're still talking and messages are close in time
                while j < len(thread_messages):
                    if thread_messages.iloc[j]["from_recipient_id"] == your_recipient_id:
                        time_gap = (
                            thread_messages.iloc[j]["date_sent"]
                            - thread_messages.iloc[j - 1]["date_sent"]
                        ) / 1000
                        if time_gap < 120:  # Within 2 minutes
                            your_sequence.append(thread_messages.iloc[j].to_dict())
                            j += 1
                        else:
                            break
                    else:
                        break

                # Get context before your sequence
                context_start = max(0, i - 5)
                context_messages = list(thread_messages.iloc[context_start:i].to_dict("records"))

                # Classify your conversation style for this sequence
                if len(your_sequence) >= 3:
                    style = "burst_sequence"
                elif len(your_sequence) == 1 and len(your_sequence[0]["body"]) > 200:
                    style = "long_form"
                elif len(your_sequence) == 2:
                    style = "double_tap"
                else:
                    style = "single_message"

                # Check for media sharing
                has_media = any(
                    bool(re.search(r"https?://\S+", msg["body"])) for msg in your_sequence
                )

                # Enhance messages with Twitter content if present
                enhanced_sequence = []
                for msg in your_sequence:
                    enhanced_text = process_message_with_twitter_content(
                        msg["body"],
                        use_images=True,  # Enable image processing for richer training data
                        image_api="openai",  # Use GPT-4o-mini for cost-effective vision processing
                    )
                    enhanced_msg = msg.copy()
                    enhanced_msg["body"] = enhanced_text
                    enhanced_msg["original_body"] = msg["body"]
                    enhanced_sequence.append(enhanced_msg)

                # Build the dynamics data
                dynamics_data = {
                    "thread_id": thread_id,
                    "context": [
                        {
                            "speaker": (
                                "You" if msg["from_recipient_id"] == your_recipient_id else "Other"
                            ),
                            "text": msg["body"],
                            "timestamp": msg["date_sent"],
                        }
                        for msg in context_messages
                    ],
                    "your_sequence": [
                        {
                            "text": msg["body"],
                            "original_text": msg["original_body"],
                            "timestamp": msg["date_sent"],
                            "enhanced": msg["body"] != msg["original_body"],
                        }
                        for msg in enhanced_sequence
                    ],
                    "style": style,
                    "metadata": {
                        "sequence_length": len(your_sequence),
                        "total_chars": sum(len(msg["body"]) for msg in your_sequence),
                        "has_media": has_media,
                        "avg_message_length": sum(len(msg["body"]) for msg in your_sequence)
                        / len(your_sequence),
                        "time_span_seconds": (
                            (your_sequence[-1]["date_sent"] - your_sequence[0]["date_sent"]) / 1000
                            if len(your_sequence) > 1
                            else 0
                        ),
                    },
                }

                conversation_dynamics.append(dynamics_data)
                i = j
            else:
                i += 1

    return conversation_dynamics


class EnhancedConversationProcessor:
    """Comprehensive conversation processor with advanced analysis capabilities."""

    def __init__(self):
        """Initialize the enhanced conversation processor."""
        self.emoji_analyzer = EmojiAnalyzer()
        self.topic_tracker = TopicTracker()
        self.personality_profiler = PersonalityProfiler()
        logger.info("EnhancedConversationProcessor initialized")

    def process_signal_message(
        self, row: pd.Series, context_messages: Optional[list[SignalMessage]] = None
    ) -> SignalMessage:
        """Process a single Signal message with comprehensive analysis."""

        # Extract basic message data
        message_id = str(row.get("_id", ""))
        thread_id = str(row.get("thread_id", ""))
        sender_id = str(row.get("from_recipient_id", ""))
        timestamp = pd.to_datetime(row.get("date_sent"), unit="ms")
        body = str(row.get("body", "")) if pd.notna(row.get("body")) else ""

        # Analyze message content
        emoji_analysis = self.emoji_analyzer.analyze_message_emoji_patterns(
            body, sender_id, timestamp
        )
        topic_analysis = self.topic_tracker.detect_message_topics(body)

        # Determine message type
        message_type = self._classify_message_type(body, context_messages)

        # Extract URLs and emojis
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, body)
        emojis = emoji_analysis.get("emojis_list", [])

        # Language analysis
        word_count = len(body.split()) if body else 0
        character_count = len(body)

        # Profanity detection (simple)
        profanity_patterns = [r"\bfuck\w*", r"\bshit\w*", r"\bdamn\w*", r"\bhell\w*"]
        contains_profanity = any(re.search(pattern, body.lower()) for pattern in profanity_patterns)

        # Academic language detection
        academic_indicators = [
            "analysis",
            "theory",
            "framework",
            "paradigm",
            "methodology",
            "furthermore",
            "moreover",
            "however",
            "nevertheless",
        ]
        academic_language = (
            sum(1 for indicator in academic_indicators if indicator in body.lower()) >= 2
        )

        # Internet slang detection
        slang_indicators = ["lmao", "lol", "omg", "wtf", "tbh", "ngl", "fr", "bruh"]
        internet_slang = any(slang in body.lower() for slang in slang_indicators)

        # Conversation context analysis
        response_to_message_id = None
        time_since_previous = None
        is_correction = False
        is_continuation = False

        if context_messages:
            last_message = context_messages[-1] if context_messages else None
            if last_message:
                time_since_previous = (timestamp - last_message.timestamp).total_seconds()

                # Simple correction detection
                if (
                    body.lower().startswith(last_message.body.lower()[:20])
                    and len(body) > len(last_message.body) * 0.8
                ):
                    is_correction = True
                    response_to_message_id = last_message.message_id

                # Continuation detection
                elif (
                    time_since_previous < 120  # Within 2 minutes
                    and last_message.sender_id == sender_id
                    and not body.lower().startswith(("no", "yes", "but", "however"))
                ):
                    is_continuation = True

        # Determine emotional tone
        if emoji_analysis.get("has_emojis", False):
            emotional_tone = emoji_analysis.get("dominant_emotion", EmotionalTone.CASUAL)
            # Map emoji emotions to tones
            emotion_mapping = {
                "joy_laughter": EmotionalTone.HUMOROUS,
                "love_affection": EmotionalTone.AFFECTIONATE,
                "anger_frustration": EmotionalTone.ANGRY,
                "sadness_crying": EmotionalTone.SAD,
                "thinking_contemplation": EmotionalTone.CONTEMPLATIVE,
                "playful_teasing": EmotionalTone.PLAYFUL,
            }
            emotional_tone = emotion_mapping.get(emotional_tone, EmotionalTone.CASUAL)
        else:
            # Determine tone from content
            if topic_analysis["primary_topic"] == TopicCategory.POLITICS:
                emotional_tone = EmotionalTone.SERIOUS
            elif topic_analysis["primary_topic"] == TopicCategory.ACADEMIC:
                emotional_tone = EmotionalTone.INTELLECTUAL
            elif contains_profanity:
                emotional_tone = EmotionalTone.CASUAL
            else:
                emotional_tone = EmotionalTone.CASUAL

        return SignalMessage(
            message_id=message_id,
            thread_id=thread_id,
            sender_id=sender_id,
            timestamp=timestamp,
            body=body,
            message_type=message_type,
            emotional_tone=emotional_tone,
            topic_category=topic_analysis["primary_topic"],
            contains_emoji=emoji_analysis.get("has_emojis", False),
            emoji_list=emojis,
            contains_url=len(urls) > 0,
            url_list=urls,
            word_count=word_count,
            character_count=character_count,
            contains_profanity=contains_profanity,
            academic_language=academic_language,
            internet_slang=internet_slang,
            response_to_message_id=response_to_message_id,
            time_since_previous=time_since_previous,
            is_correction=is_correction,
            is_continuation=is_continuation,
        )

    def _classify_message_type(
        self, body: str, context_messages: Optional[list[SignalMessage]]
    ) -> MessageType:
        """Classify the type of message based on content and context."""
        if not context_messages:
            return MessageType.STANDALONE

        # Check for corrections
        if context_messages and len(context_messages) > 0:
            last_msg = context_messages[-1]
            if (
                body.lower().startswith(last_msg.body.lower()[:20])
                and len(body) > len(last_msg.body) * 0.8
            ):
                return MessageType.CORRECTION

        # Check for media sharing
        if re.search(r"https?://[^\s]+", body):
            return MessageType.MEDIA_SHARE

        # Check for elaboration
        if (
            context_messages
            and len(context_messages) > 0
            and context_messages[-1].sender_id == context_messages[-1].sender_id  # Same sender
            and body.lower().startswith(("also", "and", "plus", "additionally", "furthermore"))
        ):
            return MessageType.ELABORATION

        # Check for responses
        if (
            context_messages
            and len(context_messages) > 0
            and context_messages[-1].sender_id != context_messages[-1].sender_id
        ):  # Different sender
            return MessageType.RESPONSE

        # Check for continuation
        if (
            context_messages
            and len(context_messages) > 0
            and context_messages[-1].sender_id == context_messages[-1].sender_id
        ):  # Same sender
            return MessageType.CONTINUATION

        return MessageType.STANDALONE

    def detect_burst_sequences(
        self, messages: list[SignalMessage], max_gap_seconds: int = 120
    ) -> list[BurstSequence]:
        """Detect burst messaging sequences in conversation."""
        if len(messages) < 3:
            return []

        burst_sequences = []
        current_burst = []

        for i, message in enumerate(messages):
            if i == 0:
                current_burst = [message]
                continue

            # Check if message continues the burst
            time_gap = (message.timestamp - messages[i - 1].timestamp).total_seconds()
            same_sender = message.sender_id == messages[i - 1].sender_id

            if time_gap <= max_gap_seconds and same_sender:
                current_burst.append(message)
            else:
                # End current burst if it's significant
                if len(current_burst) >= 3:
                    burst_seq = self._create_burst_sequence(current_burst)
                    burst_sequences.append(burst_seq)

                current_burst = [message]

        # Don't forget the last burst
        if len(current_burst) >= 3:
            burst_seq = self._create_burst_sequence(current_burst)
            burst_sequences.append(burst_seq)

        return burst_sequences

    def _create_burst_sequence(self, messages: list[SignalMessage]) -> BurstSequence:
        """Create a BurstSequence object from a list of messages."""
        message_texts = [msg.body for msg in messages]

        duration = (messages[-1].timestamp - messages[0].timestamp).total_seconds()
        avg_length = np.mean([len(msg.body) for msg in messages])

        # Check for corrections in burst
        contains_corrections = any(msg.is_correction for msg in messages)

        # Determine dominant topic
        topics = [
            msg.topic_category for msg in messages if msg.topic_category != TopicCategory.OTHER
        ]
        topic_counter = Counter(topics)
        dominant_topic = (
            topic_counter.most_common(1)[0][0] if topic_counter else TopicCategory.OTHER
        )

        # Determine emotional tone
        tones = [msg.emotional_tone for msg in messages]
        tone_counter = Counter(tones)
        dominant_tone = tone_counter.most_common(1)[0][0] if tone_counter else EmotionalTone.CASUAL

        return BurstSequence(
            messages=message_texts,
            duration_seconds=duration,
            message_count=len(messages),
            avg_message_length=avg_length,
            contains_corrections=contains_corrections,
            topic_category=dominant_topic,
            emotional_tone=dominant_tone,
        )

    def create_conversation_windows(
        self, messages: list[SignalMessage], window_size: int = 10, your_recipient_id: str = "2"
    ) -> list[ConversationWindow]:
        """Create conversation windows with comprehensive analysis."""
        windows = []

        if len(messages) < window_size:
            return windows

        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda x: x.timestamp)

        # Create sliding windows
        for i in range(len(sorted_messages) - window_size + 1):
            window_messages = sorted_messages[i : i + window_size]

            # Create window
            window = self._create_conversation_window(window_messages, your_recipient_id)
            if window:
                windows.append(window)

        return windows

    def _create_conversation_window(
        self, messages: list[SignalMessage], your_recipient_id: str
    ) -> Optional[ConversationWindow]:
        """Create a single conversation window with analysis."""
        if not messages:
            return None

        window_id = f"window_{messages[0].thread_id}_{messages[0].timestamp.isoformat()}"
        thread_id = messages[0].thread_id

        # Calculate duration
        duration_minutes = (messages[-1].timestamp - messages[0].timestamp).total_seconds() / 60

        # Analyze participants
        unique_speakers = list({msg.sender_id for msg in messages})
        message_distribution = Counter(msg.sender_id for msg in messages)

        # Determine dominant mood
        moods = [msg.emotional_tone for msg in messages]
        mood_counter = Counter(moods)

        # Map emotional tones to conversation moods
        tone_to_mood = {
            EmotionalTone.HUMOROUS: ConversationMood.HUMOROUS,
            EmotionalTone.SERIOUS: ConversationMood.SERIOUS,
            EmotionalTone.CONTEMPLATIVE: ConversationMood.PHILOSOPHICAL,
            EmotionalTone.PLAYFUL: ConversationMood.PLAYFUL,
            EmotionalTone.ANGRY: ConversationMood.HEATED,
            EmotionalTone.AFFECTIONATE: ConversationMood.SUPPORTIVE,
        }

        dominant_mood_tone = (
            mood_counter.most_common(1)[0][0] if mood_counter else EmotionalTone.CASUAL
        )
        dominant_mood = tone_to_mood.get(dominant_mood_tone, ConversationMood.CASUAL)

        # Determine primary topic
        topics = [
            msg.topic_category for msg in messages if msg.topic_category != TopicCategory.OTHER
        ]
        topic_counter = Counter(topics)
        primary_topic = topic_counter.most_common(1)[0][0] if topic_counter else TopicCategory.OTHER

        # Analyze conversation dynamics
        conversation_dynamics = self._determine_conversation_dynamics(
            messages, primary_topic, dominant_mood
        )

        # Detect burst sequences
        burst_sequences = self.detect_burst_sequences(messages)

        # Calculate response times
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].sender_id != messages[i - 1].sender_id:  # Different speakers
                response_time = (messages[i].timestamp - messages[i - 1].timestamp).total_seconds()
                if response_time < 3600:  # Within 1 hour
                    response_times.append(response_time)

        avg_response_time = np.mean(response_times) if response_times else None

        # Count content types
        total_emojis = sum(len(msg.emoji_list) for msg in messages)
        total_urls = sum(len(msg.url_list) for msg in messages)

        # Detect topic transitions
        topic_transitions = []
        prev_topic = None
        for msg in messages:
            if (
                prev_topic
                and msg.topic_category != prev_topic
                and msg.topic_category != TopicCategory.OTHER
            ):
                transition = TopicTransition(
                    from_topic=prev_topic,
                    to_topic=msg.topic_category,
                    transition_method="gradual",  # Simplified
                    trigger_message=msg.body[:50] + "..." if len(msg.body) > 50 else msg.body,
                    transition_smoothness=0.5,  # Default
                )
                topic_transitions.append(transition)
            prev_topic = msg.topic_category

        return ConversationWindow(
            window_id=window_id,
            thread_id=thread_id,
            messages=messages,
            start_timestamp=messages[0].timestamp,
            end_timestamp=messages[-1].timestamp,
            duration_minutes=duration_minutes,
            dominant_mood=dominant_mood,
            primary_topic=primary_topic,
            topic_transitions=topic_transitions,
            unique_speakers=unique_speakers,
            message_distribution=dict(message_distribution),
            conversation_dynamics=conversation_dynamics,
            total_emojis=total_emojis,
            total_urls=total_urls,
            burst_sequences=burst_sequences,
            avg_response_time=avg_response_time,
        )

    def _determine_conversation_dynamics(
        self,
        messages: list[SignalMessage],
        primary_topic: TopicCategory,
        dominant_mood: ConversationMood,
    ) -> ConversationDynamics:
        """Determine the type of conversation dynamics."""

        # Check for rapid-fire exchanges
        quick_exchanges = sum(
            1
            for i in range(1, len(messages))
            if (messages[i].timestamp - messages[i - 1].timestamp).total_seconds() < 30
        )

        if quick_exchanges / len(messages) > 0.6:
            return ConversationDynamics.RAPID_FIRE

        # Topic-based classification
        if primary_topic in [TopicCategory.POLITICS, TopicCategory.POLITICAL_THEORY]:
            return ConversationDynamics.POLITICAL_DEBATE
        elif primary_topic == TopicCategory.ACADEMIC:
            return ConversationDynamics.PHILOSOPHICAL
        elif primary_topic == TopicCategory.HUMOR:
            return ConversationDynamics.CASUAL_BANTER
        elif primary_topic == TopicCategory.SOCIAL_MEDIA:
            return ConversationDynamics.MEDIA_SHARING

        # Mood-based classification
        if dominant_mood == ConversationMood.SUPPORTIVE:
            return ConversationDynamics.SUPPORTIVE
        elif dominant_mood == ConversationMood.PHILOSOPHICAL:
            return ConversationDynamics.PHILOSOPHICAL

        # Check for monologue (one person dominating)
        sender_counts = Counter(msg.sender_id for msg in messages)
        max_sender_ratio = max(sender_counts.values()) / len(messages)

        if max_sender_ratio > 0.8:
            return ConversationDynamics.MONOLOGUE

        # Default
        return ConversationDynamics.CASUAL_BANTER

    def process_full_conversation_thread(
        self,
        messages_df: pd.DataFrame,
        thread_id: str,
        recipients_df: pd.DataFrame,
        your_recipient_id: str = "2",
    ) -> ConversationThread:
        """Process a complete conversation thread with comprehensive analysis."""
        logger.info(f"Processing conversation thread {thread_id}")

        # Filter to thread messages
        thread_messages_df = messages_df[messages_df["thread_id"] == thread_id].copy()
        thread_messages_df = thread_messages_df.sort_values("date_sent")

        # Convert to SignalMessage objects
        signal_messages = []
        for _, row in thread_messages_df.iterrows():
            signal_msg = self.process_signal_message(
                row, signal_messages[-5:] if signal_messages else None
            )
            signal_messages.append(signal_msg)

        if len(signal_messages) < 5:
            logger.warning(f"Thread {thread_id} has insufficient messages")
            return None

        # Create conversation windows
        windows = self.create_conversation_windows(
            signal_messages, your_recipient_id=your_recipient_id
        )

        # Thread metadata
        participants = list({msg.sender_id for msg in signal_messages})
        start_timestamp = signal_messages[0].timestamp
        end_timestamp = signal_messages[-1].timestamp
        total_duration_days = (end_timestamp - start_timestamp).total_seconds() / (24 * 3600)

        # Topic evolution analysis
        topic_transitions = []
        for window in windows:
            topic_transitions.extend(window.topic_transitions)

        # Dominant topics
        all_topics = [
            msg.topic_category
            for msg in signal_messages
            if msg.topic_category != TopicCategory.OTHER
        ]
        topic_counter = Counter(all_topics)
        dominant_topics = [topic for topic, _ in topic_counter.most_common(5)]

        # Mood patterns
        mood_patterns = [window.dominant_mood for window in windows]

        # Determine relationship dynamic
        if len(participants) == 2:
            relationship_dynamic = self.personality_profiler.analyze_relationship_dynamics(
                messages_df, participants[0], participants[1]
            )
        else:
            relationship_dynamic = RelationshipDynamic.CLOSE_FRIENDS  # Default for groups

        # Generate personality profiles for participants
        participant_personalities = {}
        for participant_id in participants:
            try:
                personality = self.personality_profiler.generate_personality_profile(
                    messages_df, participant_id, recipients_df
                )
                participant_personalities[participant_id] = personality
            except Exception as e:
                logger.warning(f"Could not generate personality profile for {participant_id}: {e}")

        # Calculate communication balance
        message_counts = Counter(msg.sender_id for msg in signal_messages)
        total_messages = len(signal_messages)
        communication_balance = {
            pid: count / total_messages for pid, count in message_counts.items()
        }

        # Analyze response times and emoji patterns (simplified)
        typical_response_times = {pid: MessageTiming.MODERATE for pid in participants}
        emoji_usage_patterns = {}
        url_sharing_frequency = Counter(
            msg.sender_id for msg in signal_messages if msg.contains_url
        )

        return ConversationThread(
            thread_id=thread_id,
            participants=participants,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_messages=len(signal_messages),
            total_duration_days=total_duration_days,
            windows=windows,
            topic_evolution=topic_transitions,
            dominant_topics=dominant_topics,
            mood_patterns=mood_patterns,
            relationship_dynamic=relationship_dynamic,
            participant_personalities=participant_personalities,
            communication_balance=communication_balance,
            typical_response_times=typical_response_times,
            emoji_usage_patterns=emoji_usage_patterns,
            url_sharing_frequency=dict(url_sharing_frequency),
        )

    def generate_training_data(
        self,
        conversation_thread: ConversationThread,
        your_recipient_id: str = "2",
        style_focus: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Generate training data from processed conversation thread."""
        logger.info(f"Generating training data for thread {conversation_thread.thread_id}")

        training_examples = conversation_thread.get_training_examples(
            your_recipient_id, style_focus
        )

        # Enhance with conversation analysis
        for example in training_examples:
            # Add personality context
            if your_recipient_id in conversation_thread.participant_personalities:
                personality = conversation_thread.participant_personalities[your_recipient_id]
                example["personality_context"] = {
                    "message_style": personality.message_style,
                    "humor_type": personality.humor_type,
                    "formality_level": personality.academic_tendency,
                    "signature_phrases": personality.signature_phrases[:5],
                }

            # Add relationship context
            example["relationship_context"] = {
                "relationship_type": conversation_thread.relationship_dynamic.value,
                "communication_balance": conversation_thread.communication_balance.get(
                    your_recipient_id, 0.5
                ),
                "dominant_topics": [
                    topic.value for topic in conversation_thread.dominant_topics[:3]
                ],
            }

        return training_examples
