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

import re
import requests
from urllib.parse import urlparse
import time
from bs4 import BeautifulSoup
import json
from typing import Optional, Dict, List, Tuple
import os
import tempfile
from datetime import datetime
from structured_schemas import (
    ImageDescription, TweetContent, ImageWithContext, 
    BatchImageDescription, EnhancedMessage, Sentiment,
    generate_json_schema, IMAGE_DESCRIPTION_SCHEMA
)


def extract_tweet_text(url: str, return_structured: bool = False) -> Optional[Dict[str, str] | TweetContent]:
    """
    Extract only the main tweet text from a Twitter/X URL.
    
    Args:
        url: Twitter/X URL (supports twitter.com, x.com, and t.co)
        return_structured: If True, return TweetContent object instead of dict
    
    Returns:
        Dict with 'text', 'author', 'tweet_id' or TweetContent object, or None if extraction fails
    """
    # Clean and normalize URL
    if 't.co' in url:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            url = response.url
        except:
            return None
    
    # Convert x.com to twitter.com for consistency
    url = url.replace('x.com', 'twitter.com')
    
    # Extract tweet ID
    tweet_id_match = re.search(r'/status/(\d+)', url)
    if not tweet_id_match:
        return None
    
    tweet_id = tweet_id_match.group(1)
    username_match = re.search(r'twitter\.com/([^/]+)/', url)
    username = username_match.group(1) if username_match else 'unknown'
    
    # Try Nitter instances first (privacy-friendly, no API needed)
    nitter_instances = [
        'nitter.privacydev.net',
        'nitter.poast.org',
        'nitter.net',
        'nitter.it',
        'nitter.unixfox.eu'
    ]
    
    for instance in nitter_instances:
        try:
            nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(nitter_url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find the main tweet content
                tweet_content = soup.find('div', class_='tweet-content')
                if tweet_content:
                    # Extract text, removing extra whitespace
                    tweet_text = tweet_content.get_text().strip()
                    tweet_text = ' '.join(tweet_text.split())
                    
                    # Extract hashtags and mentions if structured output requested
                    if return_structured:
                        hashtags = re.findall(r'#\w+', tweet_text)
                        mentions = re.findall(r'@\w+', tweet_text)
                        
                        return TweetContent(
                            text=tweet_text,
                            author=username,
                            tweet_id=tweet_id,
                            mentioned_users=[m[1:] for m in mentions],  # Remove @ symbol
                            hashtags=[h[1:] for h in hashtags],  # Remove # symbol
                            sentiment=Sentiment.NEUTRAL  # Default, could be enhanced
                        )
                    else:
                        return {
                            'text': tweet_text,
                            'author': username,
                            'tweet_id': tweet_id
                        }
        except Exception as e:
            continue
    
    # Fallback: Try direct Twitter scraping (less reliable)
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; bot)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Look for tweet text in meta tags (often available)
            soup = BeautifulSoup(response.content, 'html.parser')
            meta_desc = soup.find('meta', {'property': 'og:description'})
            if meta_desc and meta_desc.get('content'):
                tweet_text = meta_desc['content']
                
                if return_structured:
                    hashtags = re.findall(r'#\w+', tweet_text)
                    mentions = re.findall(r'@\w+', tweet_text)
                    
                    return TweetContent(
                        text=tweet_text,
                        author=username,
                        tweet_id=tweet_id,
                        mentioned_users=[m[1:] for m in mentions],
                        hashtags=[h[1:] for h in hashtags],
                        sentiment=Sentiment.NEUTRAL
                    )
                else:
                    return {
                        'text': tweet_text,
                        'author': username,
                        'tweet_id': tweet_id
                    }
    except:
        pass
    
    return None


def inject_tweet_context(message_with_url: str, tweet_data: Optional[Dict[str, str]]) -> str:
    """
    Inject tweet text into the message with clear markers.
    
    Args:
        message_with_url: Original message containing the Twitter URL
        tweet_data: Extracted tweet data from extract_tweet_text()
    
    Returns:
        Message with tweet content injected, or original message if no data
    """
    if not tweet_data or not tweet_data.get('text'):
        return message_with_url
    
    # Format: Keep original message but add tweet content clearly marked
    tweet_section = f"\n\n[TWEET: @{tweet_data['author']}]\n{tweet_data['text']}\n[/TWEET]"
    
    return message_with_url + tweet_section


def extract_tweet_images(url: str) -> List[str]:
    """
    Extract image URLs from the main tweet only.
    
    Args:
        url: Twitter/X URL
    
    Returns:
        List of direct image URLs (not thumbnails)
    """
    image_urls = []
    
    # Normalize URL
    if 't.co' in url:
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            url = response.url
        except:
            return []
    
    url = url.replace('x.com', 'twitter.com')
    
    # Extract tweet ID and username
    tweet_id_match = re.search(r'/status/(\d+)', url)
    if not tweet_id_match:
        return []
    
    tweet_id = tweet_id_match.group(1)
    username_match = re.search(r'twitter\.com/([^/]+)/', url)
    username = username_match.group(1) if username_match else 'unknown'
    
    # Try Nitter instances
    nitter_instances = [
        'nitter.privacydev.net',
        'nitter.poast.org',
        'nitter.net',
        'nitter.it'
    ]
    
    for instance in nitter_instances:
        try:
            nitter_url = f"https://{instance}/{username}/status/{tweet_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(nitter_url, timeout=10, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find images in the main tweet
                # Nitter usually puts images in <a> tags with class 'still-image'
                for img_link in soup.find_all('a', class_='still-image'):
                    img_url = img_link.get('href')
                    if img_url:
                        # Convert Nitter image URL to direct Twitter image URL
                        if '/pic/' in img_url:
                            # Extract the image filename
                            img_match = re.search(r'/pic/([^?]+)', img_url)
                            if img_match:
                                img_filename = img_match.group(1)
                                # Construct direct Twitter image URL
                                direct_url = f"https://pbs.twimg.com/media/{img_filename}"
                                image_urls.append(direct_url)
                
                # Also check for img tags within tweet content
                tweet_content = soup.find('div', class_='tweet-content')
                if tweet_content:
                    for img in tweet_content.find_all('img'):
                        src = img.get('src')
                        if src and 'twimg.com' in src:
                            image_urls.append(src)
                
                if image_urls:
                    return image_urls
        except:
            continue
    
    return image_urls


def describe_tweet_images(image_urls: List[str], api_endpoint: str, api_key: Optional[str] = None, batch_process: bool = True) -> List[str]:
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
    
    # Get API key from environment if not provided
    if not api_key:
        if api_endpoint == 'openai':
            api_key = os.environ.get('OPENAI_API_KEY')
        elif api_endpoint == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY')
    
    # Batch process with OpenAI if enabled and multiple images
    if batch_process and api_endpoint == 'openai' and len(image_urls) > 1 and api_key:
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
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                tmp_file.write(img_response.content)
                tmp_path = tmp_file.name
            
            try:
                # Send to appropriate API
                if api_endpoint == 'openai' and api_key:
                    desc_result = _describe_with_openai(tmp_path, api_key, use_structured=True)
                    # Convert ImageDescription to string for backward compatibility
                    if isinstance(desc_result, ImageDescription):
                        description = desc_result.to_training_format()
                    else:
                        description = desc_result
                elif api_endpoint == 'anthropic' and api_key:
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
    images_with_context: List[Dict[str, any]], 
    api_key: Optional[str] = None
) -> List[BatchImageDescription]:
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
    
    # Get API key from environment if not provided
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OpenAI API key required for image description")
    
    import base64
    
    # Build content array with all images and track their indices
    content = [
        {
            'type': 'text',
            'text': f'Please describe each of these {len(images_with_context)} images concisely. '
                   f'For each image, provide a JSON object with description, detected_text, '
                   f'main_subjects (array), and emotional_tone. Number each as "Image 1:", "Image 2:", etc.'
        }
    ]
    
    # Track mapping: index -> context
    index_to_context = {}
    valid_indices = []
    
    # Download and encode all images
    for i, img_context in enumerate(images_with_context):
        try:
            img_url = img_context['image_url']
            img_response = requests.get(img_url, timeout=10)
            if img_response.status_code == 200:
                image_data = base64.b64encode(img_response.content).decode('utf-8')
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image_data}',
                        'detail': 'low'  # Use low detail for cost optimization
                    }
                })
                
                # Create ImageWithContext object
                index_to_context[i] = ImageWithContext(
                    image_url=img_url,
                    conversation_id=img_context['conversation_id'],
                    message_id=img_context['message_id'],
                    sender_id=img_context['sender_id'],
                    timestamp=img_context['timestamp'],
                    tweet_url=img_context.get('tweet_url')
                )
                valid_indices.append(i)
        except Exception as e:
            print(f"Error downloading image {i}: {str(e)}")
            continue
    
    if not valid_indices:
        return []
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'user',
                'content': content
            }
        ],
        'response_format': IMAGE_DESCRIPTION_SCHEMA,
        'max_tokens': 500 * len(valid_indices)  # More tokens for multiple images
    }
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            results = []
            full_response = response.json()['choices'][0]['message']['content']
            
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
                            image_context=context,
                            description=description
                        )
                        results.append(batch_desc)
                
            except json.JSONDecodeError:
                # Fallback: try to parse numbered descriptions
                results = _parse_numbered_descriptions(full_response, index_to_context, valid_indices)
            
            return results
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        return []


def _parse_numbered_descriptions(response_text: str, index_to_context: Dict[int, ImageWithContext], 
                                valid_indices: List[int]) -> List[BatchImageDescription]:
    """Fallback parser for numbered descriptions if JSON parsing fails."""
    results = []
    
    # Split by "Image N:" pattern
    image_blocks = re.split(r'Image \d+:', response_text)
    
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
                subjects = [s.strip().strip('"') for s in subjects_str.split(',')]
            
            tone = tone_match.group(1) if tone_match else "neutral"
            
            # Create structured objects
            description = ImageDescription(
                description=description_text,
                detected_text=None,
                main_subjects=subjects,
                emotional_tone=tone
            )
            
            batch_desc = BatchImageDescription(
                image_context=context,
                description=description
            )
            results.append(batch_desc)
    
    return results


def _batch_describe_with_openai(image_urls: List[str], api_key: str) -> List[str]:
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
            'type': 'text',
            'text': f'Please describe each of these {len(image_urls)} images concisely in 1-2 sentences. Number each description.'
        }
    ]
    
    # Download and encode all images
    for i, img_url in enumerate(image_urls):
        try:
            img_response = requests.get(img_url, timeout=10)
            if img_response.status_code == 200:
                image_data = base64.b64encode(img_response.content).decode('utf-8')
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{image_data}',
                        'detail': 'low'  # Use low detail for cost optimization
                    }
                })
            else:
                content.append({
                    'type': 'text',
                    'text': f'Image {i+1}: Failed to download'
                })
        except Exception as e:
            content.append({
                'type': 'text',
                'text': f'Image {i+1}: Error - {str(e)}'
            })
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'user',
                'content': content
            }
        ],
        'max_tokens': 300  # Allow more tokens for multiple descriptions
    }
    
    try:
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            # Parse numbered descriptions from response
            full_response = response.json()['choices'][0]['message']['content']
            descriptions = []
            
            # Split by numbers and extract descriptions
            lines = full_response.split('\n')
            current_desc = []
            
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    if current_desc:
                        descriptions.append(' '.join(current_desc).strip())
                    current_desc = [re.sub(r'^\d+\.\s*', '', line.strip())]
                elif line.strip():
                    current_desc.append(line.strip())
            
            if current_desc:
                descriptions.append(' '.join(current_desc).strip())
            
            # Ensure we have the right number of descriptions
            while len(descriptions) < len(image_urls):
                descriptions.append("Description unavailable")
            
            return descriptions[:len(image_urls)]
        else:
            return [f"Batch API error: {response.status_code}"] * len(image_urls)
            
    except Exception as e:
        return [f"Batch processing error: {str(e)}"] * len(image_urls)


def _describe_with_openai(image_path: str, api_key: str, use_structured: bool = True) -> str | ImageDescription:
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
    
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    content = [
        {
            'type': 'text',
            'text': 'Describe this image concisely. Include any visible text, identify main subjects, and note the emotional tone.'
        },
        {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{image_data}',
                'detail': 'low'  # Use low detail for cost optimization
            }
        }
    ]
    
    payload = {
        'model': 'gpt-4o-mini',  # Using GPT-4o-mini for cost-effective vision processing
        'messages': [
            {
                'role': 'user',
                'content': content
            }
        ],
        'max_tokens': 150
    }
    
    # Add structured output format if requested
    if use_structured:
        payload['response_format'] = IMAGE_DESCRIPTION_SCHEMA
    
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content']
        
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
                emotional_tone="neutral"
            )
        else:
            return error_msg


def _describe_with_anthropic(image_path: str, api_key: str) -> str:
    """Helper function for Anthropic Claude Vision API"""
    import base64
    
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    headers = {
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'claude-3-opus-20240229',
        'max_tokens': 150,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': 'Describe this image concisely in 1-2 sentences.'
                    },
                    {
                        'type': 'image',
                        'source': {
                            'type': 'base64',
                            'media_type': 'image/jpeg',
                            'data': image_data
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        return f"Anthropic API error: {response.status_code}"


# Example usage function
def process_message_with_twitter_content(message: str, use_images: bool = True, 
                                        image_api: str = 'openai') -> str:
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
        r'https?://(?:www\.)?twitter\.com/\S+/status/\d+',
        r'https?://(?:www\.)?x\.com/\S+/status/\d+',
        r'https?://t\.co/\S+'
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
    image_api: str = 'openai'
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
        r'https?://(?:www\.)?twitter\.com/\S+/status/\d+',
        r'https?://(?:www\.)?x\.com/\S+/status/\d+',
        r'https?://t\.co/\S+'
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
                            images_with_context.append({
                                'image_url': img_url,
                                'conversation_id': conversation_id,
                                'message_id': message_id,
                                'sender_id': sender_id,
                                'timestamp': timestamp,
                                'tweet_url': url
                            })
                        
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
        image_descriptions=image_descriptions
    )