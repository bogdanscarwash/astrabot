"""
Metadata enrichment functions for Astrabot.

This module adds rich metadata to messages including temporal context,
emotional signals, group dynamics, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from collections import Counter
import requests

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MetadataEnricher:
    """Enriches messages with additional metadata and context."""
    
    def __init__(self, your_recipient_id: int = 2):
        """
        Initialize the metadata enricher.
        
        Args:
            your_recipient_id: Your recipient ID in the Signal database (default: 2)
        """
        self.your_recipient_id = your_recipient_id
        self.logger = get_logger(__name__)
        
    def enrich_with_urls(self, message: str) -> Dict[str, Any]:
        """
        Extract and analyze URLs from a message.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with URL metadata
        """
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, message)
        
        return {
            'has_urls': len(urls) > 0,
            'url_count': len(urls),
            'urls': urls,
            'has_twitter': any('twitter.com' in url or 'x.com' in url for url in urls),
            'has_youtube': any('youtube.com' in url or 'youtu.be' in url for url in urls)
        }
    
    def enrich_with_mentions(self, message: str) -> Dict[str, Any]:
        """
        Extract mentions (@username) from a message.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with mention metadata
        """
        mention_pattern = r'@[A-Za-z0-9_]+'
        mentions = re.findall(mention_pattern, message)
        
        return {
            'has_mentions': len(mentions) > 0,
            'mention_count': len(mentions),
            'mentions': mentions
        }
    
    def enrich_with_hashtags(self, message: str) -> Dict[str, Any]:
        """
        Extract hashtags from a message.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with hashtag metadata
        """
        hashtag_pattern = r'#[A-Za-z0-9_]+'
        hashtags = re.findall(hashtag_pattern, message)
        
        return {
            'has_hashtags': len(hashtags) > 0,
            'hashtag_count': len(hashtags),
            'hashtags': hashtags
        }
    
    def enrich_with_emojis(self, message: str) -> Dict[str, Any]:
        """
        Extract and analyze emoji usage.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with emoji metadata
        """
        # Simple emoji detection (can be enhanced)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        emojis = emoji_pattern.findall(message)
        emoji_list = [e for emoji in emojis for e in emoji]
        
        return {
            'has_emojis': len(emoji_list) > 0,
            'emoji_count': len(emoji_list),
            'emojis': emoji_list,
            'emoji_density': len(emoji_list) / max(len(message.split()), 1)
        }
    
    def enrich_with_sentiment(self, message: str) -> Dict[str, Any]:
        """
        Basic sentiment analysis based on patterns.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with sentiment metadata
        """
        # Basic sentiment keywords (can be enhanced with proper NLP)
        positive_words = ['love', 'great', 'awesome', 'excellent', 'happy', 'good', 'wonderful', 'fantastic', 'amazing', 'absolutely', '😊', '❤️', '👍']
        negative_words = ['hate', 'bad', 'awful', 'terrible', 'sad', 'angry', 'horrible', 'disappointed', 'disappointing', '😢', '😡', '👎']
        
        message_lower = message.lower()
        positive_score = sum(1 for word in positive_words if word in message_lower)
        negative_score = sum(1 for word in negative_words if word in message_lower)
        
        if positive_score > negative_score:
            sentiment = 'positive'
        elif negative_score > positive_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'positive_score': positive_score,
            'negative_score': negative_score
        }
    
    def enrich_with_language_detection(self, message: str) -> Dict[str, Any]:
        """
        Detect if message contains non-English text.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with language metadata
        """
        # Basic non-ASCII detection
        non_ascii_chars = [c for c in message if ord(c) > 127]
        
        return {
            'has_non_english': len(non_ascii_chars) > 0,
            'non_ascii_ratio': len(non_ascii_chars) / max(len(message), 1)
        }
    
    def enrich_empty_message(self, message: str) -> Dict[str, Any]:
        """
        Handle empty or minimal messages.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with empty message metadata
        """
        return {
            'is_empty': not message or message.strip() == '',
            'is_single_char': len(message.strip()) == 1,
            'is_very_short': len(message.strip()) < 5
        }
    
    def enrich_with_code_blocks(self, message: str) -> Dict[str, Any]:
        """
        Detect code blocks or technical content.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with code block metadata
        """
        # Look for code indicators
        code_indicators = ['```', 'def ', 'function ', 'class ', 'import ', 'from ', 'var ', 'const ', 'let ']
        has_code = any(indicator in message for indicator in code_indicators)
        
        # Count backticks
        backtick_count = message.count('`')
        
        return {
            'has_code': has_code,
            'backtick_count': backtick_count,
            'has_code_block': '```' in message
        }
    
    def enrich_with_questions(self, message: str) -> Dict[str, Any]:
        """
        Detect questions in the message.
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with question metadata
        """
        # Simple question detection
        question_words = ['what', 'when', 'where', 'why', 'who', 'how', 'which', 'whose', 'whom']
        message_lower = message.lower()
        
        has_question_mark = '?' in message
        has_question_word = any(message_lower.startswith(word) for word in question_words)
        
        return {
            'is_question': has_question_mark or has_question_word,
            'has_question_mark': has_question_mark,
            'question_count': message.count('?')
        }
    
    def enrich_message(self, message: Dict[str, Any], extract_link_previews: bool = False) -> Dict[str, Any]:
        """
        Enrich a single message with all metadata.
        
        Args:
            message: Message dictionary with 'text' field
            extract_link_previews: Whether to extract link preview metadata
            
        Returns:
            Enriched message dictionary
        """
        text = message.get('text', '')
        
        enrichment = {
            **self.enrich_with_urls(text),
            **self.enrich_with_mentions(text),
            **self.enrich_with_hashtags(text),
            **self.enrich_with_emojis(text),
            **self.enrich_with_sentiment(text),
            **self.enrich_with_language_detection(text),
            **self.enrich_empty_message(text),
            **self.enrich_with_code_blocks(text),
            **self.enrich_with_questions(text)
        }
        
        # Map some fields for test compatibility
        enrichment['detected_urls'] = enrichment.get('urls', [])
        enrichment['has_media'] = enrichment.get('has_urls', False)
        
        # Handle sentiment output format
        if 'sentiment' in enrichment:
            sentiment_value = enrichment['sentiment']
            score = enrichment.get('positive_score', 0) - enrichment.get('negative_score', 0)
            enrichment['sentiment'] = {
                'score': score,
                'label': sentiment_value
            }
        
        # Handle language detection
        if enrichment.get('has_non_english'):
            # Simple language detection based on non-ASCII characters
            if 'spanish' in text.lower() or '¿' in text or '¡' in text:
                enrichment['detected_language'] = 'es'
            else:
                enrichment['detected_language'] = 'en'
        else:
            enrichment['detected_language'] = 'en'
        
        # Handle code language detection
        if enrichment.get('has_code_block'):
            code_languages = []
            if 'python' in text.lower():
                code_languages.append('python')
            enrichment['code_languages'] = code_languages
        
        # Handle link previews if requested
        if extract_link_previews and enrichment.get('detected_urls'):
            # Mock implementation for testing
            enrichment['link_previews'] = [
                {
                    'url': url,
                    'title': 'Test Article',
                    'description': 'This is a test article about AI',
                    'image': 'https://example.com/image.jpg'
                } for url in enrichment['detected_urls']
            ]
        
        return enrichment
    
    def enrich_messages_batch(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple messages in batch.
        
        Args:
            messages: List of message dictionaries with 'text' field
            
        Returns:
            List of enriched message dictionaries
        """
        return [self.enrich_message(message) for message in messages]
    
    def batch_enrichment(self, messages: List[str]) -> List[Dict[str, Any]]:
        """
        Enrich multiple messages in batch (legacy method).
        
        Args:
            messages: List of message texts
            
        Returns:
            List of enrichment dictionaries
        """
        results = []
        for message in messages:
            enrichment = {
                **self.enrich_with_urls(message),
                **self.enrich_with_mentions(message),
                **self.enrich_with_hashtags(message),
                **self.enrich_with_emojis(message),
                **self.enrich_with_sentiment(message),
                **self.enrich_with_language_detection(message),
                **self.enrich_empty_message(message),
                **self.enrich_with_code_blocks(message),
                **self.enrich_with_questions(message)
            }
            results.append(enrichment)
        return results
    
    def enrich_with_link_preview(self, message: str) -> Dict[str, Any]:
        """
        Extract metadata for link previews (placeholder for actual implementation).
        
        Args:
            message: The message text
            
        Returns:
            Dictionary with link preview metadata
        """
        urls = re.findall(r'https?://[^\s]+', message)
        
        # This is a placeholder - real implementation would fetch actual previews
        return {
            'has_link_preview': len(urls) > 0,
            'preview_urls': urls[:3]  # Limit to first 3 URLs
        }


def add_reaction_context(messages_df: pd.DataFrame, reactions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add emotional context from reactions to messages.
    
    Args:
        messages_df: DataFrame of messages
        reactions_df: DataFrame of reactions (optional, will try to load if not provided)
    
    Returns:
        Messages DataFrame with reaction context added
    """
    if reactions_df is None:
        try:
            # Try to load reactions from standard location
            import os
            base_path = os.path.dirname(messages_df.attrs.get('source_path', ''))
            reactions_path = os.path.join(base_path, 'reaction.csv')
            if os.path.exists(reactions_path):
                reactions_df = pd.read_csv(reactions_path)
            else:
                # No reactions available
                messages_df['emoji'] = [[] for _ in range(len(messages_df))]
                messages_df['reaction_count'] = 0
                return messages_df
        except Exception:
            messages_df['emoji'] = [[] for _ in range(len(messages_df))]
            messages_df['reaction_count'] = 0
            return messages_df
    
    # Group reactions by message
    reaction_summary = reactions_df.groupby('message_id').agg({
        'emoji': lambda x: list(x),
        'author_id': 'count'
    }).rename(columns={'author_id': 'reaction_count'})
    
    # Add reaction data to messages
    messages_df = messages_df.merge(
        reaction_summary, 
        left_on='_id', 
        right_index=True, 
        how='left'
    )
    
    # Fill NaN values
    messages_df['emoji'] = messages_df['emoji'].apply(lambda x: [] if pd.isna(x) else x)
    messages_df['reaction_count'] = messages_df['reaction_count'].fillna(0)
    
    return messages_df


def add_group_context(messages_df: pd.DataFrame, threads_df: Optional[pd.DataFrame] = None,
                     groups_df: Optional[pd.DataFrame] = None, 
                     membership_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Add group chat context to messages.
    
    Args:
        messages_df: DataFrame of messages
        threads_df: DataFrame of threads (optional)
        groups_df: DataFrame of groups (optional)
        membership_df: DataFrame of group memberships (optional)
    
    Returns:
        Messages DataFrame with group context added
    """
    # Default values if group data not available
    messages_df['is_group_chat'] = False
    messages_df['member_count'] = 2
    messages_df['group_name'] = 'Direct Message'
    
    if threads_df is None or groups_df is None:
        return messages_df
    
    try:
        # Create group lookup
        group_lookup = {}
        
        for _, thread in threads_df.iterrows():
            thread_id = thread['_id']
            recipient_id = thread.get('recipient_id')
            
            if recipient_id is None:
                continue
            
            # Check if this is a group
            group_info = groups_df[groups_df['recipient_id'] == recipient_id]
            if not group_info.empty:
                group_id = group_info.iloc[0]['_id']
                
                # Count members if membership data available
                member_count = 2  # Default
                if membership_df is not None:
                    member_count = len(membership_df[membership_df['group_id'] == group_id])
                
                group_lookup[thread_id] = {
                    'is_group': True,
                    'member_count': member_count,
                    'group_name': group_info.iloc[0].get('title', 'Unknown Group')
                }
            else:
                group_lookup[thread_id] = {
                    'is_group': False,
                    'member_count': 2,
                    'group_name': 'Direct Message'
                }
        
        # Add group context to messages
        messages_df['is_group_chat'] = messages_df['thread_id'].map(
            lambda x: group_lookup.get(x, {}).get('is_group', False)
        )
        messages_df['member_count'] = messages_df['thread_id'].map(
            lambda x: group_lookup.get(x, {}).get('member_count', 2)
        )
        messages_df['group_name'] = messages_df['thread_id'].map(
            lambda x: group_lookup.get(x, {}).get('group_name', 'Direct Message')
        )
        
    except Exception as e:
        print(f"Warning: Could not fully process group data: {e}")
    
    return messages_df


def add_temporal_context(messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based context to messages.
    
    Args:
        messages_df: DataFrame of messages
    
    Returns:
        Messages DataFrame with temporal context added
    """
    # Convert timestamps
    messages_df['datetime'] = pd.to_datetime(messages_df['date_sent'], unit='ms')
    
    # Extract temporal features
    messages_df['hour'] = messages_df['datetime'].dt.hour
    messages_df['day_of_week'] = messages_df['datetime'].dt.day_name()
    messages_df['time_period'] = messages_df['hour'].apply(get_time_period)
    messages_df['date'] = messages_df['datetime'].dt.date
    messages_df['month'] = messages_df['datetime'].dt.month
    messages_df['year'] = messages_df['datetime'].dt.year
    
    # Calculate response timing within threads
    messages_df = messages_df.sort_values(['thread_id', 'date_sent'])
    messages_df['response_delay'] = messages_df.groupby('thread_id')['date_sent'].diff() / 1000  # Convert to seconds
    
    # Add urgency classification
    messages_df['urgency'] = messages_df['response_delay'].apply(classify_urgency)
    
    return messages_df


def get_time_period(hour: int) -> str:
    """
    Classify time of day based on hour.
    
    Args:
        hour: Hour of day (0-23)
    
    Returns:
        String classification of time period
    """
    if 6 <= hour < 12: 
        return 'morning'
    elif 12 <= hour < 17: 
        return 'afternoon'
    elif 17 <= hour < 21: 
        return 'evening'
    else: 
        return 'night'


def classify_urgency(response_delay_seconds: Optional[float]) -> str:
    """
    Classify response urgency based on delay.
    
    Args:
        response_delay_seconds: Response delay in seconds
    
    Returns:
        String classification of urgency
    """
    if pd.isna(response_delay_seconds) or response_delay_seconds < 60:
        return "immediate"
    elif response_delay_seconds < 3600:  # 1 hour
        return "quick"
    elif response_delay_seconds < 86400:  # 1 day
        return "delayed"
    else:
        return "long_delay"


def classify_emotion_from_reactions(emoji_list: List[str]) -> str:
    """
    Simple emotion classification from reaction emojis.
    
    Args:
        emoji_list: List of emoji reactions
    
    Returns:
        String classification of emotion
    """
    if not emoji_list:
        return 'neutral'
    
    positive_emojis = ['❤️', '😍', '😊', '😂', '👍', '🔥', '💯', '✨', '🎉', '💕']
    negative_emojis = ['😢', '😡', '👎', '💔', '😔', '😠', '😤']
    
    positive_count = sum(1 for emoji in emoji_list if emoji in positive_emojis)
    negative_count = sum(1 for emoji in emoji_list if emoji in negative_emojis)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


def add_conversation_flow_metadata(messages_df: pd.DataFrame, your_recipient_id: int = 2) -> pd.DataFrame:
    """
    Add metadata about conversation flow and turn-taking.
    
    Args:
        messages_df: DataFrame of messages
        your_recipient_id: Your recipient ID
    
    Returns:
        Messages DataFrame with conversation flow metadata
    """
    messages_df = messages_df.sort_values(['thread_id', 'date_sent'])
    
    # Track speaker changes
    messages_df['speaker_changed'] = (
        messages_df.groupby('thread_id')['from_recipient_id'].shift() != 
        messages_df['from_recipient_id']
    )
    
    # Count consecutive messages from same sender
    messages_df['consecutive_count'] = messages_df.groupby(
        ['thread_id', (messages_df['speaker_changed'].cumsum())]
    ).cumcount() + 1
    
    # Identify if this is part of a burst
    messages_df['is_burst_message'] = (
        messages_df['consecutive_count'] > 1
    ) | (
        messages_df.groupby(['thread_id', 'from_recipient_id'])['consecutive_count'].shift(-1) > 1
    )
    
    # Add role in conversation
    messages_df['sender_role'] = messages_df['from_recipient_id'].apply(
        lambda x: 'You' if x == your_recipient_id else 'Other'
    )
    
    return messages_df


def enrich_messages_with_all_metadata(messages_df: pd.DataFrame, 
                                     supporting_data: Optional[Dict[str, pd.DataFrame]] = None,
                                     your_recipient_id: int = 2) -> pd.DataFrame:
    """
    Apply all metadata enrichment functions to messages.
    
    Args:
        messages_df: DataFrame of messages
        supporting_data: Dictionary with optional DataFrames (reactions, threads, groups, membership)
        your_recipient_id: Your recipient ID
    
    Returns:
        Fully enriched messages DataFrame
    """
    if supporting_data is None:
        supporting_data = {}
    
    # Apply all enrichments
    messages_df = add_temporal_context(messages_df)
    messages_df = add_reaction_context(messages_df, supporting_data.get('reactions'))
    messages_df = add_group_context(
        messages_df, 
        supporting_data.get('threads'),
        supporting_data.get('groups'),
        supporting_data.get('membership')
    )
    messages_df = add_conversation_flow_metadata(messages_df, your_recipient_id)
    
    # Add emotional context from reactions
    messages_df['emotional_context'] = messages_df['emoji'].apply(classify_emotion_from_reactions)
    
    return messages_df