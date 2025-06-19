"""
Emoji Analysis Module for Astrabot.

This module provides emoji-specific analysis capabilities including:
- Emoji sentiment analysis and emotional mapping
- Context-aware emoji interpretation
- Personal emoji signature detection
- Emotional state correlation through emoji usage
"""

import emoji
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import pandas as pd

from src.models.schemas import EmojiUsagePattern, EmotionalTone
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmojiAnalyzer:
    """Comprehensive emoji analysis for conversation data."""
    
    def __init__(self):
        """Initialize emoji analyzer with emotion mappings."""
        self._init_emotion_mappings()
        logger.info("EmojiAnalyzer initialized with emotion mappings")
    
    def _init_emotion_mappings(self) -> None:
        """Initialize emoji to emotion category mappings based on Signal data analysis."""
        self.emotion_categories = {
            # Joy and laughter - most common in Signal data
            'joy_laughter': {
                'emojis': ['😂', '🤣', '😄', '😃', '😁', '😊', '😀', '🙂', '😋', '😌', '😆'],
                'tone': EmotionalTone.HUMOROUS,
                'sentiment_score': 0.9
            },
            
            # Love and affection - strong positive emotions
            'love_affection': {
                'emojis': ['❤️', '💖', '💕', '💗', '💓', '💘', '💝', '💜', '🧡', '💛', '💚', '💙', 
                          '🤍', '🖤', '🤎', '💯', '😍', '🥰', '😘', '😗', '😙', '😚'],
                'tone': EmotionalTone.AFFECTIONATE,
                'sentiment_score': 0.8
            },
            
            # Sadness and crying
            'sadness_crying': {
                'emojis': ['😢', '😭', '😞', '😔', '😟', '😕', '🙁', '☹️', '😩', '😫'],
                'tone': EmotionalTone.SAD,
                'sentiment_score': -0.6
            },
            
            # Anger and frustration
            'anger_frustration': {
                'emojis': ['😠', '😡', '🤬', '😤', '💢', '😾', '😖', '😣'],
                'tone': EmotionalTone.ANGRY,
                'sentiment_score': -0.7
            },
            
            # Surprise and shock
            'surprise_shock': {
                'emojis': ['😮', '😯', '😲', '🤯', '😱', '🙀', '😳'],
                'tone': EmotionalTone.ANXIOUS,
                'sentiment_score': 0.2
            },
            
            # Thinking and contemplation - philosophical discussions
            'thinking_contemplation': {
                'emojis': ['🤔', '🧐', '🤨', '🙄', '😐', '😑', '🤐'],
                'tone': EmotionalTone.CONTEMPLATIVE,
                'sentiment_score': 0.1
            },
            
            # Playful and teasing - common in casual banter
            'playful_teasing': {
                'emojis': ['😏', '😜', '😝', '😛', '🤪', '🤭', '😈', '👿', '🤡'],
                'tone': EmotionalTone.PLAYFUL,
                'sentiment_score': 0.6
            },
            
            # Support and encouragement
            'support_encouragement': {
                'emojis': ['👍', '👌', '✌️', '🤝', '👏', '🙌', '💪', '🎉', '🎊', '✨'],
                'tone': EmotionalTone.HAPPY,
                'sentiment_score': 0.7
            },
            
            # Confusion and uncertainty
            'confusion_uncertainty': {
                'emojis': ['😵', '🤷', '🤦', '😅', '😬'],
                'tone': EmotionalTone.ANXIOUS,
                'sentiment_score': -0.2
            },
            
            # Cool and casual - sarcastic or laid-back
            'cool_casual': {
                'emojis': ['😎', '🤓', '🥶', '🥵', '🤠'],
                'tone': EmotionalTone.CASUAL,
                'sentiment_score': 0.3
            }
        }
        
        # Create reverse mapping for quick lookup
        self.emoji_to_category = {}
        self.emoji_to_tone = {}
        self.emoji_to_sentiment = {}
        
        for category, data in self.emotion_categories.items():
            for emoji_char in data['emojis']:
                self.emoji_to_category[emoji_char] = category
                self.emoji_to_tone[emoji_char] = data['tone']
                self.emoji_to_sentiment[emoji_char] = data['sentiment_score']
    
    def extract_emojis_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract emojis with their position and context."""
        if not isinstance(text, str) or not text:
            return []
        
        emojis_found = []
        
        for i, char in enumerate(text):
            if char in emoji.EMOJI_DATA:
                # Get surrounding context (15 chars before and after)
                start = max(0, i - 15)
                end = min(len(text), i + 16)
                context = text[start:end].strip()
                
                # Determine position in message
                position_type = "mid_message"
                if i < 3:
                    position_type = "start"
                elif i > len(text) - 4:
                    position_type = "end"
                elif len(text.strip()) == 1 and text.strip() == char:
                    position_type = "standalone"
                
                emojis_found.append({
                    'emoji': char,
                    'position': i,
                    'context': context,
                    'position_type': position_type,
                    'emotion_category': self.emoji_to_category.get(char, 'other'),
                    'emotional_tone': self.emoji_to_tone.get(char, EmotionalTone.CASUAL),
                    'sentiment_score': self.emoji_to_sentiment.get(char, 0.0)
                })
        
        return emojis_found
    
    def analyze_message_emoji_patterns(self, message: str, sender_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Analyze emoji patterns in a single message."""
        emojis = self.extract_emojis_from_text(message)
        
        if not emojis:
            return {
                'has_emojis': False,
                'emoji_count': 0,
                'emotional_intensity': 0.0,
                'dominant_emotion': None
            }
        
        # Calculate emotional metrics
        sentiment_scores = [e['sentiment_score'] for e in emojis]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        
        # Find dominant emotion category
        emotion_counts = Counter(e['emotion_category'] for e in emojis if e['emotion_category'] != 'other')
        dominant_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        
        # Calculate emotional intensity (0-1)
        max_abs_sentiment = max(abs(score) for score in sentiment_scores) if sentiment_scores else 0.0
        emotional_intensity = min(max_abs_sentiment, 1.0)
        
        return {
            'has_emojis': True,
            'emoji_count': len(emojis),
            'unique_emojis': len(set(e['emoji'] for e in emojis)),
            'emojis_list': [e['emoji'] for e in emojis],
            'emotional_intensity': emotional_intensity,
            'avg_sentiment': avg_sentiment,
            'dominant_emotion': dominant_emotion,
            'emotion_categories': list(emotion_counts.keys()),
            'position_patterns': Counter(e['position_type'] for e in emojis),
            'sender_id': sender_id,
            'timestamp': timestamp
        }
    
    def detect_emoji_signatures(self, messages_df: pd.DataFrame, min_usage: int = 5) -> Dict[str, List[EmojiUsagePattern]]:
        """Detect personal emoji signatures for each sender."""
        logger.info("Detecting emoji signatures for senders")
        
        sender_emoji_patterns = defaultdict(Counter)
        sender_contexts = defaultdict(lambda: defaultdict(list))
        sender_message_counts = Counter()
        
        # Collect emoji usage data
        for _, row in messages_df.iterrows():
            body = row.get('body', '')
            sender_id = row.get('from_recipient_id')
            
            if not isinstance(body, str) or not sender_id:
                continue
            
            sender_message_counts[sender_id] += 1
            emojis = self.extract_emojis_from_text(body)
            
            for emoji_data in emojis:
                emoji_char = emoji_data['emoji']
                sender_emoji_patterns[sender_id][emoji_char] += 1
                sender_contexts[sender_id][emoji_char].append({
                    'context': emoji_data['context'],
                    'position': emoji_data['position_type'],
                    'message': body
                })
        
        # Generate emoji signatures
        signatures = {}
        
        for sender_id, emoji_counts in sender_emoji_patterns.items():
            sender_signatures = []
            total_messages = sender_message_counts.get(sender_id, 1)
            
            for emoji_char, count in emoji_counts.items():
                if count >= min_usage:
                    # Calculate usage frequency
                    frequency_rate = count / total_messages
                    
                    # Analyze usage contexts
                    contexts = sender_contexts[sender_id][emoji_char]
                    position_types = Counter(c['position'] for c in contexts)
                    most_common_position = position_types.most_common(1)[0][0] if position_types else 'mid_message'
                    
                    # Determine if this is a signature emoji (used significantly more than average)
                    is_signature = count >= min_usage and frequency_rate > 0.05  # 5% or more of messages
                    
                    pattern = EmojiUsagePattern(
                        emoji=emoji_char,
                        frequency=count,
                        emotional_category=self.emoji_to_category.get(emoji_char, 'other'),
                        usage_context=most_common_position,
                        sender_signature=is_signature
                    )
                    
                    sender_signatures.append(pattern)
            
            # Sort by frequency
            sender_signatures.sort(key=lambda x: x.frequency, reverse=True)
            signatures[str(sender_id)] = sender_signatures
        
        logger.info(f"Generated emoji signatures for {len(signatures)} senders")
        return signatures
    
    def analyze_emotional_state_correlation(self, messages_df: pd.DataFrame, time_window_minutes: int = 30) -> Dict[str, Any]:
        """Analyze how emoji usage correlates with emotional states over time."""
        logger.info("Analyzing emotional state correlation through emoji usage")
        
        # Convert timestamps and sort
        messages_df = messages_df.copy()
        messages_df['datetime'] = pd.to_datetime(messages_df['date_sent'], unit='ms')
        messages_df = messages_df.sort_values('datetime')
        
        emotional_timeline = []
        
        for sender_id in messages_df['from_recipient_id'].unique():
            sender_messages = messages_df[messages_df['from_recipient_id'] == sender_id]
            
            # Group messages into time windows
            for i in range(len(sender_messages)):
                message = sender_messages.iloc[i]
                window_start = message['datetime']
                window_end = window_start + pd.Timedelta(minutes=time_window_minutes)
                
                # Get messages in this time window
                window_messages = sender_messages[
                    (sender_messages['datetime'] >= window_start) & 
                    (sender_messages['datetime'] <= window_end)
                ]
                
                # Analyze emoji patterns in window
                window_emojis = []
                sentiment_scores = []
                
                for _, msg in window_messages.iterrows():
                    emoji_analysis = self.analyze_message_emoji_patterns(
                        msg.get('body', ''), 
                        sender_id, 
                        msg['datetime']
                    )
                    
                    if emoji_analysis['has_emojis']:
                        window_emojis.extend(emoji_analysis['emojis_list'])
                        sentiment_scores.append(emoji_analysis['avg_sentiment'])
                
                if window_emojis:
                    # Calculate window emotional metrics
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
                    emotion_categories = Counter(
                        self.emoji_to_category.get(e, 'other') for e in window_emojis
                    )
                    
                    emotional_timeline.append({
                        'sender_id': sender_id,
                        'window_start': window_start,
                        'window_end': window_end,
                        'emoji_count': len(window_emojis),
                        'unique_emojis': len(set(window_emojis)),
                        'avg_sentiment': avg_sentiment,
                        'dominant_emotions': emotion_categories.most_common(3),
                        'message_count': len(window_messages)
                    })
        
        return {
            'emotional_timeline': emotional_timeline,
            'analysis_window_minutes': time_window_minutes,
            'total_windows': len(emotional_timeline)
        }
    
    def generate_emoji_insights(self, messages_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive emoji insights from conversation data."""
        logger.info("Generating comprehensive emoji insights")
        
        # Basic emoji statistics
        total_messages = len(messages_df[messages_df['body'].notna()])
        messages_with_emojis = 0
        all_emojis = []
        
        for _, row in messages_df.iterrows():
            body = row.get('body', '')
            if isinstance(body, str):
                emojis = self.extract_emojis_from_text(body)
                if emojis:
                    messages_with_emojis += 1
                    all_emojis.extend([e['emoji'] for e in emojis])
        
        emoji_frequency = Counter(all_emojis)
        
        # Detect signatures
        emoji_signatures = self.detect_emoji_signatures(messages_df)
        
        # Emotional correlation
        emotional_correlation = self.analyze_emotional_state_correlation(messages_df)
        
        # Generate insights
        insights = {
            'usage_statistics': {
                'total_messages': total_messages,
                'messages_with_emojis': messages_with_emojis,
                'emoji_usage_rate': messages_with_emojis / total_messages if total_messages > 0 else 0,
                'total_emojis': len(all_emojis),
                'unique_emojis': len(emoji_frequency),
                'most_common_emojis': emoji_frequency.most_common(10)
            },
            'emoji_signatures': emoji_signatures,
            'emotional_patterns': emotional_correlation,
            'emotion_distribution': self._calculate_emotion_distribution(all_emojis),
            'sentiment_analysis': self._calculate_sentiment_trends(all_emojis)
        }
        
        logger.info("Emoji insights generated successfully")
        return insights
    
    def _calculate_emotion_distribution(self, emojis: List[str]) -> Dict[str, int]:
        """Calculate distribution of emotions across all emojis."""
        emotion_counts = defaultdict(int)
        
        for emoji_char in emojis:
            category = self.emoji_to_category.get(emoji_char, 'other')
            emotion_counts[category] += 1
        
        return dict(emotion_counts)
    
    def _calculate_sentiment_trends(self, emojis: List[str]) -> Dict[str, float]:
        """Calculate overall sentiment trends from emoji usage."""
        if not emojis:
            return {'avg_sentiment': 0.0, 'positive_ratio': 0.0, 'negative_ratio': 0.0}
        
        sentiment_scores = [self.emoji_to_sentiment.get(e, 0.0) for e in emojis]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        positive_count = sum(1 for score in sentiment_scores if score > 0.2)
        negative_count = sum(1 for score in sentiment_scores if score < -0.2)
        
        return {
            'avg_sentiment': avg_sentiment,
            'positive_ratio': positive_count / len(emojis),
            'negative_ratio': negative_count / len(emojis),
            'neutral_ratio': (len(emojis) - positive_count - negative_count) / len(emojis)
        }
    
    def interpret_emoji_in_context(self, emoji_char: str, message_context: str, sender_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provide context-aware interpretation of emoji usage."""
        base_interpretation = {
            'emoji': emoji_char,
            'base_emotion': self.emoji_to_category.get(emoji_char, 'other'),
            'base_sentiment': self.emoji_to_sentiment.get(emoji_char, 0.0),
            'emotional_tone': self.emoji_to_tone.get(emoji_char, EmotionalTone.CASUAL)
        }
        
        # Context-based modifications
        context_lower = message_context.lower()
        
        # Sarcasm detection
        sarcasm_indicators = ['yeah right', 'sure', 'totally', 'obviously', 'definitely']
        if any(indicator in context_lower for indicator in sarcasm_indicators):
            base_interpretation['context_modifier'] = 'sarcastic'
            base_interpretation['adjusted_sentiment'] = -base_interpretation['base_sentiment']
        
        # Emphasis detection
        if len([c for c in message_context if c == emoji_char]) > 1:
            base_interpretation['context_modifier'] = 'emphasized'
            base_interpretation['intensity_multiplier'] = 1.5
        
        # Question context
        if '?' in message_context:
            base_interpretation['context_modifier'] = 'questioning'
        
        return base_interpretation