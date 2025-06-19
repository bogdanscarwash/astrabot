"""
Communication Style Analysis Module for Astrabot.

This module provides comprehensive communication style analysis including:
- Individual communication pattern analysis 
- Style classification based on messaging behaviors
- Response pattern and timing analysis
- Emoji usage pattern detection
- Adaptation pattern analysis between conversation partners
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime

from src.models.schemas import TopicCategory, EmotionalTone, MessageType
from src.models.conversation_schemas import ConversationDynamics, RelationshipDynamic, MessageTiming
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StyleAnalyzer:
    """Comprehensive communication style analysis for Signal chat data."""
    
    def __init__(self):
        """Initialize style analyzer with default settings."""
        self.burst_threshold_seconds = 120  # 2 minutes
        self.min_messages_for_analysis = 50
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        )
        logger.info("StyleAnalyzer initialized")
    
    def analyze_all_communication_styles(self, messages_df: pd.DataFrame, recipients_df: pd.DataFrame,
                                       min_messages: int = None) -> Dict[int, Dict[str, Any]]:
        """
        Analyze communication styles for all frequent contacts.
        
        Args:
            messages_df: DataFrame of messages
            recipients_df: DataFrame of recipients
            min_messages: Minimum messages to include a contact
        
        Returns:
            Dictionary mapping recipient IDs to their communication style analysis
        """
        if min_messages is None:
            min_messages = self.min_messages_for_analysis
            
        logger.info(f"Analyzing communication styles with minimum {min_messages} messages")
        
        # Get recipient names for better readability
        recipient_lookup = recipients_df.set_index('_id')['profile_given_name'].fillna('Unknown').to_dict()
        
        communication_styles = {}
        
        # Analyze each frequent contact
        frequent_contacts = messages_df['from_recipient_id'].value_counts()
        frequent_contacts = frequent_contacts[frequent_contacts >= min_messages]
        
        logger.info(f"Analyzing communication styles for {len(frequent_contacts)} frequent contacts")
        
        for recipient_id in frequent_contacts.index:
            contact_messages = messages_df[messages_df['from_recipient_id'] == recipient_id]
            
            # Analyze their style
            style = {
                'name': recipient_lookup.get(recipient_id, f'Contact_{recipient_id}'),
                'total_messages': len(contact_messages),
                'avg_message_length': contact_messages['body'].str.len().mean(),
                'message_length_std': contact_messages['body'].str.len().std(),
                'burst_patterns': self.analyze_message_bursts(contact_messages),
                'preferred_times': self.analyze_timing_patterns(contact_messages),
                'emoji_usage': self.analyze_emoji_usage(contact_messages),
                'response_speed': self.analyze_response_patterns(contact_messages, messages_df)
            }
            
            # Classify communication style
            style['style_type'] = self.classify_communication_style(style)
            
            communication_styles[recipient_id] = style
        
        logger.info(f"Completed style analysis for {len(communication_styles)} contacts")
        return communication_styles

    def classify_communication_style(self, style_data: Dict[str, Any]) -> str:
        """
        Classify someone's communication style based on their patterns.
        
        Args:
            style_data: Dictionary with style metrics
        
        Returns:
            String classification of communication style
        """
        avg_length = style_data.get('avg_message_length', 0)
        burst_freq = style_data.get('burst_patterns', {}).get('burst_frequency', 0)
        avg_burst_size = style_data.get('burst_patterns', {}).get('avg_burst_size', 0)
        emoji_freq = style_data.get('emoji_usage', {}).get('emoji_frequency', 0)
        
        # Multi-dimensional classification
        if burst_freq > 0.4 and avg_burst_size > 3:
            if avg_length < 50:
                return "rapid_burst_chatter"  # Many short messages in quick succession
            else:
                return "verbose_burst_chatter"  # Multiple longer messages in succession
        elif avg_length > 200:
            if emoji_freq > 0.3:
                return "expressive_lengthy_texter"  # Long messages with lots of emojis
            else:
                return "formal_lengthy_texter"  # Long, detailed messages
        elif avg_length < 30:
            if emoji_freq > 0.5:
                return "emoji_heavy_texter"  # Short messages with lots of emojis
            else:
                return "concise_texter"  # Short, to-the-point messages
        elif burst_freq > 0.2:
            return "moderate_burst_chatter"  # Some bursting behavior
        elif emoji_freq > 0.4:
            return "expressive_communicator"  # High emoji usage
        else:
            return "balanced_communicator"  # Balanced approach

    def analyze_message_bursts(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze burst messaging patterns.
        
        Args:
            messages: DataFrame of messages from one sender
        
        Returns:
            Dictionary with burst pattern metrics
        """
        if len(messages) == 0:
            return {'total_bursts': 0, 'avg_burst_size': 0, 'burst_frequency': 0}
            
        messages = messages.sort_values('date_sent')
        
        bursts = []
        current_burst = []
        
        for i, (_, msg) in enumerate(messages.iterrows()):
            if i == 0:
                current_burst = [msg]
                continue
                
            time_diff = msg['date_sent'] - messages.iloc[i-1]['date_sent']
            
            # If less than burst threshold apart, it's part of a burst
            if time_diff < (self.burst_threshold_seconds * 1000):  # Convert to milliseconds
                current_burst.append(msg)
            else:
                if len(current_burst) > 1:
                    bursts.append(current_burst)
                current_burst = [msg]
        
        # Don't forget the last burst
        if len(current_burst) > 1:
            bursts.append(current_burst)
        
        # Analyze burst characteristics
        burst_sizes = [len(burst) for burst in bursts]
        burst_durations = []
        for burst in bursts:
            if len(burst) > 1:
                duration = (burst[-1]['date_sent'] - burst[0]['date_sent']) / 1000  # seconds
                burst_durations.append(duration)
        
        return {
            'total_bursts': len(bursts),
            'avg_burst_size': np.mean(burst_sizes) if burst_sizes else 0,
            'max_burst_size': max(burst_sizes) if burst_sizes else 0,
            'avg_burst_duration_seconds': np.mean(burst_durations) if burst_durations else 0,
            'burst_frequency': len(bursts) / len(messages) if len(messages) > 0 else 0,
            'messages_in_bursts': sum(burst_sizes),
            'burst_ratio': sum(burst_sizes) / len(messages) if len(messages) > 0 else 0
        }

    def analyze_emoji_usage(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze emoji usage patterns in messages.
        
        Args:
            messages: DataFrame of messages
        
        Returns:
            Dictionary with emoji usage metrics
        """
        if len(messages) == 0:
            return {'emoji_frequency': 0, 'messages_with_emojis': 0, 'total_messages': 0}
        
        total_messages = len(messages)
        messages_with_emojis = messages['body'].str.contains(self.emoji_pattern, regex=True, na=False).sum()
        
        # Count total emojis
        total_emojis = 0
        emoji_list = []
        for body in messages['body'].dropna():
            emojis_found = self.emoji_pattern.findall(str(body))
            total_emojis += len(emojis_found)
            emoji_list.extend(emojis_found)
        
        # Most common emojis
        emoji_counter = Counter(emoji_list)
        
        return {
            'emoji_frequency': messages_with_emojis / total_messages if total_messages > 0 else 0,
            'messages_with_emojis': messages_with_emojis,
            'total_messages': total_messages,
            'total_emojis': total_emojis,
            'avg_emojis_per_message': total_emojis / total_messages if total_messages > 0 else 0,
            'avg_emojis_per_emoji_message': total_emojis / messages_with_emojis if messages_with_emojis > 0 else 0,
            'most_common_emojis': emoji_counter.most_common(10)
        }

    def analyze_timing_patterns(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze when someone typically sends messages.
        
        Args:
            messages: DataFrame of messages
        
        Returns:
            Dictionary with timing pattern analysis
        """
        if len(messages) == 0:
            return {}
        
        # Convert timestamps to datetime
        messages_with_time = messages.copy()
        messages_with_time['datetime'] = pd.to_datetime(messages_with_time['date_sent'], unit='ms')
        messages_with_time['hour'] = messages_with_time['datetime'].dt.hour
        messages_with_time['day_of_week'] = messages_with_time['datetime'].dt.dayofweek
        
        hour_distribution = messages_with_time['hour'].value_counts().sort_index()
        peak_hours = hour_distribution.nlargest(3).index.tolist()
        
        # Calculate night owl / early bird tendencies
        night_hours = list(range(22, 24)) + list(range(0, 6))
        morning_hours = list(range(6, 10))
        afternoon_hours = list(range(12, 18))
        evening_hours = list(range(18, 22))
        
        night_messages = hour_distribution[hour_distribution.index.isin(night_hours)].sum()
        morning_messages = hour_distribution[hour_distribution.index.isin(morning_hours)].sum()
        afternoon_messages = hour_distribution[hour_distribution.index.isin(afternoon_hours)].sum()
        evening_messages = hour_distribution[hour_distribution.index.isin(evening_hours)].sum()
        
        total_messages = len(messages_with_time)
        
        # Day of week analysis (0=Monday, 6=Sunday)
        day_distribution = messages_with_time['day_of_week'].value_counts().sort_index()
        weekday_messages = day_distribution[day_distribution.index.isin(range(0, 5))].sum()  # Mon-Fri
        weekend_messages = day_distribution[day_distribution.index.isin([5, 6])].sum()  # Sat-Sun
        
        return {
            'peak_hours': peak_hours,
            'hour_distribution': hour_distribution.to_dict(),
            'time_periods': {
                'night_owl_ratio': night_messages / total_messages,
                'early_bird_ratio': morning_messages / total_messages,
                'afternoon_ratio': afternoon_messages / total_messages,
                'evening_ratio': evening_messages / total_messages
            },
            'activity_classification': self._classify_activity_pattern(
                night_messages, morning_messages, afternoon_messages, evening_messages, total_messages
            ),
            'day_patterns': {
                'weekday_ratio': weekday_messages / total_messages,
                'weekend_ratio': weekend_messages / total_messages,
                'day_distribution': day_distribution.to_dict()
            }
        }
    
    def _classify_activity_pattern(self, night: int, morning: int, afternoon: int, evening: int, total: int) -> str:
        """Classify activity pattern based on time distribution."""
        if total == 0:
            return "unknown"
        
        ratios = {
            'night': night / total,
            'morning': morning / total,
            'afternoon': afternoon / total,
            'evening': evening / total
        }
        
        max_period = max(ratios, key=ratios.get)
        max_ratio = ratios[max_period]
        
        if max_ratio > 0.4:
            return f"{max_period}_person"
        elif ratios['night'] > 0.3:
            return "night_owl"
        elif ratios['morning'] > 0.3:
            return "early_bird"
        else:
            return "balanced_schedule"

    def analyze_response_patterns(self, contact_messages: pd.DataFrame, all_messages: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze how quickly someone responds to messages.
        
        Args:
            contact_messages: Messages from the contact
            all_messages: All messages in the dataset
        
        Returns:
            Dictionary with response pattern metrics
        """
        response_times = []
        contact_id = contact_messages['from_recipient_id'].iloc[0] if len(contact_messages) > 0 else None
        
        if contact_id is None:
            return {
                'avg_response_time_minutes': None,
                'median_response_time_minutes': None,
                'response_speed_classification': 'unknown',
                'quick_responder': False,
                'total_responses_analyzed': 0
            }
        
        # Analyze response times in each thread
        for thread_id in contact_messages['thread_id'].unique():
            thread_msgs = all_messages[all_messages['thread_id'] == thread_id].sort_values('date_sent')
            
            for i in range(len(thread_msgs) - 1):
                current_msg = thread_msgs.iloc[i]
                next_msg = thread_msgs.iloc[i + 1]
                
                # If this contact is responding to someone else
                if (current_msg['from_recipient_id'] != contact_id and 
                    next_msg['from_recipient_id'] == contact_id):
                    
                    response_time = (next_msg['date_sent'] - current_msg['date_sent']) / (1000 * 60)  # minutes
                    if response_time < 1440:  # Less than 24 hours
                        response_times.append(response_time)
        
        if response_times:
            avg_response_time = np.mean(response_times)
            median_response_time = np.median(response_times)
            
            # Classify response speed
            if avg_response_time < 5:
                speed_class = "instant_responder"
            elif avg_response_time < 30:
                speed_class = "quick_responder"
            elif avg_response_time < 120:
                speed_class = "moderate_responder"
            elif avg_response_time < 360:
                speed_class = "slow_responder"
            else:
                speed_class = "delayed_responder"
            
            return {
                'avg_response_time_minutes': avg_response_time,
                'median_response_time_minutes': median_response_time,
                'response_speed_classification': speed_class,
                'quick_responder': avg_response_time < 30,
                'total_responses_analyzed': len(response_times),
                'response_time_distribution': {
                    'under_5_minutes': len([t for t in response_times if t < 5]),
                    'under_30_minutes': len([t for t in response_times if t < 30]),
                    'under_2_hours': len([t for t in response_times if t < 120]),
                    'over_2_hours': len([t for t in response_times if t >= 120])
                }
            }
        else:
            return {
                'avg_response_time_minutes': None,
                'median_response_time_minutes': None,
                'response_speed_classification': 'unknown',
                'quick_responder': False,
                'total_responses_analyzed': 0
            }

    def create_adaptation_context(self, current_msg: pd.Series, your_response: pd.Series,
                                other_person_style: Dict[str, Any]) -> List[str]:
        """
        Create context about how you're adapting to their communication style.
        
        Args:
            current_msg: The message you're responding to
            your_response: Your response message
            other_person_style: Style analysis of the other person
        
        Returns:
            List of adaptation patterns detected
        """
        adaptations = []
        
        other_style = other_person_style.get('style_type', 'unknown')
        other_length = len(str(current_msg['body']))
        your_length = len(str(your_response['body']))
        
        # Analyze length adaptation
        if other_style in ['formal_lengthy_texter', 'expressive_lengthy_texter'] and your_length > 100:
            adaptations.append("matching_lengthy_style")
        elif other_style == 'concise_texter' and your_length < 50:
            adaptations.append("matching_concise_style")
        elif other_style in ['rapid_burst_chatter', 'verbose_burst_chatter']:
            adaptations.append("responding_to_burst_chatter")
        elif abs(your_length - other_length) < 20:  # Similar length
            adaptations.append("length_mirroring")
        
        # Analyze emoji adaptation
        other_emoji_freq = other_person_style.get('emoji_usage', {}).get('emoji_frequency', 0)
        your_has_emoji = bool(self.emoji_pattern.search(str(your_response['body'])))
        other_has_emoji = bool(self.emoji_pattern.search(str(current_msg['body'])))
        
        if other_emoji_freq > 0.3 and your_has_emoji:
            adaptations.append("matching_emoji_usage")
        elif other_has_emoji and your_has_emoji:
            adaptations.append("emoji_mirroring")
        elif other_style == 'emoji_heavy_texter' and your_has_emoji:
            adaptations.append("adapting_to_emoji_heavy_style")
        
        # Analyze timing adaptation (if possible)
        response_delay = (your_response['date_sent'] - current_msg['date_sent']) / (1000 * 60)  # minutes
        other_response_speed = other_person_style.get('response_speed', {}).get('response_speed_classification', 'unknown')
        
        if other_response_speed == 'quick_responder' and response_delay < 30:
            adaptations.append("matching_quick_response")
        elif other_response_speed in ['instant_responder', 'quick_responder'] and response_delay < 10:
            adaptations.append("matching_immediate_response")
        
        return adaptations

    def analyze_your_adaptation_patterns(self, training_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how you adapt to different communication styles.
        
        Args:
            training_data: List of training examples with adaptation data
        
        Returns:
            Dictionary mapping style types to adaptation analysis
        """
        logger.info("Analyzing adaptation patterns in training data")
        adaptation_analysis = {}
        
        for example in training_data:
            other_style = example.get('other_person_style', 'unknown')
            your_response_length = len(example.get('response', ''))
            adaptations = example.get('adaptation_context', [])
            
            if other_style not in adaptation_analysis:
                adaptation_analysis[other_style] = {
                    'total_examples': 0,
                    'avg_response_length': [],
                    'adaptation_types': {},
                    'example_responses': [],
                    'adaptation_frequency': {}
                }
            
            style_data = adaptation_analysis[other_style]
            style_data['total_examples'] += 1
            style_data['avg_response_length'].append(your_response_length)
            
            for adaptation in adaptations:
                style_data['adaptation_types'][adaptation] = \
                    style_data['adaptation_types'].get(adaptation, 0) + 1
            
            # Store some example responses
            if len(style_data['example_responses']) < 3:
                response_text = example.get('response', '')[:100]
                if response_text:
                    style_data['example_responses'].append(response_text)
        
        # Calculate statistics
        for style_type, style_data in adaptation_analysis.items():
            if style_data['avg_response_length']:
                style_data['avg_response_length'] = np.mean(style_data['avg_response_length'])
            else:
                style_data['avg_response_length'] = 0
            
            # Calculate adaptation frequencies
            total_examples = style_data['total_examples']
            for adaptation_type, count in style_data['adaptation_types'].items():
                style_data['adaptation_frequency'][adaptation_type] = count / total_examples
        
        logger.info(f"Analyzed adaptation patterns for {len(adaptation_analysis)} style types")
        return adaptation_analysis

    def generate_style_summary(self, communication_styles: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of communication styles analysis.
        
        Args:
            communication_styles: Output from analyze_all_communication_styles
            
        Returns:
            Dictionary with style analysis summary
        """
        if not communication_styles:
            return {}
        
        # Style type distribution
        style_types = [style['style_type'] for style in communication_styles.values()]
        style_distribution = Counter(style_types)
        
        # Average metrics
        avg_message_lengths = [style['avg_message_length'] for style in communication_styles.values()]
        burst_frequencies = [style['burst_patterns']['burst_frequency'] for style in communication_styles.values()]
        emoji_frequencies = [style['emoji_usage']['emoji_frequency'] for style in communication_styles.values()]
        
        # Response speed distribution
        response_speeds = []
        for style in communication_styles.values():
            speed_class = style.get('response_speed', {}).get('response_speed_classification', 'unknown')
            if speed_class != 'unknown':
                response_speeds.append(speed_class)
        
        response_speed_distribution = Counter(response_speeds)
        
        return {
            'total_contacts_analyzed': len(communication_styles),
            'style_type_distribution': dict(style_distribution),
            'average_metrics': {
                'avg_message_length': np.mean(avg_message_lengths) if avg_message_lengths else 0,
                'avg_burst_frequency': np.mean(burst_frequencies) if burst_frequencies else 0,
                'avg_emoji_frequency': np.mean(emoji_frequencies) if emoji_frequencies else 0
            },
            'response_speed_distribution': dict(response_speed_distribution),
            'most_common_style': style_distribution.most_common(1)[0] if style_distribution else None,
            'style_diversity': len(style_distribution)
        }


# Backwards compatibility: Export functions for existing code
def analyze_all_communication_styles(messages_df: pd.DataFrame, recipients_df: pd.DataFrame,
                                   min_messages: int = 50) -> Dict[int, Dict[str, Any]]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_all_communication_styles(messages_df, recipients_df, min_messages)


def classify_communication_style(style_data: Dict[str, Any]) -> str:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.classify_communication_style(style_data)


def analyze_message_bursts(messages: pd.DataFrame) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_message_bursts(messages)


def analyze_emoji_usage(messages: pd.DataFrame) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_emoji_usage(messages)


def analyze_timing_patterns(messages: pd.DataFrame) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_timing_patterns(messages)


def analyze_response_patterns(contact_messages: pd.DataFrame, all_messages: pd.DataFrame) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_response_patterns(contact_messages, all_messages)


def create_adaptation_context(current_msg: pd.Series, your_response: pd.Series,
                            other_person_style: Dict[str, Any]) -> List[str]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.create_adaptation_context(current_msg, your_response, other_person_style)


def analyze_your_adaptation_patterns(training_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Backwards compatibility wrapper."""
    analyzer = StyleAnalyzer()
    return analyzer.analyze_your_adaptation_patterns(training_data)