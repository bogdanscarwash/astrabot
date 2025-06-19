"""
Conversation Analysis Module for Astrabot.

This module provides comprehensive conversation analysis capabilities including:
- Natural dialogue flow analysis and window creation
- Conversation episode segmentation based on temporal patterns
- Role modeling and turn-taking analysis
- Personal texting style fingerprinting
- Message burst pattern detection
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import Counter

from src.models.schemas import TopicCategory, EmotionalTone, MessageType
from src.models.conversation_schemas import ConversationDynamics, RelationshipDynamic
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConversationAnalyzer:
    """Comprehensive conversation analysis for Signal chat data."""
    
    def __init__(self):
        """Initialize conversation analyzer with default settings."""
        self.default_window_size = 5
        self.default_time_gap_minutes = 30
        self.burst_threshold_seconds = 120
        logger.info("ConversationAnalyzer initialized")
    
    def create_conversation_windows(self, messages_df: pd.DataFrame, window_size: int = None, 
                                  your_recipient_id: int = 2) -> List[Dict[str, Any]]:
        """
        Create conversation windows that capture natural dialogue flow.
        
        Args:
            messages_df: DataFrame of messages
            window_size: Number of messages to include for context (default 5)
            your_recipient_id: Your recipient ID to identify your messages
        
        Returns:
            List of conversation windows with rich metadata
        """
        if window_size is None:
            window_size = self.default_window_size
            
        logger.info(f"Creating conversation windows with size {window_size}")
        conversation_windows = []
        
        # Group by thread for coherent conversations
        for thread_id in messages_df['thread_id'].unique():
            thread_messages = messages_df[
                messages_df['thread_id'] == thread_id
            ].sort_values('date_sent')
            
            if len(thread_messages) < 3:  # Need at least 3 messages for context
                continue
            
            # Create sliding windows through the conversation
            for i in range(len(thread_messages) - 1):
                # Check if you're the next speaker
                if thread_messages.iloc[i + 1]['from_recipient_id'] != your_recipient_id:
                    continue
                
                # Get context window
                start_idx = max(0, i - window_size + 1)
                context_messages = thread_messages.iloc[start_idx:i + 1]
                your_response = thread_messages.iloc[i + 1]
                
                # Calculate conversation dynamics
                time_gaps = []
                for j in range(1, len(context_messages)):
                    time_gap = (context_messages.iloc[j]['date_sent'] - 
                               context_messages.iloc[j-1]['date_sent']) / 1000  # Convert to seconds
                    time_gaps.append(time_gap)
                
                # Detect conversation momentum
                avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
                momentum = 'rapid' if avg_gap < 60 else 'moderate' if avg_gap < 300 else 'slow'
                
                # Build conversation window
                context = []
                for _, msg in context_messages.iterrows():
                    context.append({
                        'speaker': 'You' if msg['from_recipient_id'] == your_recipient_id else 'Other',
                        'text': msg['body'],
                        'timestamp': msg['date_sent'],
                        'has_media': bool(re.search(r'https?://\S+', msg['body']))
                    })
                
                window = {
                    'thread_id': thread_id,
                    'context': context,
                    'response': {
                        'text': your_response['body'],
                        'timestamp': your_response['date_sent']
                    },
                    'metadata': {
                        'momentum': momentum,
                        'context_size': len(context),
                        'avg_time_gap': avg_gap,
                        'response_delay': (your_response['date_sent'] - 
                                         context_messages.iloc[-1]['date_sent']) / 1000
                    }
                }
                
                conversation_windows.append(window)
        
        logger.info(f"Created {len(conversation_windows)} conversation windows")
        return conversation_windows

    def segment_natural_dialogues(self, messages_df: pd.DataFrame, time_gap_minutes: int = None,
                                your_recipient_id: int = 2) -> List[Dict[str, Any]]:
        """
        Segment conversations into natural dialogue episodes based on time gaps and context.
        
        Args:
            messages_df: DataFrame of messages
            time_gap_minutes: Minutes of inactivity to consider new conversation episode
            your_recipient_id: Your recipient ID
        
        Returns:
            List of conversation episodes with complete dialogue arcs
        """
        if time_gap_minutes is None:
            time_gap_minutes = self.default_time_gap_minutes
            
        logger.info(f"Segmenting dialogues with {time_gap_minutes} minute gap threshold")
        dialogue_episodes = []
        
        for thread_id in messages_df['thread_id'].unique():
            thread_messages = messages_df[
                messages_df['thread_id'] == thread_id
            ].sort_values('date_sent')
            
            if len(thread_messages) < 2:
                continue
            
            # Identify conversation episodes
            episodes = []
            current_episode = [thread_messages.iloc[0].to_dict()]
            
            for i in range(1, len(thread_messages)):
                current_msg = thread_messages.iloc[i]
                prev_msg = thread_messages.iloc[i-1]
                
                # Check time gap
                time_gap = (current_msg['date_sent'] - prev_msg['date_sent']) / (1000 * 60)  # to minutes
                
                if time_gap > time_gap_minutes:
                    # New episode detected
                    if len(current_episode) >= 2:  # Only save meaningful episodes
                        episodes.append(current_episode)
                    current_episode = [current_msg.to_dict()]
                else:
                    current_episode.append(current_msg.to_dict())
            
            # Don't forget the last episode
            if len(current_episode) >= 2:
                episodes.append(current_episode)
            
            # Process each episode
            for episode in episodes:
                # Analyze episode characteristics
                participants = set([msg['from_recipient_id'] for msg in episode])
                your_messages = [msg for msg in episode if msg['from_recipient_id'] == your_recipient_id]
                
                if not your_messages:  # Skip episodes where you didn't participate
                    continue
                
                # Detect conversation patterns
                turn_pattern = []
                current_speaker = episode[0]['from_recipient_id']
                turn_count = 1
                
                for msg in episode[1:]:
                    if msg['from_recipient_id'] != current_speaker:
                        turn_pattern.append(('You' if current_speaker == your_recipient_id else 'Other', turn_count))
                        current_speaker = msg['from_recipient_id']
                        turn_count = 1
                    else:
                        turn_count += 1
                turn_pattern.append(('You' if current_speaker == your_recipient_id else 'Other', turn_count))
                
                # Create episode data
                episode_data = {
                    'thread_id': thread_id,
                    'messages': [{
                        'speaker': 'You' if msg['from_recipient_id'] == your_recipient_id else 'Other',
                        'text': msg['body'],
                        'timestamp': msg['date_sent']
                    } for msg in episode],
                    'metadata': {
                        'episode_length': len(episode),
                        'duration_minutes': (episode[-1]['date_sent'] - episode[0]['date_sent']) / (1000 * 60),
                        'your_message_count': len(your_messages),
                        'turn_pattern': turn_pattern,
                        'initiated_by': 'You' if episode[0]['from_recipient_id'] == your_recipient_id else 'Other',
                        'ended_by': 'You' if episode[-1]['from_recipient_id'] == your_recipient_id else 'Other'
                    }
                }
                
                dialogue_episodes.append(episode_data)
        
        logger.info(f"Segmented {len(dialogue_episodes)} dialogue episodes")
        return dialogue_episodes

    def model_conversation_roles(self, dialogue_episodes: List[Dict[str, Any]], 
                               your_recipient_id: int = 2) -> List[Dict[str, Any]]:
        """
        Analyze and model conversation roles and dynamics.
        
        Args:
            dialogue_episodes: List of conversation episodes from segment_natural_dialogues
            your_recipient_id: Your recipient ID
        
        Returns:
            Conversation data with role patterns and dynamics
        """
        logger.info("Modeling conversation roles and dynamics")
        role_patterns = []
        
        for episode in dialogue_episodes:
            # Analyze conversation initiation patterns
            initiated_by_you = episode['metadata']['initiated_by'] == 'You'
            ended_by_you = episode['metadata']['ended_by'] == 'You'
            
            # Analyze turn-taking patterns
            turn_pattern = episode['metadata']['turn_pattern']
            your_turns = [turn for turn in turn_pattern if turn[0] == 'You']
            other_turns = [turn for turn in turn_pattern if turn[0] == 'Other']
            
            # Calculate conversation balance
            your_message_ratio = episode['metadata']['your_message_count'] / episode['metadata']['episode_length']
            
            # Detect conversation role
            if initiated_by_you and your_message_ratio > 0.6:
                role = 'conversation_driver'
            elif not initiated_by_you and your_message_ratio < 0.4:
                role = 'responsive_participant'
            elif len(your_turns) > len(other_turns):
                role = 'active_engager'
            else:
                role = 'balanced_conversationalist'
            
            # Extract conversation segments for training
            messages = episode['messages']
            
            # Find your responses with full context
            for i, msg in enumerate(messages):
                if msg['speaker'] == 'You' and i > 0:
                    # Get conversation context
                    context_start = max(0, i - 5)
                    context = messages[context_start:i]
                    
                    # Determine response type
                    if i == 1 and initiated_by_you:
                        response_type = 'continuation_after_initiation'
                    elif i == len(messages) - 1:
                        response_type = 'conversation_closer'
                    elif len([m for m in messages[i:i+3] if m['speaker'] == 'You']) >= 2:
                        response_type = 'burst_starter'
                    else:
                        response_type = 'turn_taking_response'
                    
                    role_data = {
                        'episode_id': f"{episode['thread_id']}_{messages[0]['timestamp']}",
                        'context': context,
                        'response': msg,
                        'role': role,
                        'response_type': response_type,
                        'metadata': {
                            'position_in_episode': i / len(messages),
                            'initiated_by_you': initiated_by_you,
                            'ended_by_you': ended_by_you,
                            'episode_duration': episode['metadata']['duration_minutes'],
                            'your_dominance': your_message_ratio
                        }
                    }
                    
                    role_patterns.append(role_data)
        
        logger.info(f"Identified {len(role_patterns)} role patterns")
        return role_patterns

    def analyze_personal_texting_style(self, messages_df: pd.DataFrame, your_recipient_id: int = 2) -> Dict[str, Any]:
        """
        Analyze your natural texting patterns to preserve your authentic style.
        
        Args:
            messages_df: DataFrame of messages
            your_recipient_id: Your recipient ID
        
        Returns:
            Dictionary containing style analysis metrics
        """
        logger.info("Analyzing personal texting style")
        your_messages = messages_df[messages_df['from_recipient_id'] == your_recipient_id]
        
        if len(your_messages) == 0:
            logger.warning("No messages found for the specified recipient ID")
            return {}
        
        # Analyze message patterns
        message_lengths = your_messages['body'].str.len()
        burst_analysis = self.analyze_message_bursts(your_messages)
        
        # Detect emoji usage
        emoji_messages = your_messages[your_messages['body'].str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', na=False)]
        emoji_frequency = len(emoji_messages) / len(your_messages) if len(your_messages) > 0 else 0
        
        # Detect URL sharing patterns
        url_messages = your_messages[your_messages['body'].str.contains(r'https?://', na=False)]
        url_frequency = len(url_messages) / len(your_messages) if len(your_messages) > 0 else 0
        
        # Analyze timing patterns
        your_messages_sorted = your_messages.sort_values('date_sent')
        response_times = []
        for i in range(1, len(your_messages_sorted)):
            time_diff = (your_messages_sorted.iloc[i]['date_sent'] - 
                        your_messages_sorted.iloc[i-1]['date_sent']) / 1000  # seconds
            if time_diff < 3600:  # Within 1 hour
                response_times.append(time_diff)
        
        avg_response_time = np.mean(response_times) if response_times else 0
        
        style_analysis = {
            'message_statistics': {
                'total_messages': len(your_messages),
                'avg_message_length': message_lengths.mean(),
                'median_message_length': message_lengths.median(),
                'message_length_std': message_lengths.std(),
                'message_length_distribution': {
                    'short': len(message_lengths[message_lengths <= 30]),
                    'medium': len(message_lengths[(message_lengths > 30) & (message_lengths <= 100)]),
                    'long': len(message_lengths[message_lengths > 100])
                }
            },
            'communication_patterns': {
                'burst_patterns': burst_analysis,
                'emoji_frequency': emoji_frequency,
                'url_sharing_frequency': url_frequency,
                'avg_response_time_seconds': avg_response_time
            },
            'style_classification': {
                'preferred_length': 'lengthy' if message_lengths.mean() > 100 else 'moderate' if message_lengths.mean() > 30 else 'concise',
                'burst_tendency': 'high' if burst_analysis['burst_frequency'] > 0.3 else 'moderate' if burst_analysis['burst_frequency'] > 0.1 else 'low',
                'multimedia_usage': 'high' if url_frequency > 0.2 else 'moderate' if url_frequency > 0.05 else 'low',
                'emoji_usage': 'frequent' if emoji_frequency > 0.3 else 'occasional' if emoji_frequency > 0.1 else 'rare'
            }
        }
        
        logger.info("Personal texting style analysis completed")
        return style_analysis

    def analyze_message_bursts(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if you send multiple messages in quick succession.
        
        Args:
            messages: DataFrame of messages from one sender
        
        Returns:
            Dictionary with burst pattern analysis
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

    def analyze_conversational_patterns(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the conversational training data to provide insights about communication patterns.
        
        Args:
            training_data: List of training examples
        
        Returns:
            Dictionary with pattern analysis
        """
        logger.info("Analyzing conversational patterns in training data")
        
        if not training_data:
            return {}
        
        # Type distribution
        type_counts = Counter(ex.get('metadata', {}).get('type', 'unknown') for ex in training_data)
        
        # Response delays
        delays = [ex.get('metadata', {}).get('response_delay', 0) for ex in training_data 
                  if ex.get('metadata', {}).get('response_delay') is not None]
        
        # Message lengths
        output_lengths = [len(ex.get('output', '')) for ex in training_data]
        
        # Conversation roles
        roles = [ex.get('metadata', {}).get('role', 'unknown') for ex in training_data 
                 if 'role' in ex.get('metadata', {})]
        role_counts = Counter(roles)
        
        # Burst sequences
        burst_sequences = [ex for ex in training_data if ex.get('metadata', {}).get('type') == 'burst_sequence']
        burst_lengths = [ex.get('metadata', {}).get('sequence_length', 0) for ex in burst_sequences]
        
        # Context sizes
        context_sizes = []
        for ex in training_data:
            context = ex.get('context', '')
            if isinstance(context, list):
                context_sizes.append(len(context))
            elif isinstance(context, str):
                context_sizes.append(len(context.split('\n')))
        
        analysis_result = {
            'dataset_size': len(training_data),
            'type_distribution': dict(type_counts),
            'response_metrics': {
                'avg_response_delay': np.mean(delays) if delays else 0,
                'median_response_delay': np.median(delays) if delays else 0,
                'response_delay_std': np.std(delays) if delays else 0
            },
            'message_metrics': {
                'avg_message_length': np.mean(output_lengths) if output_lengths else 0,
                'median_message_length': np.median(output_lengths) if output_lengths else 0,
                'message_length_std': np.std(output_lengths) if output_lengths else 0
            },
            'role_distribution': dict(role_counts),
            'burst_analysis': {
                'burst_frequency': len(burst_sequences) / len(training_data) if training_data else 0,
                'avg_burst_length': np.mean(burst_lengths) if burst_lengths else 0,
                'total_burst_sequences': len(burst_sequences)
            },
            'context_analysis': {
                'avg_context_size': np.mean(context_sizes) if context_sizes else 0,
                'median_context_size': np.median(context_sizes) if context_sizes else 0
            }
        }
        
        logger.info("Conversational pattern analysis completed")
        return analysis_result

    def generate_analysis_summary(self, messages_df: pd.DataFrame, your_recipient_id: int = 2) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis summary of the conversation data.
        
        Args:
            messages_df: DataFrame of messages
            your_recipient_id: Your recipient ID
            
        Returns:
            Dictionary with comprehensive analysis summary
        """
        logger.info("Generating comprehensive conversation analysis summary")
        
        # Basic statistics
        total_messages = len(messages_df)
        your_messages = messages_df[messages_df['from_recipient_id'] == your_recipient_id]
        other_messages = messages_df[messages_df['from_recipient_id'] != your_recipient_id]
        
        # Temporal analysis
        messages_with_time = messages_df.copy()
        messages_with_time['datetime'] = pd.to_datetime(messages_with_time['date_sent'], unit='ms')
        date_range = (messages_with_time['datetime'].max() - messages_with_time['datetime'].min()).days
        
        # Thread analysis
        thread_counts = messages_df['thread_id'].value_counts()
        
        # Generate component analyses
        style_analysis = self.analyze_personal_texting_style(messages_df, your_recipient_id)
        windows = self.create_conversation_windows(messages_df, your_recipient_id=your_recipient_id)
        episodes = self.segment_natural_dialogues(messages_df, your_recipient_id=your_recipient_id)
        
        summary = {
            'dataset_overview': {
                'total_messages': total_messages,
                'your_messages': len(your_messages),
                'other_messages': len(other_messages),
                'your_message_ratio': len(your_messages) / total_messages if total_messages > 0 else 0,
                'unique_threads': len(thread_counts),
                'date_range_days': date_range,
                'avg_messages_per_thread': thread_counts.mean(),
                'most_active_thread': thread_counts.index[0] if len(thread_counts) > 0 else None,
                'most_active_thread_messages': thread_counts.iloc[0] if len(thread_counts) > 0 else 0
            },
            'personal_style': style_analysis,
            'conversation_dynamics': {
                'total_windows': len(windows),
                'total_episodes': len(episodes),
                'avg_episode_length': np.mean([ep['metadata']['episode_length'] for ep in episodes]) if episodes else 0,
                'avg_episode_duration_minutes': np.mean([ep['metadata']['duration_minutes'] for ep in episodes]) if episodes else 0
            },
            'temporal_patterns': {
                'date_range': f"{messages_with_time['datetime'].min().date()} to {messages_with_time['datetime'].max().date()}",
                'peak_activity_hour': messages_with_time['datetime'].dt.hour.mode().iloc[0] if len(messages_with_time) > 0 else None,
                'messages_per_day': total_messages / max(date_range, 1)
            }
        }
        
        logger.info("Comprehensive analysis summary generated")
        return summary


# Backwards compatibility: Export functions for existing code
def create_conversation_windows(messages_df: pd.DataFrame, window_size: int = 5, 
                              your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.create_conversation_windows(messages_df, window_size, your_recipient_id)


def segment_natural_dialogues(messages_df: pd.DataFrame, time_gap_minutes: int = 30,
                            your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.segment_natural_dialogues(messages_df, time_gap_minutes, your_recipient_id)


def model_conversation_roles(dialogue_episodes: List[Dict[str, Any]], 
                           your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.model_conversation_roles(dialogue_episodes, your_recipient_id)


def analyze_personal_texting_style(messages_df: pd.DataFrame, your_recipient_id: int = 2) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.analyze_personal_texting_style(messages_df, your_recipient_id)


def analyze_message_bursts(messages: pd.DataFrame) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.analyze_message_bursts(messages)


def analyze_conversational_patterns(training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Backwards compatibility wrapper."""
    analyzer = ConversationAnalyzer()
    return analyzer.analyze_conversational_patterns(training_data)