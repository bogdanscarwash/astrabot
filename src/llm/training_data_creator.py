"""
Training Data Creation Module for Astrabot

This module provides functions to create high-quality training data from Signal conversations
for fine-tuning language models. It preserves natural conversation flow and personal
communication styles.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

from src.extractors.twitter_extractor import process_message_with_twitter_content
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingDataCreator:
    """Creates training data from Signal conversation exports."""
    
    def __init__(self, your_recipient_id: int = 2):
        """
        Initialize the training data creator.
        
        Args:
            your_recipient_id: Your recipient ID in the Signal database (default: 2)
        """
        self.your_recipient_id = your_recipient_id
        self.logger = get_logger(__name__)
        
    def create_conversation_windows(
        self, 
        messages_df: pd.DataFrame, 
        window_size: int = 5
    ) -> List[Dict]:
        """
        Create conversation windows that capture natural dialogue flow.
        
        Args:
            messages_df: DataFrame of messages
            window_size: Number of messages to include for context (default 5)
        
        Returns:
            List of conversation windows with rich metadata
        """
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
                if thread_messages.iloc[i + 1]['from_recipient_id'] != self.your_recipient_id:
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
                        'speaker': 'You' if msg['from_recipient_id'] == self.your_recipient_id else 'Other',
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
        
        self.logger.info(f"Created {len(conversation_windows)} conversation windows")
        return conversation_windows
    
    def segment_natural_dialogues(
        self, 
        messages_df: pd.DataFrame, 
        time_gap_minutes: int = 30
    ) -> List[Dict]:
        """
        Segment conversations into natural dialogue episodes based on time gaps.
        
        Args:
            messages_df: DataFrame of messages
            time_gap_minutes: Minutes of inactivity to consider new conversation episode
        
        Returns:
            List of conversation episodes with complete dialogue arcs
        """
        dialogue_episodes = []
        
        for thread_id in messages_df['thread_id'].unique():
            thread_messages = messages_df[
                messages_df['thread_id'] == thread_id
            ].sort_values('date_sent')
            
            if len(thread_messages) < 2:
                continue
            
            # Identify conversation episodes
            episodes = []
            current_episode = [thread_messages.iloc[0]]
            
            for i in range(1, len(thread_messages)):
                current_msg = thread_messages.iloc[i]
                prev_msg = thread_messages.iloc[i-1]
                
                # Check time gap
                time_gap = (current_msg['date_sent'] - prev_msg['date_sent']) / (1000 * 60)  # to minutes
                
                if time_gap > time_gap_minutes:
                    # New episode detected
                    if len(current_episode) >= 2:  # Only save meaningful episodes
                        episodes.append(current_episode)
                    current_episode = [current_msg]
                else:
                    current_episode.append(current_msg)
            
            # Don't forget the last episode
            if len(current_episode) >= 2:
                episodes.append(current_episode)
            
            # Process each episode
            for episode in episodes:
                # Analyze episode characteristics
                participants = set([msg['from_recipient_id'] for msg in episode])
                your_messages = [msg for msg in episode if msg['from_recipient_id'] == self.your_recipient_id]
                
                if not your_messages:  # Skip episodes where you didn't participate
                    continue
                
                # Detect conversation patterns
                turn_pattern = []
                current_speaker = episode[0]['from_recipient_id']
                turn_count = 1
                
                for msg in episode[1:]:
                    if msg['from_recipient_id'] != current_speaker:
                        turn_pattern.append(('You' if current_speaker == self.your_recipient_id else 'Other', turn_count))
                        current_speaker = msg['from_recipient_id']
                        turn_count = 1
                    else:
                        turn_count += 1
                turn_pattern.append(('You' if current_speaker == self.your_recipient_id else 'Other', turn_count))
                
                # Create episode data
                episode_data = {
                    'thread_id': thread_id,
                    'messages': [{
                        'speaker': 'You' if msg['from_recipient_id'] == self.your_recipient_id else 'Other',
                        'text': msg['body'],
                        'timestamp': msg['date_sent']
                    } for msg in episode],
                    'metadata': {
                        'episode_length': len(episode),
                        'duration_minutes': (episode[-1]['date_sent'] - episode[0]['date_sent']) / (1000 * 60),
                        'your_message_count': len(your_messages),
                        'turn_pattern': turn_pattern,
                        'initiated_by': 'You' if episode[0]['from_recipient_id'] == self.your_recipient_id else 'Other',
                        'ended_by': 'You' if episode[-1]['from_recipient_id'] == self.your_recipient_id else 'Other'
                    }
                }
                
                dialogue_episodes.append(episode_data)
        
        self.logger.info(f"Segmented into {len(dialogue_episodes)} dialogue episodes")
        return dialogue_episodes
    
    def analyze_personal_texting_style(self, messages_df: pd.DataFrame) -> Dict:
        """
        Analyze personal texting patterns to preserve authentic style.
        
        Args:
            messages_df: DataFrame of messages
            
        Returns:
            Dictionary containing style analysis
        """
        your_messages = messages_df[messages_df['from_recipient_id'] == self.your_recipient_id]
        
        # Analyze message patterns
        style_analysis = {
            'avg_message_length': your_messages['body'].str.len().mean(),
            'message_length_distribution': your_messages['body'].str.len().describe().to_dict(),
            'burst_patterns': self._analyze_message_bursts(your_messages),
            'preferred_length': 'lengthy' if your_messages['body'].str.len().mean() > 100 else 'concise',
            'emoji_usage': self._analyze_emoji_usage(your_messages),
            'total_messages': len(your_messages)
        }
        
        return style_analysis
    
    def _analyze_message_bursts(self, messages: pd.DataFrame) -> Dict:
        """Detect if you send multiple messages in quick succession."""
        messages = messages.sort_values('date_sent')
        
        bursts = []
        current_burst = []
        
        for i, (_, msg) in enumerate(messages.iterrows()):
            if i == 0:
                current_burst = [msg]
                continue
                
            time_diff = msg['date_sent'] - messages.iloc[i-1]['date_sent']
            
            # If less than 2 minutes apart, it's part of a burst
            if time_diff < 120000:  # 2 minutes in milliseconds
                current_burst.append(msg)
            else:
                if len(current_burst) > 1:
                    bursts.append(current_burst)
                current_burst = [msg]
        
        if len(current_burst) > 1:
            bursts.append(current_burst)
        
        return {
            'total_bursts': len(bursts),
            'avg_burst_size': sum(len(burst) for burst in bursts) / len(bursts) if bursts else 1,
            'burst_frequency': len(bursts) / len(messages) if len(messages) > 0 else 0,
            'max_burst_size': max(len(burst) for burst in bursts) if bursts else 1
        }
    
    def _analyze_emoji_usage(self, messages: pd.DataFrame) -> Dict:
        """Analyze emoji usage patterns."""
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        )
        
        total_messages = len(messages)
        messages_with_emojis = messages['body'].str.contains(emoji_pattern, regex=True, na=False).sum()
        
        return {
            'emoji_frequency': messages_with_emojis / total_messages if total_messages > 0 else 0,
            'messages_with_emojis': int(messages_with_emojis),
            'emoji_usage_rate': f"{(messages_with_emojis / total_messages * 100):.1f}%" if total_messages > 0 else "0%"
        }
    
    def create_training_examples(
        self,
        messages_df: pd.DataFrame,
        recipients_df: pd.DataFrame,
        include_twitter_content: bool = True,
        max_examples: Optional[int] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Create comprehensive training examples from conversations.
        
        Args:
            messages_df: DataFrame of messages
            recipients_df: DataFrame of recipients
            include_twitter_content: Whether to process Twitter links
            max_examples: Maximum number of examples to create
            
        Returns:
            Tuple of (training_examples, style_analysis)
        """
        self.logger.info("Creating training examples...")
        
        # Filter for meaningful text messages
        text_messages = messages_df[
            (messages_df['body'].notna()) & 
            (messages_df['body'].str.len() > 5)
        ].copy()
        
        # Create recipient lookup for names
        recipient_lookup = recipients_df.set_index('_id')['profile_given_name'].fillna('Unknown').to_dict()
        
        # Analyze personal style
        style_analysis = self.analyze_personal_texting_style(text_messages)
        
        training_examples = []
        
        # 1. Conversation Windows
        conv_windows = self.create_conversation_windows(text_messages)
        for window in conv_windows:
            # Format context as natural conversation
            context_text = "\n".join([
                f"{msg['speaker']}: {msg['text']}" 
                for msg in window['context']
            ])
            
            training_examples.append({
                'instruction': f"Continue this {window['metadata']['momentum']} conversation naturally",
                'input': context_text,
                'output': window['response']['text'],
                'metadata': {
                    'type': 'conversation_window',
                    'momentum': window['metadata']['momentum'],
                    'response_delay': window['metadata']['response_delay'],
                    'context_size': window['metadata']['context_size']
                }
            })
        
        # 2. Natural Dialogue Episodes
        episodes = self.segment_natural_dialogues(text_messages)
        
        # 3. Process with Twitter content if enabled
        if include_twitter_content:
            training_examples = self._enhance_with_twitter_content(training_examples, text_messages)
        
        # Limit examples if specified
        if max_examples and len(training_examples) > max_examples:
            training_examples = training_examples[:max_examples]
        
        self.logger.info(f"Created {len(training_examples)} training examples")
        
        return training_examples, style_analysis
    
    def _enhance_with_twitter_content(
        self,
        training_examples: List[Dict],
        messages_df: pd.DataFrame
    ) -> List[Dict]:
        """Enhance training examples with Twitter content."""
        # This is a simplified version - the full implementation would process
        # Twitter links and add enhanced content
        return training_examples
    
    def save_training_data(
        self,
        training_examples: List[Dict],
        output_path: str,
        style_analysis: Optional[Dict] = None
    ):
        """
        Save training data to JSON file.
        
        Args:
            training_examples: List of training examples
            output_path: Path to save the JSON file
            style_analysis: Optional style analysis to include
        """
        output_data = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'total_examples': len(training_examples),
            'examples': training_examples
        }
        
        if style_analysis:
            output_data['style_analysis'] = style_analysis
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(training_examples)} training examples to {output_path}")


def create_training_data_from_signal(
    messages_csv_path: str,
    recipients_csv_path: str,
    output_path: str,
    your_recipient_id: int = 2,
    include_twitter: bool = True,
    max_examples: Optional[int] = None
) -> Dict:
    """
    Convenience function to create training data from Signal CSV exports.
    
    Args:
        messages_csv_path: Path to signal.csv file
        recipients_csv_path: Path to recipient.csv file
        output_path: Path to save training data JSON
        your_recipient_id: Your recipient ID in Signal
        include_twitter: Whether to enhance with Twitter content
        max_examples: Maximum number of examples to create
        
    Returns:
        Dictionary with statistics about the created dataset
    """
    logger = get_logger(__name__)
    
    try:
        # Load data
        logger.info("Loading Signal data...")
        messages_df = pd.read_csv(messages_csv_path)
        recipients_df = pd.read_csv(recipients_csv_path)
        
        # Create training data
        creator = TrainingDataCreator(your_recipient_id)
        training_examples, style_analysis = creator.create_training_examples(
            messages_df,
            recipients_df,
            include_twitter_content=include_twitter,
            max_examples=max_examples
        )
        
        # Save results
        creator.save_training_data(training_examples, output_path, style_analysis)
        
        # Return statistics
        return {
            'success': True,
            'total_examples': len(training_examples),
            'style_analysis': style_analysis,
            'output_path': output_path
        }
        
    except Exception as e:
        logger.error(f"Failed to create training data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }