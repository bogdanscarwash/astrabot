"""
Conversation analysis functions for Astrabot.

This module contains functions for analyzing and processing Signal conversations
to create natural training data that preserves communication style.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime


def create_conversation_windows(messages_df: pd.DataFrame, window_size: int = 5, 
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
    
    return conversation_windows


def segment_natural_dialogues(messages_df: pd.DataFrame, time_gap_minutes: int = 30,
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
    
    return dialogue_episodes


def model_conversation_roles(dialogue_episodes: List[Dict[str, Any]], 
                           your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """
    Analyze and model conversation roles and dynamics.
    
    Args:
        dialogue_episodes: List of conversation episodes from segment_natural_dialogues
        your_recipient_id: Your recipient ID
    
    Returns:
        Conversation data with role patterns and dynamics
    """
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
    
    return role_patterns


def analyze_personal_texting_style(messages_df: pd.DataFrame, your_recipient_id: int = 2) -> Dict[str, Any]:
    """
    Analyze your natural texting patterns to preserve your authentic style.
    
    Args:
        messages_df: DataFrame of messages
        your_recipient_id: Your recipient ID
    
    Returns:
        Dictionary containing style analysis metrics
    """
    your_messages = messages_df[messages_df['from_recipient_id'] == your_recipient_id]
    
    # Analyze message patterns
    style_analysis = {
        'avg_message_length': your_messages['body'].str.len().mean(),
        'message_length_distribution': your_messages['body'].str.len().describe().to_dict(),
        'burst_patterns': analyze_message_bursts(your_messages),
        'preferred_length': 'lengthy' if your_messages['body'].str.len().mean() > 100 else 'concise'
    }
    
    return style_analysis


def analyze_message_bursts(messages: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect if you send multiple messages in quick succession.
    
    Args:
        messages: DataFrame of messages from one sender
    
    Returns:
        Dictionary with burst pattern analysis
    """
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
    
    # Don't forget the last burst
    if len(current_burst) > 1:
        bursts.append(current_burst)
    
    return {
        'total_bursts': len(bursts),
        'avg_burst_size': sum(len(burst) for burst in bursts) / len(bursts) if bursts else 1,
        'burst_frequency': len(bursts) / len(messages) if len(messages) > 0 else 0
    }


def analyze_conversational_patterns(training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the conversational training data to provide insights about communication patterns.
    
    Args:
        training_data: List of training examples
    
    Returns:
        Dictionary with pattern analysis
    """
    from collections import Counter
    
    # Type distribution
    type_counts = Counter(ex['metadata']['type'] for ex in training_data)
    
    # Response delays
    delays = [ex['metadata'].get('response_delay', 0) for ex in training_data 
              if ex['metadata'].get('response_delay') is not None]
    
    # Message lengths
    output_lengths = [len(ex['output']) for ex in training_data]
    
    # Conversation roles
    roles = [ex['metadata'].get('role', 'unknown') for ex in training_data 
             if 'role' in ex['metadata']]
    role_counts = Counter(roles)
    
    # Burst sequences
    burst_sequences = [ex for ex in training_data if ex['metadata']['type'] == 'burst_sequence']
    burst_lengths = [ex['metadata']['sequence_length'] for ex in burst_sequences]
    
    return {
        'type_distribution': dict(type_counts),
        'avg_response_delay': sum(delays) / len(delays) if delays else 0,
        'avg_message_length': sum(output_lengths) / len(output_lengths),
        'role_distribution': dict(role_counts),
        'burst_frequency': len(burst_sequences) / len(training_data) if training_data else 0,
        'burst_lengths': burst_lengths
    }