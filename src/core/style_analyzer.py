"""
Communication style analysis for Astrabot.

This module analyzes communication styles of different people and how you adapt
your communication style when talking to them.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter


def analyze_all_communication_styles(messages_df: pd.DataFrame, recipients_df: pd.DataFrame,
                                   min_messages: int = 50) -> Dict[int, Dict[str, Any]]:
    """
    Analyze communication styles for all frequent contacts.
    
    Args:
        messages_df: DataFrame of messages
        recipients_df: DataFrame of recipients
        min_messages: Minimum messages to include a contact
    
    Returns:
        Dictionary mapping recipient IDs to their communication style analysis
    """
    # Get recipient names for better readability
    recipient_lookup = recipients_df.set_index('_id')['profile_given_name'].fillna('Unknown').to_dict()
    
    communication_styles = {}
    
    # Analyze each frequent contact
    frequent_contacts = messages_df['from_recipient_id'].value_counts()
    frequent_contacts = frequent_contacts[frequent_contacts >= min_messages]
    
    print(f"Analyzing communication styles for {len(frequent_contacts)} frequent contacts...")
    
    for recipient_id in frequent_contacts.index:
        contact_messages = messages_df[messages_df['from_recipient_id'] == recipient_id]
        
        # Analyze their style
        style = {
            'name': recipient_lookup.get(recipient_id, f'Contact_{recipient_id}'),
            'total_messages': len(contact_messages),
            'avg_message_length': contact_messages['body'].str.len().mean(),
            'message_length_std': contact_messages['body'].str.len().std(),
            'burst_patterns': analyze_message_bursts(contact_messages),
            'preferred_times': analyze_timing_patterns(contact_messages),
            'emoji_usage': analyze_emoji_usage(contact_messages),
            'response_speed': analyze_response_patterns(contact_messages, messages_df)
        }
        
        # Classify communication style
        style['style_type'] = classify_communication_style(style)
        
        communication_styles[recipient_id] = style
    
    return communication_styles


def classify_communication_style(style_data: Dict[str, Any]) -> str:
    """
    Classify someone's communication style based on their patterns.
    
    Args:
        style_data: Dictionary with style metrics
    
    Returns:
        String classification of communication style
    """
    avg_length = style_data['avg_message_length']
    burst_freq = style_data['burst_patterns']['burst_frequency']
    avg_burst_size = style_data['burst_patterns']['avg_burst_size']
    
    if burst_freq > 0.4 and avg_burst_size > 3:
        if avg_length < 50:
            return "rapid_burst_chatter"  # Many short messages in quick succession
        else:
            return "verbose_burst_chatter"  # Multiple longer messages in succession
    elif avg_length > 200:
        return "lengthy_texter"  # Long, detailed messages
    elif avg_length < 30:
        return "concise_texter"  # Short, to-the-point messages
    elif burst_freq > 0.2:
        return "moderate_burst_chatter"  # Some bursting behavior
    else:
        return "balanced_communicator"  # Balanced approach


def analyze_message_bursts(messages: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze burst messaging patterns.
    
    Args:
        messages: DataFrame of messages from one sender
    
    Returns:
        Dictionary with burst pattern metrics
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


def analyze_emoji_usage(messages: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze emoji usage patterns in messages.
    
    Args:
        messages: DataFrame of messages
    
    Returns:
        Dictionary with emoji usage metrics
    """
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    )
    
    total_messages = len(messages)
    messages_with_emojis = messages['body'].str.contains(emoji_pattern, regex=True, na=False).sum()
    
    return {
        'emoji_frequency': messages_with_emojis / total_messages if total_messages > 0 else 0,
        'messages_with_emojis': messages_with_emojis,
        'total_messages': total_messages
    }


def analyze_timing_patterns(messages: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze when someone typically sends messages.
    
    Args:
        messages: DataFrame of messages
    
    Returns:
        Dictionary with timing pattern analysis
    """
    messages['hour'] = pd.to_datetime(messages['date_sent'], unit='ms').dt.hour
    
    hour_distribution = messages['hour'].value_counts().sort_index()
    peak_hours = hour_distribution.nlargest(3).index.tolist()
    
    # Calculate night owl / early bird tendencies
    night_hours = list(range(22, 24)) + list(range(0, 6))
    morning_hours = list(range(6, 10))
    
    night_messages = hour_distribution[hour_distribution.index.isin(night_hours)].sum()
    morning_messages = hour_distribution[hour_distribution.index.isin(morning_hours)].sum()
    
    return {
        'peak_hours': peak_hours,
        'hour_distribution': hour_distribution.to_dict(),
        'night_owl': night_messages > len(messages) * 0.3,
        'early_bird': morning_messages > len(messages) * 0.3
    }


def analyze_response_patterns(contact_messages: pd.DataFrame, all_messages: pd.DataFrame) -> Dict[str, Any]:
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
            'quick_responder': False,
            'total_responses_analyzed': 0
        }
    
    for thread_id in contact_messages['thread_id'].unique():
        thread_msgs = all_messages[all_messages['thread_id'] == thread_id].sort_values('date_sent')
        
        for i in range(len(thread_msgs) - 1):
            current_msg = thread_msgs.iloc[i]
            next_msg = thread_msgs.iloc[i + 1]
            
            # If this contact is responding to someone else
            if (current_msg['from_recipient_id'] != contact_id and 
                next_msg['from_recipient_id'] == contact_id):
                
                response_time = next_msg['date_sent'] - current_msg['date_sent']
                response_times.append(response_time)
    
    if response_times:
        avg_response_time = np.mean(response_times) / (1000 * 60)  # Convert to minutes
        return {
            'avg_response_time_minutes': avg_response_time,
            'quick_responder': avg_response_time < 30,  # Responds within 30 minutes on average
            'total_responses_analyzed': len(response_times)
        }
    else:
        return {
            'avg_response_time_minutes': None,
            'quick_responder': False,
            'total_responses_analyzed': 0
        }


def create_adaptation_context(current_msg: pd.Series, your_response: pd.Series,
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
    other_length = len(current_msg['body'])
    your_length = len(your_response['body'])
    
    # Analyze length adaptation
    if other_style == 'lengthy_texter' and your_length > 100:
        adaptations.append("matching_lengthy_style")
    elif other_style == 'concise_texter' and your_length < 50:
        adaptations.append("matching_concise_style")
    elif other_style in ['rapid_burst_chatter', 'verbose_burst_chatter']:
        adaptations.append("responding_to_burst_chatter")
    
    # Analyze emoji adaptation
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
    )
    
    other_emoji_freq = other_person_style.get('emoji_usage', {}).get('emoji_frequency', 0)
    your_has_emoji = bool(emoji_pattern.search(your_response['body']))
    
    if other_emoji_freq > 0.3 and your_has_emoji:
        adaptations.append("matching_emoji_usage")
    
    return adaptations


def analyze_your_adaptation_patterns(training_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze how you adapt to different communication styles.
    
    Args:
        training_data: List of training examples with adaptation data
    
    Returns:
        Dictionary mapping style types to adaptation analysis
    """
    adaptation_analysis = {}
    
    for example in training_data:
        other_style = example.get('other_person_style', 'unknown')
        your_response_length = len(example['response'])
        adaptations = example.get('adaptation_context', [])
        
        if other_style not in adaptation_analysis:
            adaptation_analysis[other_style] = {
                'total_examples': 0,
                'avg_response_length': [],
                'adaptation_types': {},
                'example_responses': []
            }
        
        adaptation_analysis[other_style]['total_examples'] += 1
        adaptation_analysis[other_style]['avg_response_length'].append(your_response_length)
        
        for adaptation in adaptations:
            adaptation_analysis[other_style]['adaptation_types'][adaptation] = \
                adaptation_analysis[other_style]['adaptation_types'].get(adaptation, 0) + 1
        
        # Store some example responses
        if len(adaptation_analysis[other_style]['example_responses']) < 3:
            adaptation_analysis[other_style]['example_responses'].append(example['response'][:100])
    
    # Calculate averages
    for style_data in adaptation_analysis.values():
        if style_data['avg_response_length']:
            style_data['avg_response_length'] = np.mean(style_data['avg_response_length'])
        else:
            style_data['avg_response_length'] = 0
    
    return adaptation_analysis