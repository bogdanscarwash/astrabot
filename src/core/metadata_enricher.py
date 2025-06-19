"""
Metadata enrichment functions for Astrabot.

This module adds rich metadata to messages including temporal context,
emotional signals, group dynamics, and more.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime


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