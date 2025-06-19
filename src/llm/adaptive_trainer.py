"""
Adaptive training functions for Astrabot.

This module provides functions for creating training data that captures how you
adapt your communication style to different people.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter

from src.core.style_analyzer import (
    analyze_all_communication_styles,
    create_adaptation_context
)


def create_adaptive_training_data(messages_df: pd.DataFrame, recipients_df: pd.DataFrame, 
                                communication_styles: Dict[int, Dict[str, Any]], 
                                your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """
    Create training data that captures how you adapt to different communication styles.
    
    Args:
        messages_df: DataFrame of messages
        recipients_df: DataFrame of recipients
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID
    
    Returns:
        List of adaptive training examples
    """
    training_data = []
    
    print("Creating adaptive training examples...")
    
    # Group by thread and create conversations
    for thread_id in messages_df['thread_id'].unique():
        thread_messages = messages_df[
            messages_df['thread_id'] == thread_id
        ].sort_values('date_sent')
        
        if len(thread_messages) < 2:
            continue
            
        # Identify the other person in this conversation
        other_participants = thread_messages[
            thread_messages['from_recipient_id'] != your_recipient_id
        ]['from_recipient_id'].unique()
        
        if len(other_participants) != 1:  # Skip group chats for now
            continue
            
        other_person_id = other_participants[0]
        other_person_style = communication_styles.get(other_person_id, {})
        
        # Create conversation pairs
        for i in range(len(thread_messages) - 1):
            current_msg = thread_messages.iloc[i]
            next_msg = thread_messages.iloc[i + 1]
            
            # Only create training examples where you're responding
            if next_msg['from_recipient_id'] == your_recipient_id:
                
                # Build context with style awareness
                context_start = max(0, i - 4)  # Include more context for style adaptation
                context_messages = thread_messages.iloc[context_start:i+1]
                
                # Format conversation with style indicators
                conversation_context = []
                for _, msg in context_messages.iterrows():
                    if msg['from_recipient_id'] == your_recipient_id:
                        sender_name = "You"
                    else:
                        sender_name = other_person_style.get('name', 'Other')
                        # Add style indicator for the other person's messages
                        style_type = other_person_style.get('style_type', 'unknown')
                        if style_type in ['rapid_burst_chatter', 'verbose_burst_chatter']:
                            sender_name += " (burst chatter)"
                        elif style_type == 'lengthy_texter':
                            sender_name += " (lengthy texter)"
                        elif style_type == 'concise_texter':
                            sender_name += " (concise texter)"
                    
                    conversation_context.append(f"{sender_name}: {msg['body']}")
                
                # Create enhanced training example
                training_example = {
                    'instruction': "\n".join(conversation_context),
                    'response': next_msg['body'],
                    'thread_id': thread_id,
                    'timestamp': next_msg['date_sent'],
                    'other_person_style': other_person_style.get('style_type', 'unknown'),
                    'other_person_name': other_person_style.get('name', 'Unknown'),
                    'adaptation_context': create_adaptation_context(current_msg, next_msg, other_person_style)
                }
                
                training_data.append(training_example)
    
    print(f"Created {len(training_data)} adaptive training examples")
    
    # Show breakdown by communication styles
    style_breakdown = {}
    for example in training_data:
        style = example['other_person_style']
        style_breakdown[style] = style_breakdown.get(style, 0) + 1
    
    print("\nTraining examples by communication style:")
    for style, count in sorted(style_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {style}: {count} examples ({count/len(training_data)*100:.1f}%)")
    
    return training_data


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


def create_style_aware_instructions(training_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create training examples with explicit style-aware instructions.
    
    Args:
        training_data: List of adaptive training examples
    
    Returns:
        List of examples with style-aware instructions
    """
    style_aware_data = []
    
    # Define instruction templates for different styles
    style_instructions = {
        'rapid_burst_chatter': "The other person tends to send many short messages quickly. Respond appropriately.",
        'verbose_burst_chatter': "The other person sends multiple detailed messages in succession. Craft your response accordingly.",
        'lengthy_texter': "The other person writes long, detailed messages. Consider matching their communication depth.",
        'concise_texter': "The other person keeps messages brief and to the point. Be concise in your response.",
        'moderate_burst_chatter': "The other person sometimes sends multiple messages. Respond naturally.",
        'balanced_communicator': "Have a natural, balanced conversation.",
        'unknown': "Continue the conversation naturally."
    }
    
    for example in training_data:
        style = example.get('other_person_style', 'unknown')
        style_instruction = style_instructions.get(style, style_instructions['unknown'])
        
        # Create new example with style-aware instruction
        new_example = {
            'instruction': style_instruction,
            'input': example['instruction'],  # The conversation context
            'output': example['response'],
            'metadata': {
                'other_person_style': style,
                'other_person_name': example.get('other_person_name', 'Unknown'),
                'adaptations': example.get('adaptation_context', [])
            }
        }
        
        style_aware_data.append(new_example)
    
    return style_aware_data


def analyze_style_matching_patterns(messages_df: pd.DataFrame, communication_styles: Dict[int, Dict[str, Any]], 
                                  your_recipient_id: int = 2) -> Dict[str, Any]:
    """
    Analyze how well you match different communication styles.
    
    Args:
        messages_df: DataFrame of messages
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID
    
    Returns:
        Dictionary with style matching analysis
    """
    matching_analysis = {
        'style_matching_scores': {},
        'adaptation_examples': {},
        'summary_stats': {}
    }
    
    # Analyze conversations with each person
    for recipient_id, their_style in communication_styles.items():
        # Get conversations between you and this person
        conversations = messages_df[
            ((messages_df['from_recipient_id'] == your_recipient_id) & 
             (messages_df['to_recipient_id'] == recipient_id)) |
            ((messages_df['from_recipient_id'] == recipient_id) & 
             (messages_df['to_recipient_id'] == your_recipient_id))
        ].sort_values('date_sent')
        
        if len(conversations) < 10:  # Need enough messages for analysis
            continue
        
        # Analyze your messages to this person
        your_messages_to_them = conversations[
            conversations['from_recipient_id'] == your_recipient_id
        ]
        
        if len(your_messages_to_them) == 0:
            continue
        
        # Calculate style matching metrics
        their_avg_length = their_style.get('avg_message_length', 100)
        your_avg_length_to_them = your_messages_to_them['body'].str.len().mean()
        
        # Length matching score (0-1, where 1 is perfect match)
        length_diff_ratio = abs(their_avg_length - your_avg_length_to_them) / max(their_avg_length, your_avg_length_to_them)
        length_matching_score = 1 - min(length_diff_ratio, 1)
        
        # Emoji matching
        their_emoji_freq = their_style.get('emoji_usage', {}).get('emoji_frequency', 0)
        emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+'
        )
        your_emoji_messages = your_messages_to_them['body'].str.contains(emoji_pattern, regex=True, na=False).sum()
        your_emoji_freq = your_emoji_messages / len(your_messages_to_them) if len(your_messages_to_them) > 0 else 0
        
        emoji_diff = abs(their_emoji_freq - your_emoji_freq)
        emoji_matching_score = 1 - min(emoji_diff, 1)
        
        # Overall matching score
        overall_score = (length_matching_score + emoji_matching_score) / 2
        
        person_name = their_style.get('name', f'Person_{recipient_id}')
        
        matching_analysis['style_matching_scores'][person_name] = {
            'their_style': their_style.get('style_type', 'unknown'),
            'length_matching': length_matching_score,
            'emoji_matching': emoji_matching_score,
            'overall_matching': overall_score,
            'your_avg_length': your_avg_length_to_them,
            'their_avg_length': their_avg_length,
            'your_emoji_freq': your_emoji_freq,
            'their_emoji_freq': their_emoji_freq
        }
    
    # Calculate summary statistics
    if matching_analysis['style_matching_scores']:
        all_scores = [data['overall_matching'] for data in matching_analysis['style_matching_scores'].values()]
        matching_analysis['summary_stats'] = {
            'avg_matching_score': np.mean(all_scores),
            'best_match': max(matching_analysis['style_matching_scores'].items(), 
                            key=lambda x: x[1]['overall_matching'])[0],
            'worst_match': min(matching_analysis['style_matching_scores'].items(), 
                             key=lambda x: x[1]['overall_matching'])[0],
            'adaptation_range': max(all_scores) - min(all_scores)
        }
    
    return matching_analysis


def create_persona_based_training_data(messages_df: pd.DataFrame, recipients_df: pd.DataFrame,
                                     communication_styles: Dict[int, Dict[str, Any]], 
                                     your_recipient_id: int = 2) -> List[Dict[str, Any]]:
    """
    Create training data that includes persona information for better style adaptation.
    
    Args:
        messages_df: DataFrame of messages
        recipients_df: DataFrame of recipients
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID
    
    Returns:
        List of persona-based training examples
    """
    training_data = []
    
    # Create persona descriptions for each communication style
    persona_descriptions = {
        'rapid_burst_chatter': "someone who sends many quick, short messages",
        'verbose_burst_chatter': "someone who sends multiple detailed messages rapidly",
        'lengthy_texter': "someone who writes long, comprehensive messages",
        'concise_texter': "someone who keeps messages brief and efficient",
        'moderate_burst_chatter': "someone who occasionally sends message bursts",
        'balanced_communicator': "someone with a balanced messaging style",
        'unknown': "someone"
    }
    
    for thread_id in messages_df['thread_id'].unique():
        thread_messages = messages_df[
            messages_df['thread_id'] == thread_id
        ].sort_values('date_sent')
        
        if len(thread_messages) < 3:
            continue
        
        # Identify the other person
        other_participants = thread_messages[
            thread_messages['from_recipient_id'] != your_recipient_id
        ]['from_recipient_id'].unique()
        
        if len(other_participants) != 1:
            continue
        
        other_person_id = other_participants[0]
        other_person_style = communication_styles.get(other_person_id, {})
        style_type = other_person_style.get('style_type', 'unknown')
        persona_desc = persona_descriptions.get(style_type, persona_descriptions['unknown'])
        
        # Create training examples with persona context
        for i in range(len(thread_messages) - 1):
            current_msg = thread_messages.iloc[i]
            next_msg = thread_messages.iloc[i + 1]
            
            if next_msg['from_recipient_id'] == your_recipient_id:
                # Build context
                context_start = max(0, i - 3)
                context_messages = thread_messages.iloc[context_start:i+1]
                
                conversation = []
                for _, msg in context_messages.iterrows():
                    speaker = "You" if msg['from_recipient_id'] == your_recipient_id else "Them"
                    conversation.append(f"{speaker}: {msg['body']}")
                
                # Create persona-aware instruction
                instruction = f"You're having a conversation with {persona_desc}. Based on their communication style and the conversation context, provide an appropriate response."
                
                training_example = {
                    'instruction': instruction,
                    'input': "\n".join(conversation),
                    'output': next_msg['body'],
                    'metadata': {
                        'persona_type': style_type,
                        'thread_id': thread_id,
                        'other_person_name': other_person_style.get('name', 'Unknown')
                    }
                }
                
                training_data.append(training_example)
    
    return training_data