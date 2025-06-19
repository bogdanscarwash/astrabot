"""
Q&A extraction functions for Astrabot.

This module provides enhanced Q&A extraction that handles Twitter content and 
better question patterns. Note: This approach has been deprecated in favor of
the conversational training system, but is kept for reference.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional

from src.core.conversation_processor import process_message_with_twitter_content


def extract_qa_pairs_enhanced(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhanced Q&A extraction that handles Twitter content and better question patterns.
    
    Note: This method has been deprecated in favor of the conversational training
    approach which better preserves natural dialogue flow.
    
    Args:
        conversations: List of conversation dictionaries with 'messages' field
    
    Returns:
        List of Q&A pairs with enhanced content and metadata
    """
    qa_pairs = []
    
    # Enhanced question patterns
    question_patterns = [
        r'\?',  # Explicit question mark
        r'^(what|how|why|when|where|who|which|whose)\s',  # Question words at start
        r'\b(can you|could you|would you|will you|should i|do you|does|is it|are you)\b',  # Common question phrases
        r'\b(please explain|please help|please tell|what about|how about)\b',  # Request patterns
        r'\b(any idea|any thoughts|any suggestions|anyone know)\b',  # Seeking input
    ]
    
    for conv in conversations:
        messages = conv['messages']
        
        for i in range(len(messages) - 1):
            current = messages[i]
            next_msg = messages[i + 1]
            
            # Skip if same person (not a Q&A pair)
            if current['role'] == next_msg['role']:
                continue
            
            current_content = current['content']
            
            # Check if it's just a URL without question text
            is_just_url = bool(re.match(r'^https?://\S+$', current_content.strip()))
            
            # Check for question patterns
            is_question = False
            if not is_just_url:
                for pattern in question_patterns:
                    if re.search(pattern, current_content, re.IGNORECASE):
                        is_question = True
                        break
            
            if is_question:
                # Enhance with Twitter content if URLs present
                enhanced_question = process_message_with_twitter_content(
                    current_content, 
                    use_images=False  # For now, just text
                )
                enhanced_response = process_message_with_twitter_content(
                    next_msg['content'],
                    use_images=False
                )
                
                # Quality checks
                if len(next_msg['content']) < 10:  # Too short
                    continue
                if next_msg['content'].lower() in ['ok', 'okay', 'yeah', 'yes', 'no', 'lol', 'haha']:
                    continue
                if '?' in next_msg['content'] and len(next_msg['content']) < 50:  # Just another question
                    continue
                
                qa_pairs.append({
                    "instruction": enhanced_question,
                    "response": enhanced_response,
                    "context": conv.get('conversation_id', ''),
                    "original_question": current_content,
                    "has_twitter_content": enhanced_question != current_content
                })
    
    return qa_pairs


def extract_qa_pairs_with_quality_filters(
    messages_df: pd.DataFrame,
    min_response_length: int = 20,
    max_response_length: int = 1000,
    your_recipient_id: int = 2
) -> List[Dict[str, Any]]:
    """
    Extract Q&A pairs with quality filters and context preservation.
    
    Args:
        messages_df: DataFrame of messages
        min_response_length: Minimum length for responses
        max_response_length: Maximum length for responses
        your_recipient_id: Your recipient ID
    
    Returns:
        List of high-quality Q&A pairs
    """
    qa_pairs = []
    
    # Enhanced question patterns
    question_patterns = [
        r'\?$',  # Ends with question mark
        r'^(what|how|why|when|where|who|which|whose)\s',  # Question words
        r'\b(can you|could you|would you|will you|should i|do you|does|is it|are you)\b',
        r'\b(please explain|please help|please tell|what about|how about)\b',
        r'\b(any idea|any thoughts|any suggestions|anyone know)\b',
    ]
    
    # Group by thread
    for thread_id in messages_df['thread_id'].unique():
        thread_messages = messages_df[
            messages_df['thread_id'] == thread_id
        ].sort_values('date_sent')
        
        for i in range(len(thread_messages) - 1):
            current = thread_messages.iloc[i]
            next_msg = thread_messages.iloc[i + 1]
            
            # Check if it's a Q&A pattern (different speakers)
            if current['from_recipient_id'] == next_msg['from_recipient_id']:
                continue
            
            # Check if current message is a question
            is_question = False
            current_text = current['body']
            
            if pd.notna(current_text) and len(current_text.strip()) > 5:
                for pattern in question_patterns:
                    if re.search(pattern, current_text, re.IGNORECASE):
                        is_question = True
                        break
            
            if is_question and pd.notna(next_msg['body']):
                response_text = next_msg['body']
                response_length = len(response_text)
                
                # Apply quality filters
                if (min_response_length <= response_length <= max_response_length and
                    response_text.lower() not in ['ok', 'okay', 'yeah', 'yes', 'no', 'lol', 'haha'] and
                    not (response_text.endswith('?') and response_length < 50)):
                    
                    # Determine who asked and who answered
                    asker = 'You' if current['from_recipient_id'] == your_recipient_id else 'Other'
                    answerer = 'You' if next_msg['from_recipient_id'] == your_recipient_id else 'Other'
                    
                    qa_pairs.append({
                        'question': current_text,
                        'answer': response_text,
                        'thread_id': thread_id,
                        'question_timestamp': current['date_sent'],
                        'answer_timestamp': next_msg['date_sent'],
                        'response_delay': (next_msg['date_sent'] - current['date_sent']) / 1000,  # seconds
                        'asker': asker,
                        'answerer': answerer,
                        'question_length': len(current_text),
                        'answer_length': response_length
                    })
    
    return qa_pairs


def analyze_qa_patterns(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in Q&A pairs to understand conversation dynamics.
    
    Args:
        qa_pairs: List of Q&A pairs from extract_qa_pairs_with_quality_filters
    
    Returns:
        Dictionary with analysis results
    """
    if not qa_pairs:
        return {
            'total_pairs': 0,
            'your_questions': 0,
            'your_answers': 0,
            'avg_response_delay': 0,
            'avg_answer_length': 0
        }
    
    your_questions = sum(1 for qa in qa_pairs if qa['asker'] == 'You')
    your_answers = sum(1 for qa in qa_pairs if qa['answerer'] == 'You')
    
    response_delays = [qa['response_delay'] for qa in qa_pairs]
    answer_lengths = [qa['answer_length'] for qa in qa_pairs]
    
    return {
        'total_pairs': len(qa_pairs),
        'your_questions': your_questions,
        'your_answers': your_answers,
        'question_ratio': your_questions / len(qa_pairs) if qa_pairs else 0,
        'answer_ratio': your_answers / len(qa_pairs) if qa_pairs else 0,
        'avg_response_delay': sum(response_delays) / len(response_delays),
        'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
        'response_delay_range': (min(response_delays), max(response_delays)),
        'answer_length_range': (min(answer_lengths), max(answer_lengths))
    }