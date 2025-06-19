#!/usr/bin/env python3
"""
Personality Marker Extraction Script

Analyzes Signal conversation data to extract unique communication quirks and personality markers:
- Individual communication style fingerprints
- Language pattern evolution over time
- Emotional expression patterns
- Relationship dynamic analysis
- Personal linguistic signatures

Usage:
    python scripts/extract_personality_markers.py [--sender-id ID]
"""

import pandas as pd
import re
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

def load_signal_data(data_dir: str = "data/raw/signal-flatfiles") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Signal CSV files."""
    base_path = Path(data_dir)
    messages_df = pd.read_csv(base_path / "signal.csv")
    recipients_df = pd.read_csv(base_path / "recipient.csv")
    return messages_df, recipients_df

def extract_linguistic_patterns(messages: List[str]) -> Dict[str, Any]:
    """Extract linguistic patterns from a collection of messages."""
    
    # Punctuation patterns
    exclamation_count = sum(msg.count('!') for msg in messages)
    question_count = sum(msg.count('?') for msg in messages)
    ellipsis_count = sum(len(re.findall(r'\.{2,}', msg)) for msg in messages)
    caps_count = sum(len(re.findall(r'[A-Z]{2,}', msg)) for msg in messages)
    
    # Sentence starters
    starters = []
    for msg in messages:
        sentences = re.split(r'[.!?]+', msg)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:
                first_word = sentence.split()[0].lower() if sentence.split() else ''
                if first_word:
                    starters.append(first_word)
    
    starter_patterns = Counter(starters).most_common(10)
    
    # Filler words and expressions
    filler_words = ['like', 'um', 'uh', 'you know', 'i mean', 'basically', 'literally', 'actually', 'honestly']
    filler_counts = {word: sum(msg.lower().count(word) for msg in messages) for word in filler_words}
    
    # Internet slang patterns
    slang_patterns = {
        'lmao': r'\blmao\b',
        'lol': r'\blol\b', 
        'omg': r'\bomg\b',
        'wtf': r'\bwtf\b',
        'tbh': r'\btbh\b',
        'ngl': r'\bngl\b',
        'fr': r'\bfr\b',
        'bruh': r'\bbruh\b',
        'imo': r'\bimo\b',
        'imho': r'\bimho\b'
    }
    
    slang_usage = {}
    for slang, pattern in slang_patterns.items():
        count = sum(len(re.findall(pattern, msg.lower())) for msg in messages)
        slang_usage[slang] = count
    
    # Contraction preferences
    contractions = {
        "don't": ["do not", "don't"],
        "can't": ["cannot", "can't"],
        "won't": ["will not", "won't"],
        "it's": ["it is", "it's"],
        "you're": ["you are", "you're"],
        "we're": ["we are", "we're"]
    }
    
    contraction_usage = {}
    for contraction, variants in contractions.items():
        formal_count = sum(msg.lower().count(variants[0]) for msg in messages)
        informal_count = sum(msg.lower().count(variants[1]) for msg in messages)
        total = formal_count + informal_count
        if total > 0:
            contraction_usage[contraction] = informal_count / total
    
    return {
        'punctuation': {
            'exclamation_frequency': exclamation_count / len(messages) if messages else 0,
            'question_frequency': question_count / len(messages) if messages else 0,
            'ellipsis_frequency': ellipsis_count / len(messages) if messages else 0,
            'caps_frequency': caps_count / len(messages) if messages else 0
        },
        'sentence_starters': starter_patterns,
        'filler_words': filler_counts,
        'slang_usage': slang_usage,
        'contraction_preferences': contraction_usage
    }

def analyze_emotional_expression(messages: List[str], timestamps: List[int]) -> Dict[str, Any]:
    """Analyze emotional expression patterns."""
    
    # Emotion keywords
    emotion_patterns = {
        'excitement': [r'\b(amazing|awesome|incredible|fantastic|wow|yay|excited)\b', r'!{2,}'],
        'frustration': [r'\b(ugh|argh|fuck|shit|damn|annoying|frustrated|irritating)\b'],
        'happiness': [r'\b(happy|glad|joy|pleased|delighted|love|haha|lol)\b'],
        'sadness': [r'\b(sad|depressed|down|upset|crying|tear)\b'],
        'anger': [r'\b(angry|mad|pissed|furious|rage|hate)\b'],
        'confusion': [r'\b(confused|wtf|huh|what|unclear|lost)\b'],
        'agreement': [r'\b(yes|yeah|yep|exactly|absolutely|definitely|true|right)\b'],
        'disagreement': [r'\b(no|nope|wrong|disagree|bullshit|false)\b']
    }
    
    emotion_counts = defaultdict(int)
    emotion_timestamps = defaultdict(list)
    
    for msg, timestamp in zip(messages, timestamps):
        msg_lower = msg.lower()
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower):
                    emotion_counts[emotion] += 1
                    emotion_timestamps[emotion].append(timestamp)
                    break  # Only count once per message per emotion
    
    # Analyze emotional volatility (how quickly emotions change)
    all_emotions = []
    for msg, timestamp in zip(messages, timestamps):
        msg_lower = msg.lower()
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower):
                    all_emotions.append((timestamp, emotion))
                    break
    
    # Calculate emotional transitions
    transitions = []
    for i in range(1, len(all_emotions)):
        prev_time, prev_emotion = all_emotions[i-1]
        curr_time, curr_emotion = all_emotions[i]
        time_diff = (curr_time - prev_time) / 1000 / 60  # minutes
        
        if time_diff < 60 and prev_emotion != curr_emotion:  # Within 1 hour and different emotions
            transitions.append((prev_emotion, curr_emotion, time_diff))
    
    return {
        'emotion_frequencies': dict(emotion_counts),
        'emotion_timestamps': {k: v for k, v in emotion_timestamps.items()},
        'emotional_transitions': transitions,
        'emotional_volatility': len(transitions) / len(messages) if messages else 0
    }

def analyze_conversation_adaptation(messages_df: pd.DataFrame, sender_id: int) -> Dict[str, Any]:
    """Analyze how communication style adapts to different conversation partners."""
    
    sender_messages = messages_df[messages_df['from_recipient_id'] == sender_id].copy()
    
    if len(sender_messages) == 0:
        return {}
    
    # Group by thread to identify conversation partners
    adaptation_patterns = {}
    
    for thread_id in sender_messages['thread_id'].unique():
        thread_messages = messages_df[messages_df['thread_id'] == thread_id].copy()
        
        # Identify other participants in this thread
        other_senders = thread_messages[thread_messages['from_recipient_id'] != sender_id]['from_recipient_id'].unique()
        
        if len(other_senders) == 0:
            continue
        
        # Get sender's messages in this thread
        sender_thread_msgs = thread_messages[thread_messages['from_recipient_id'] == sender_id]
        messages_text = [msg for msg in sender_thread_msgs['body'].fillna('') if isinstance(msg, str) and len(msg.strip()) > 0]
        
        if len(messages_text) < 5:  # Need enough messages for analysis
            continue
        
        # Analyze style in this thread
        linguistic_patterns = extract_linguistic_patterns(messages_text)
        
        # Calculate message length stats
        lengths = [len(msg) for msg in messages_text]
        avg_length = np.mean(lengths) if lengths else 0
        
        # Calculate response times (time to respond to others)
        response_times = []
        sorted_thread = thread_messages.sort_values('date_sent')
        
        for i in range(1, len(sorted_thread)):
            if (sorted_thread.iloc[i]['from_recipient_id'] == sender_id and 
                sorted_thread.iloc[i-1]['from_recipient_id'] != sender_id):
                response_time = (sorted_thread.iloc[i]['date_sent'] - sorted_thread.iloc[i-1]['date_sent']) / 1000
                if response_time < 3600:  # Within 1 hour
                    response_times.append(response_time)
        
        avg_response_time = np.mean(response_times) if response_times else 0
        
        adaptation_patterns[thread_id] = {
            'other_participants': other_senders.tolist(),
            'message_count': len(messages_text),
            'avg_message_length': avg_length,
            'avg_response_time': avg_response_time,
            'linguistic_patterns': linguistic_patterns,
            'sample_messages': messages_text[:3]  # First 3 messages for style reference
        }
    
    return adaptation_patterns

def extract_time_based_patterns(messages_df: pd.DataFrame, sender_id: int) -> Dict[str, Any]:
    """Analyze how communication patterns change over time."""
    
    sender_messages = messages_df[messages_df['from_recipient_id'] == sender_id].copy()
    
    if len(sender_messages) == 0:
        return {}
    
    # Convert timestamps and sort
    sender_messages['datetime'] = pd.to_datetime(sender_messages['date_sent'], unit='ms')
    sender_messages = sender_messages.sort_values('datetime')
    
    # Group by month to see evolution
    sender_messages['month'] = sender_messages['datetime'].dt.to_period('M')
    monthly_patterns = {}
    
    for month in sender_messages['month'].unique():
        month_messages = sender_messages[sender_messages['month'] == month]
        messages_text = [msg for msg in month_messages['body'].fillna('') if isinstance(msg, str) and len(msg.strip()) > 0]
        
        if len(messages_text) < 5:  # Need enough messages
            continue
        
        linguistic_patterns = extract_linguistic_patterns(messages_text)
        
        monthly_patterns[str(month)] = {
            'message_count': len(messages_text),
            'avg_length': np.mean([len(msg) for msg in messages_text]),
            'linguistic_patterns': linguistic_patterns
        }
    
    # Analyze time-of-day patterns
    sender_messages['hour'] = sender_messages['datetime'].dt.hour
    hourly_activity = sender_messages['hour'].value_counts().sort_index().to_dict()
    
    # Most active time periods
    morning = sum(hourly_activity.get(h, 0) for h in range(6, 12))
    afternoon = sum(hourly_activity.get(h, 0) for h in range(12, 18))
    evening = sum(hourly_activity.get(h, 0) for h in range(18, 24))
    night = sum(hourly_activity.get(h, 0) for h in list(range(0, 6)))
    
    return {
        'monthly_evolution': monthly_patterns,
        'hourly_activity': hourly_activity,
        'time_preferences': {
            'morning': morning,
            'afternoon': afternoon,
            'evening': evening,
            'night': night
        },
        'most_active_hours': sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:5]
    }

def generate_personality_profile(messages_df: pd.DataFrame, recipients_df: pd.DataFrame, sender_id: int) -> Dict[str, Any]:
    """Generate comprehensive personality profile for a specific sender."""
    
    print(f"👤 Generating personality profile for sender {sender_id}...")
    
    # Get sender info
    sender_info = recipients_df[recipients_df['_id'] == sender_id].iloc[0] if len(recipients_df[recipients_df['_id'] == sender_id]) > 0 else {}
    sender_name = sender_info.get('profile_given_name', f'User_{sender_id}')
    
    # Get all messages from this sender
    sender_messages = messages_df[messages_df['from_recipient_id'] == sender_id].copy()
    
    if len(sender_messages) == 0:
        return {'error': f'No messages found for sender {sender_id}'}
    
    # Extract valid text messages
    messages_text = []
    timestamps = []
    
    for _, row in sender_messages.iterrows():
        body = row.get('body', '')
        if isinstance(body, str) and len(body.strip()) > 0:
            messages_text.append(body)
            timestamps.append(row.get('date_sent', 0))
    
    if len(messages_text) < 5:
        return {'error': f'Insufficient messages for analysis (found {len(messages_text)})'}
    
    # Run all analyses
    linguistic_patterns = extract_linguistic_patterns(messages_text)
    emotional_patterns = analyze_emotional_expression(messages_text, timestamps)
    adaptation_patterns = analyze_conversation_adaptation(messages_df, sender_id)
    time_patterns = extract_time_based_patterns(messages_df, sender_id)
    
    # Calculate basic stats
    total_messages = len(messages_text)
    total_chars = sum(len(msg) for msg in messages_text)
    avg_message_length = total_chars / total_messages
    
    # Message length distribution
    lengths = [len(msg) for msg in messages_text]
    length_stats = {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': min(lengths),
        'max': max(lengths)
    }
    
    # Unique threads participated in
    unique_threads = sender_messages['thread_id'].nunique()
    
    return {
        'sender_info': {
            'id': sender_id,
            'name': sender_name,
            'total_messages': total_messages,
            'total_characters': total_chars,
            'avg_message_length': avg_message_length,
            'unique_conversations': unique_threads
        },
        'linguistic_patterns': linguistic_patterns,
        'emotional_patterns': emotional_patterns,
        'adaptation_patterns': adaptation_patterns,
        'time_patterns': time_patterns,
        'message_length_stats': length_stats,
        'analysis_timestamp': datetime.now().isoformat()
    }

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Extract personality markers from Signal conversations')
    parser.add_argument('--sender-id', type=int, help='Specific sender ID to analyze (default: analyze all active senders)')
    parser.add_argument('--min-messages', type=int, default=50, help='Minimum messages required for analysis')
    
    args = parser.parse_args()
    
    print("🔍 Starting personality marker extraction...")
    
    try:
        # Load data
        messages_df, recipients_df = load_signal_data()
        print(f"✅ Loaded {len(messages_df)} messages, {len(recipients_df)} recipients")
        
        # Determine which senders to analyze
        if args.sender_id:
            sender_ids = [args.sender_id]
        else:
            # Find senders with enough messages
            message_counts = messages_df['from_recipient_id'].value_counts()
            sender_ids = message_counts[message_counts >= args.min_messages].index.tolist()
            print(f"📊 Found {len(sender_ids)} senders with {args.min_messages}+ messages")
        
        # Generate profiles
        profiles = {}
        for sender_id in sender_ids:
            try:
                profile = generate_personality_profile(messages_df, recipients_df, sender_id)
                if 'error' not in profile:
                    profiles[sender_id] = profile
                    sender_name = profile['sender_info']['name']
                    msg_count = profile['sender_info']['total_messages']
                    print(f"✅ Generated profile for {sender_name} ({msg_count} messages)")
                else:
                    print(f"⚠️  Skipped sender {sender_id}: {profile['error']}")
            except Exception as e:
                print(f"❌ Error analyzing sender {sender_id}: {str(e)}")
        
        # Save results
        output_path = Path("data/processed/personality_profiles.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(profiles, f, indent=2, default=str)
        
        print(f"📝 Analysis complete! {len(profiles)} profiles saved to {output_path}")
        
        # Print summary
        if profiles:
            print("\n" + "="*50)
            print("PERSONALITY PROFILE SUMMARY")
            print("="*50)
            
            for sender_id, profile in profiles.items():
                info = profile['sender_info']
                ling = profile['linguistic_patterns']
                emot = profile['emotional_patterns']
                
                print(f"\n👤 {info['name']} (ID: {sender_id})")
                print(f"   Messages: {info['total_messages']:,}")
                print(f"   Avg Length: {info['avg_message_length']:.1f} chars")
                print(f"   Conversations: {info['unique_conversations']}")
                
                # Top slang used
                top_slang = sorted(ling['slang_usage'].items(), key=lambda x: x[1], reverse=True)[:3]
                if any(count > 0 for _, count in top_slang):
                    slang_list = [slang for slang, count in top_slang if count > 0]
                    print(f"   Top Slang: {', '.join(slang_list)}")
                
                # Dominant emotions
                top_emotions = sorted(emot['emotion_frequencies'].items(), key=lambda x: x[1], reverse=True)[:3]
                if top_emotions:
                    emotion_list = [emotion for emotion, count in top_emotions if count > 0]
                    print(f"   Emotions: {', '.join(emotion_list)}")
        
        print("\n✨ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()