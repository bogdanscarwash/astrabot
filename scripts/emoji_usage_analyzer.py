#!/usr/bin/env python3
"""
Emoji Usage Analysis Script

Comprehensive analysis of emoji usage patterns in Signal conversations:
- Emoji frequency and context analysis
- Emotional correlation mapping
- Personal emoji signature extraction
- Sender-specific emoji preferences
- Temporal emoji usage patterns

Usage:
    python scripts/emoji_usage_analyzer.py [--detailed]
"""

import pandas as pd
import emoji
import re
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_signal_data(data_dir: str = "data/raw/signal-flatfiles") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Signal CSV files."""
    base_path = Path(data_dir)
    messages_df = pd.read_csv(base_path / "signal.csv")
    recipients_df = pd.read_csv(base_path / "recipient.csv")
    return messages_df, recipients_df

def extract_emojis_with_context(text: str) -> List[Dict[str, Any]]:
    """Extract emojis with their position and surrounding context."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    
    emojis_found = []
    
    for i, char in enumerate(text):
        if char in emoji.EMOJI_DATA:
            # Get surrounding context (10 chars before and after)
            start = max(0, i - 10)
            end = min(len(text), i + 11)
            context = text[start:end]
            
            emojis_found.append({
                'emoji': char,
                'position': i,
                'context': context,
                'at_start': i < 3,
                'at_end': i > len(text) - 4,
                'standalone': len(text.strip()) == 1 and text.strip() == char
            })
    
    return emojis_found

def categorize_emoji_by_emotion(emoji_char: str) -> str:
    """Categorize emoji by emotional content."""
    
    # Define emotion categories
    emotion_categories = {
        'joy_laughter': ['😂', '🤣', '😄', '😃', '😁', '😊', '😀', '🙂', '😋', '😌'],
        'love_affection': ['❤️', '💖', '💕', '💗', '💓', '💘', '💝', '💜', '🧡', '💛', '💚', '💙', '🤍', '🖤', '🤎', '💯', '😍', '🥰', '😘', '😗', '😙', '😚'],
        'sadness_crying': ['😢', '😭', '😞', '😔', '😟', '😕', '🙁', '☹️', '😩', '😫'],
        'anger_frustration': ['😠', '😡', '🤬', '😤', '💢', '😾', '😖', '😣'],
        'surprise_shock': ['😮', '😯', '😲', '🤯', '😱', '🙀', '😳'],
        'thinking_contemplation': ['🤔', '🧐', '🤨', '🙄', '😐', '😑', '🤐'],
        'playful_teasing': ['😏', '😜', '😝', '😛', '🤪', '🤭', '😈', '👿', '🤡'],
        'support_encouragement': ['👍', '👌', '✌️', '🤝', '👏', '🙌', '💪', '🎉', '🎊', '✨'],
        'confusion_uncertainty': ['😵', '🤷', '🤦', '😅', '😬'],
        'cool_casual': ['😎', '🤓', '🥶', '🥵', '🤠']
    }
    
    # Find category for emoji
    for category, emojis in emotion_categories.items():
        if emoji_char in emojis:
            return category
    
    return 'other'

def analyze_emoji_context_patterns(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how emojis are used in different contexts."""
    
    print("🔍 Analyzing emoji context patterns...")
    
    emoji_contexts = defaultdict(lambda: {
        'total_count': 0,
        'standalone': 0,
        'at_start': 0,
        'at_end': 0,
        'mid_message': 0,
        'contexts': [],
        'emotion_category': 'other'
    })
    
    # Process all messages
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        if pd.isna(body) or not isinstance(body, str):
            continue
        
        emojis_in_msg = extract_emojis_with_context(body)
        
        for emoji_data in emojis_in_msg:
            emoji_char = emoji_data['emoji']
            
            # Update counts
            emoji_contexts[emoji_char]['total_count'] += 1
            
            if emoji_data['standalone']:
                emoji_contexts[emoji_char]['standalone'] += 1
            elif emoji_data['at_start']:
                emoji_contexts[emoji_char]['at_start'] += 1
            elif emoji_data['at_end']:
                emoji_contexts[emoji_char]['at_end'] += 1
            else:
                emoji_contexts[emoji_char]['mid_message'] += 1
            
            # Store context examples (limit to 5 per emoji)
            if len(emoji_contexts[emoji_char]['contexts']) < 5:
                emoji_contexts[emoji_char]['contexts'].append({
                    'context': emoji_data['context'],
                    'message': body,
                    'sender': row.get('from_recipient_id'),
                    'timestamp': row.get('date_sent')
                })
            
            # Set emotion category
            emoji_contexts[emoji_char]['emotion_category'] = categorize_emoji_by_emotion(emoji_char)
    
    return dict(emoji_contexts)

def analyze_emoji_by_sender(messages_df: pd.DataFrame, recipients_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze emoji usage patterns by sender."""
    
    print("👥 Analyzing emoji usage by sender...")
    
    # Get sender info
    sender_lookup = recipients_df.set_index('_id')['profile_given_name'].fillna('Unknown').to_dict()
    
    sender_emoji_patterns = defaultdict(lambda: {
        'total_emojis': 0,
        'unique_emojis': set(),
        'emoji_counts': Counter(),
        'messages_with_emojis': 0,
        'total_messages': 0,
        'emoji_density': 0.0,
        'signature_emojis': [],
        'emotional_profile': defaultdict(int)
    })
    
    # Process messages by sender
    for sender_id in messages_df['from_recipient_id'].unique():
        sender_messages = messages_df[messages_df['from_recipient_id'] == sender_id]
        
        sender_data = sender_emoji_patterns[sender_id]
        sender_data['name'] = sender_lookup.get(sender_id, f'User_{sender_id}')
        sender_data['total_messages'] = len(sender_messages)
        
        messages_with_emojis = 0
        
        for _, row in sender_messages.iterrows():
            body = row.get('body', '')
            if pd.isna(body) or not isinstance(body, str):
                continue
            
            # Extract emojis
            emojis_in_msg = [char for char in body if char in emoji.EMOJI_DATA]
            
            if emojis_in_msg:
                messages_with_emojis += 1
                sender_data['total_emojis'] += len(emojis_in_msg)
                sender_data['unique_emojis'].update(emojis_in_msg)
                sender_data['emoji_counts'].update(emojis_in_msg)
                
                # Update emotional profile
                for emoji_char in set(emojis_in_msg):  # Unique emojis per message
                    emotion_cat = categorize_emoji_by_emotion(emoji_char)
                    sender_data['emotional_profile'][emotion_cat] += 1
        
        sender_data['messages_with_emojis'] = messages_with_emojis
        
        # Calculate emoji density
        if sender_data['total_messages'] > 0:
            sender_data['emoji_density'] = messages_with_emojis / sender_data['total_messages']
        
        # Find signature emojis (emojis used significantly more than average)
        if sender_data['total_emojis'] > 10:  # Only if they use emojis regularly
            most_used = sender_data['emoji_counts'].most_common(5)
            sender_data['signature_emojis'] = [emoji_char for emoji_char, count in most_used if count >= 3]
        
        # Convert sets to lists for JSON serialization
        sender_data['unique_emojis'] = list(sender_data['unique_emojis'])
        sender_data['emoji_counts'] = dict(sender_data['emoji_counts'])
        sender_data['emotional_profile'] = dict(sender_data['emotional_profile'])
    
    return dict(sender_emoji_patterns)

def analyze_temporal_emoji_patterns(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze how emoji usage changes over time."""
    
    print("📅 Analyzing temporal emoji patterns...")
    
    # Convert timestamps
    messages_df['datetime'] = pd.to_datetime(messages_df['date_sent'], unit='ms')
    messages_df['month'] = messages_df['datetime'].dt.to_period('M')
    messages_df['hour'] = messages_df['datetime'].dt.hour
    messages_df['day_of_week'] = messages_df['datetime'].dt.day_name()
    
    # Monthly emoji evolution
    monthly_emoji_usage = defaultdict(Counter)
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        month = row.get('month')
        
        if pd.isna(body) or not isinstance(body, str) or pd.isna(month):
            continue
        
        emojis = [char for char in body if char in emoji.EMOJI_DATA]
        monthly_emoji_usage[str(month)].update(emojis)
    
    # Hourly emoji patterns
    hourly_emoji_usage = defaultdict(Counter)
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        hour = row.get('hour')
        
        if pd.isna(body) or not isinstance(body, str) or pd.isna(hour):
            continue
        
        emojis = [char for char in body if char in emoji.EMOJI_DATA]
        hourly_emoji_usage[hour].update(emojis)
    
    # Day of week patterns
    daily_emoji_usage = defaultdict(Counter)
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        day = row.get('day_of_week')
        
        if pd.isna(body) or not isinstance(body, str) or pd.isna(day):
            continue
        
        emojis = [char for char in body if char in emoji.EMOJI_DATA]
        daily_emoji_usage[day].update(emojis)
    
    return {
        'monthly_evolution': {month: dict(counter) for month, counter in monthly_emoji_usage.items()},
        'hourly_patterns': {hour: dict(counter) for hour, counter in hourly_emoji_usage.items()},
        'daily_patterns': {day: dict(counter) for day, counter in daily_emoji_usage.items()}
    }

def calculate_emoji_sentiment_scores(emoji_contexts: Dict[str, Any]) -> Dict[str, float]:
    """Calculate sentiment scores for emojis based on usage context."""
    
    sentiment_scores = {}
    
    # Simple sentiment keywords for context analysis
    positive_words = ['love', 'good', 'great', 'awesome', 'amazing', 'happy', 'yes', 'haha', 'lol', 'nice', 'cool']
    negative_words = ['bad', 'hate', 'awful', 'terrible', 'sad', 'no', 'ugh', 'damn', 'shit', 'fuck', 'annoying']
    
    for emoji_char, data in emoji_contexts.items():
        if data['total_count'] == 0:
            continue
        
        positive_contexts = 0
        negative_contexts = 0
        
        for context_data in data['contexts']:
            message = context_data['message'].lower()
            
            # Count positive/negative words in the message
            pos_count = sum(1 for word in positive_words if word in message)
            neg_count = sum(1 for word in negative_words if word in message)
            
            if pos_count > neg_count:
                positive_contexts += 1
            elif neg_count > pos_count:
                negative_contexts += 1
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_contexts = positive_contexts + negative_contexts
        if total_sentiment_contexts > 0:
            score = (positive_contexts - negative_contexts) / total_sentiment_contexts
        else:
            score = 0.0
        
        sentiment_scores[emoji_char] = score
    
    return sentiment_scores

def generate_emoji_report(messages_df: pd.DataFrame, recipients_df: pd.DataFrame, detailed: bool = False) -> Dict[str, Any]:
    """Generate comprehensive emoji usage report."""
    
    print("📊 Generating comprehensive emoji usage report...")
    
    # Run all analyses
    context_patterns = analyze_emoji_context_patterns(messages_df)
    sender_patterns = analyze_emoji_by_sender(messages_df, recipients_df)
    temporal_patterns = analyze_temporal_emoji_patterns(messages_df)
    sentiment_scores = calculate_emoji_sentiment_scores(context_patterns)
    
    # Overall statistics
    total_messages = len(messages_df[messages_df['body'].notna()])
    messages_with_emojis = sum(1 for _, row in messages_df.iterrows() 
                              if isinstance(row.get('body'), str) and 
                              any(char in emoji.EMOJI_DATA for char in row.get('body', '')))
    
    total_emojis = sum(data['total_count'] for data in context_patterns.values())
    unique_emojis = len(context_patterns)
    
    # Most common emojis
    emoji_frequency = [(emoji_char, data['total_count']) for emoji_char, data in context_patterns.items()]
    emoji_frequency.sort(key=lambda x: x[1], reverse=True)
    
    # Emotional distribution
    emotion_distribution = defaultdict(int)
    for emoji_char, data in context_patterns.items():
        emotion_cat = data['emotion_category']
        emotion_distribution[emotion_cat] += data['total_count']
    
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_messages': total_messages,
            'messages_with_emojis': messages_with_emojis,
            'emoji_usage_rate': messages_with_emojis / total_messages if total_messages > 0 else 0,
            'total_emojis': total_emojis,
            'unique_emojis': unique_emojis,
            'emojis_per_message': total_emojis / messages_with_emojis if messages_with_emojis > 0 else 0
        },
        'frequency_analysis': {
            'most_common': emoji_frequency[:20],
            'usage_distribution': {
                'high_frequency': len([e for e in emoji_frequency if e[1] >= 10]),
                'medium_frequency': len([e for e in emoji_frequency if 3 <= e[1] < 10]),
                'low_frequency': len([e for e in emoji_frequency if e[1] < 3])
            }
        },
        'emotional_analysis': {
            'emotion_distribution': dict(emotion_distribution),
            'sentiment_scores': sentiment_scores,
            'most_positive': sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)[:10],
            'most_negative': sorted(sentiment_scores.items(), key=lambda x: x[1])[:10]
        },
        'context_patterns': context_patterns if detailed else {k: v for k, v in list(context_patterns.items())[:10]},
        'sender_analysis': sender_patterns,
        'temporal_patterns': temporal_patterns
    }
    
    return report

def create_emoji_visualizations(report: Dict[str, Any], output_dir: str = "data/processed/emoji_plots"):
    """Create visualizations for emoji usage patterns."""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Top emojis bar chart
        top_emojis = report['frequency_analysis']['most_common'][:15]
        if top_emojis:
            fig, ax = plt.subplots(figsize=(12, 6))
            emojis, counts = zip(*top_emojis)
            
            bars = ax.bar(range(len(emojis)), counts)
            ax.set_xlabel('Emojis')
            ax.set_ylabel('Usage Count')
            ax.set_title('Most Frequently Used Emojis')
            ax.set_xticks(range(len(emojis)))
            ax.set_xticklabels(emojis, fontsize=16)
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / 'top_emojis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Emotion distribution pie chart
        emotion_dist = report['emotional_analysis']['emotion_distribution']
        if emotion_dist:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            emotions = list(emotion_dist.keys())
            counts = list(emotion_dist.values())
            
            # Only show emotions with significant usage
            threshold = max(counts) * 0.02  # At least 2% of max
            filtered_data = [(emotion, count) for emotion, count in zip(emotions, counts) if count > threshold]
            
            if filtered_data:
                emotions, counts = zip(*filtered_data)
                
                wedges, texts, autotexts = ax.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=90)
                ax.set_title('Emoji Usage by Emotional Category')
                
                plt.tight_layout()
                plt.savefig(output_path / 'emotion_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📈 Visualizations saved to {output_path}")
        
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available, skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error creating visualizations: {str(e)}")

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze emoji usage patterns in Signal conversations')
    parser.add_argument('--detailed', action='store_true', help='Include detailed context analysis for all emojis')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations (requires matplotlib)')
    
    args = parser.parse_args()
    
    print("😊 Starting emoji usage analysis...")
    
    try:
        # Load data
        messages_df, recipients_df = load_signal_data()
        print(f"✅ Loaded {len(messages_df)} messages, {len(recipients_df)} recipients")
        
        # Generate report
        report = generate_emoji_report(messages_df, recipients_df, detailed=args.detailed)
        
        # Save report
        output_path = Path("data/processed/emoji_usage_analysis.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"📝 Analysis complete! Report saved to {output_path}")
        
        # Create visualizations if requested
        if args.visualize:
            create_emoji_visualizations(report)
        
        # Print summary
        meta = report['metadata']
        freq = report['frequency_analysis']
        emot = report['emotional_analysis']
        
        print("\n" + "="*50)
        print("EMOJI USAGE SUMMARY")
        print("="*50)
        
        print(f"📊 Total Messages: {meta['total_messages']:,}")
        print(f"😊 Messages with Emojis: {meta['messages_with_emojis']:,} ({meta['emoji_usage_rate']:.1%})")
        print(f"🎭 Total Emojis Used: {meta['total_emojis']:,}")
        print(f"🌟 Unique Emojis: {meta['unique_emojis']}")
        print(f"📈 Emojis per Message: {meta['emojis_per_message']:.2f}")
        
        print(f"\n🎯 Top 10 Emojis:")
        for emoji_char, count in freq['most_common'][:10]:
            print(f"   {emoji_char} : {count:,}")
        
        print(f"\n💭 Emotional Distribution:")
        for emotion, count in sorted(emot['emotion_distribution'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {emotion}: {count:,}")
        
        if args.detailed:
            print(f"\n📋 Detailed context analysis included for all {meta['unique_emojis']} unique emojis")
        
        print("\n✨ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()