#!/usr/bin/env python3
"""
Topic Analysis Script

Analyzes Signal conversation data to identify and categorize discussion topics:
- Political discussion categorization and threading
- Personal interest mapping and evolution
- Humor pattern detection and meme tracking
- Current events and news discussion tracking
- Academic/intellectual discourse analysis
- Twitter/social media content integration

Usage:
    python scripts/topic_analysis.py [--min-messages 10] [--detailed]
"""

import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import argparse
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

def load_signal_data(data_dir: str = "data/raw/signal-flatfiles") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Signal CSV files."""
    base_path = Path(data_dir)
    messages_df = pd.read_csv(base_path / "signal.csv")
    recipients_df = pd.read_csv(base_path / "recipient.csv")
    return messages_df, recipients_df

def extract_urls_from_messages(messages_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract all URLs from messages with context."""
    
    url_pattern = r'https?://[^\s]+'
    urls_found = []
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        if pd.isna(body) or not isinstance(body, str):
            continue
        
        urls = re.findall(url_pattern, body)
        
        for url in urls:
            # Clean URL (remove trailing punctuation)
            url = re.sub(r'[,.;!?]+$', '', url)
            
            urls_found.append({
                'url': url,
                'message_id': row.get('_id'),
                'sender_id': row.get('from_recipient_id'),
                'thread_id': row.get('thread_id'),
                'timestamp': row.get('date_sent'),
                'full_message': body,
                'domain': urlparse(url).netloc.lower()
            })
    
    return urls_found

def categorize_political_discussions(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify and categorize political/theoretical discussions."""
    
    print("🏛️ Analyzing political discussions...")
    
    # Political theory keywords with categories
    political_categories = {
        'marxist_theory': [
            'dialectical', 'materialist', 'bourgeois', 'proletariat', 'class struggle',
            'surplus value', 'commodity', 'labor power', 'means of production',
            'base and superstructure', 'ideology', 'hegemony', 'praxis'
        ],
        'fascism_analysis': [
            'fascism', 'fascist', 'nazis', 'nazi', 'authoritarianism', 'totalitarian',
            'revanchist', 'supranationalist', 'militarism', 'ultranationalism'
        ],
        'current_politics': [
            'biden', 'trump', 'election', 'congress', 'senate', 'democrat', 'republican',
            'liberal', 'conservative', 'progressive', 'policy', 'legislation'
        ],
        'international_affairs': [
            'nato', 'ukraine', 'russia', 'china', 'palestine', 'israel', 'zionism',
            'imperialism', 'geopolitics', 'foreign policy', 'sanctions', 'war'
        ],
        'economic_theory': [
            'capitalism', 'socialism', 'communism', 'neoliberalism', 'austerity',
            'market', 'finance', 'banks', 'wall street', 'inequality', 'wealth'
        ],
        'social_movements': [
            'organizing', 'activism', 'protest', 'strike', 'union', 'solidarity',
            'mutual aid', 'direct action', 'revolution', 'reform'
        ],
        'intelligence_agencies': [
            'cia', 'fbi', 'nsa', 'intelligence', 'surveillance', 'spying',
            'national security', 'classified', 'whistleblower'
        ]
    }
    
    # Academic/theoretical language indicators
    academic_indicators = [
        'analysis', 'theory', 'framework', 'paradigm', 'methodology', 'thesis',
        'argument', 'critique', 'synthesis', 'dialectic', 'discourse', 'praxis'
    ]
    
    political_discussions = defaultdict(list)
    academic_discussions = []
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        if pd.isna(body) or not isinstance(body, str):
            continue
        
        body_lower = body.lower()
        
        # Check for political categories
        for category, keywords in political_categories.items():
            if any(keyword in body_lower for keyword in keywords):
                political_discussions[category].append({
                    'message_id': row.get('_id'),
                    'sender_id': row.get('from_recipient_id'),
                    'thread_id': row.get('thread_id'),
                    'timestamp': row.get('date_sent'),
                    'text': body,
                    'matched_keywords': [kw for kw in keywords if kw in body_lower]
                })
        
        # Check for academic language
        academic_count = sum(1 for indicator in academic_indicators if indicator in body_lower)
        if academic_count >= 2 or len(body) > 200:  # Either multiple indicators or long-form
            academic_discussions.append({
                'message_id': row.get('_id'),
                'sender_id': row.get('from_recipient_id'),
                'thread_id': row.get('thread_id'),
                'timestamp': row.get('date_sent'),
                'text': body,
                'academic_indicators': [ind for ind in academic_indicators if ind in body_lower],
                'word_count': len(body.split())
            })
    
    return {
        'political_categories': dict(political_discussions),
        'academic_discussions': academic_discussions,
        'category_stats': {cat: len(msgs) for cat, msgs in political_discussions.items()},
        'total_political_messages': sum(len(msgs) for msgs in political_discussions.values()),
        'total_academic_messages': len(academic_discussions)
    }

def analyze_personal_interests(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify personal interests and casual topics."""
    
    print("🎯 Analyzing personal interests...")
    
    interest_categories = {
        'food_cooking': [
            'food', 'cooking', 'recipe', 'eat', 'dinner', 'lunch', 'breakfast',
            'restaurant', 'kitchen', 'cook', 'meal', 'hungry', 'delicious',
            'taste', 'flavor', 'spice', 'arepas', 'coffee', 'drink'
        ],
        'technology': [
            'computer', 'software', 'programming', 'code', 'tech', 'app',
            'website', 'internet', 'digital', 'algorithm', 'ai', 'machine learning'
        ],
        'entertainment': [
            'movie', 'film', 'tv', 'show', 'music', 'song', 'band', 'artist',
            'game', 'gaming', 'book', 'read', 'novel', 'story'
        ],
        'personal_life': [
            'work', 'job', 'career', 'family', 'friend', 'relationship',
            'home', 'house', 'travel', 'vacation', 'health', 'exercise'
        ],
        'academic_subjects': [
            'philosophy', 'history', 'economics', 'sociology', 'psychology',
            'literature', 'science', 'research', 'study', 'education'
        ],
        'current_events': [
            'news', 'breaking', 'happened', 'today', 'yesterday', 'recent',
            'update', 'development', 'situation', 'event'
        ]
    }
    
    interest_discussions = defaultdict(list)
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        if pd.isna(body) or not isinstance(body, str):
            continue
        
        body_lower = body.lower()
        
        for category, keywords in interest_categories.items():
            matched_keywords = [kw for kw in keywords if kw in body_lower]
            if matched_keywords:
                interest_discussions[category].append({
                    'message_id': row.get('_id'),
                    'sender_id': row.get('from_recipient_id'),
                    'thread_id': row.get('thread_id'),
                    'timestamp': row.get('date_sent'),
                    'text': body,
                    'matched_keywords': matched_keywords
                })
    
    return {
        'interest_categories': dict(interest_discussions),
        'category_stats': {cat: len(msgs) for cat, msgs in interest_discussions.items()},
        'total_personal_messages': sum(len(msgs) for msgs in interest_discussions.values())
    }

def detect_humor_and_memes(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect humor patterns and meme usage."""
    
    print("😂 Detecting humor and meme patterns...")
    
    humor_indicators = {
        'internet_slang': [
            'lmao', 'lol', 'lmfao', 'rofl', 'omg', 'wtf', 'bruh', 'fr', 'ngl',
            'tbh', 'imo', 'imho', 'smh', 'fml', 'yolo', 'af', 'lowkey', 'highkey'
        ],
        'humor_expressions': [
            'haha', 'hehe', 'lololol', 'ahahaha', 'bahahaha', 'dead', 'dying',
            'crying', 'weak', 'can\'t even', 'i can\'t', 'im dead'
        ],
        'sarcasm_indicators': [
            'yeah right', 'sure thing', 'oh really', 'how shocking', 'what a surprise',
            'totally', 'absolutely', 'definitely', 'obviously', 'clearly'
        ],
        'meme_references': [
            'meme', 'viral', 'trending', 'based', 'cringe', 'sus', 'salty',
            'triggered', 'woke', 'karen', 'boomer', 'ok boomer', 'big mood'
        ]
    }
    
    humor_messages = defaultdict(list)
    
    # Messages with excessive punctuation (humor indicator)
    excessive_punct_pattern = r'[!]{2,}|[?]{2,}|[.]{3,}'
    
    # Capitalization patterns (excitement/emphasis)
    caps_pattern = r'[A-Z]{3,}'
    
    for _, row in messages_df.iterrows():
        body = row.get('body', '')
        if pd.isna(body) or not isinstance(body, str):
            continue
        
        body_lower = body.lower()
        
        # Check humor categories
        for category, indicators in humor_indicators.items():
            matched = [ind for ind in indicators if ind in body_lower]
            if matched:
                humor_messages[category].append({
                    'message_id': row.get('_id'),
                    'sender_id': row.get('from_recipient_id'),
                    'thread_id': row.get('thread_id'),
                    'timestamp': row.get('date_sent'),
                    'text': body,
                    'matched_indicators': matched
                })
        
        # Check for excessive punctuation
        if re.search(excessive_punct_pattern, body):
            humor_messages['excessive_punctuation'].append({
                'message_id': row.get('_id'),
                'sender_id': row.get('from_recipient_id'),
                'thread_id': row.get('thread_id'),
                'timestamp': row.get('date_sent'),
                'text': body,
                'pattern': 'excessive_punctuation'
            })
        
        # Check for caps (excitement)
        caps_words = re.findall(caps_pattern, body)
        if caps_words:
            humor_messages['caps_emphasis'].append({
                'message_id': row.get('_id'),
                'sender_id': row.get('from_recipient_id'),
                'thread_id': row.get('thread_id'),
                'timestamp': row.get('date_sent'),
                'text': body,
                'caps_words': caps_words
            })
    
    return {
        'humor_categories': dict(humor_messages),
        'category_stats': {cat: len(msgs) for cat, msgs in humor_messages.items()},
        'total_humor_messages': sum(len(msgs) for msgs in humor_messages.values())
    }

def analyze_social_media_content(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze social media links and content sharing patterns."""
    
    print("🐦 Analyzing social media content sharing...")
    
    # Extract URLs
    urls_data = extract_urls_from_messages(messages_df)
    
    # Categorize by platform
    platform_patterns = {
        'twitter': ['twitter.com', 'x.com', 't.co'],
        'youtube': ['youtube.com', 'youtu.be'],
        'reddit': ['reddit.com'],
        'tiktok': ['tiktok.com'],
        'instagram': ['instagram.com'],
        'facebook': ['facebook.com', 'fb.com'],
        'news_sites': ['nytimes.com', 'washingtonpost.com', 'cnn.com', 'bbc.com', 'reuters.com', 'npr.org'],
        'academic': ['arxiv.org', 'jstor.org', 'scholar.google.com', 'researchgate.net'],
        'political': ['jacobin.com', 'politico.com', 'thehill.com', 'salon.com', 'slate.com']
    }
    
    platform_sharing = defaultdict(list)
    
    for url_data in urls_data:
        domain = url_data['domain']
        
        # Find matching platform
        platform_found = False
        for platform, domains in platform_patterns.items():
            if any(d in domain for d in domains):
                platform_sharing[platform].append(url_data)
                platform_found = True
                break
        
        if not platform_found:
            platform_sharing['other'].append(url_data)
    
    # Analyze Twitter content specifically
    twitter_analysis = {}
    if 'twitter' in platform_sharing:
        twitter_urls = platform_sharing['twitter']
        
        # Count by sender
        twitter_by_sender = defaultdict(int)
        for url_data in twitter_urls:
            twitter_by_sender[url_data['sender_id']] += 1
        
        # Extract commentary patterns (text around Twitter URLs)
        twitter_commentary = []
        for url_data in twitter_urls:
            message = url_data['full_message']
            url = url_data['url']
            
            # Get text before and after URL
            url_index = message.find(url)
            if url_index > 0:
                before_text = message[:url_index].strip()
                after_text = message[url_index + len(url):].strip()
                
                if before_text or after_text:
                    twitter_commentary.append({
                        'url': url,
                        'before': before_text,
                        'after': after_text,
                        'sender_id': url_data['sender_id'],
                        'timestamp': url_data['timestamp']
                    })
        
        twitter_analysis = {
            'total_shared': len(twitter_urls),
            'by_sender': dict(twitter_by_sender),
            'with_commentary': len(twitter_commentary),
            'commentary_examples': twitter_commentary[:10]  # First 10 examples
        }
    
    return {
        'platform_sharing': {platform: len(urls) for platform, urls in platform_sharing.items()},
        'detailed_sharing': dict(platform_sharing),
        'twitter_analysis': twitter_analysis,
        'total_urls': len(urls_data),
        'url_sharing_rate': len(urls_data) / len(messages_df) if len(messages_df) > 0 else 0
    }

def identify_conversation_threads(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """Identify coherent conversation threads and topic transitions."""
    
    print("🧵 Identifying conversation threads...")
    
    # Group by thread and analyze topic continuity
    thread_analysis = {}
    
    for thread_id in messages_df['thread_id'].unique():
        thread_messages = messages_df[messages_df['thread_id'] == thread_id].sort_values('date_sent')
        
        if len(thread_messages) < 5:  # Skip short threads
            continue
        
        # Extract key topics from thread
        thread_text = ' '.join(thread_messages['body'].fillna('').astype(str))
        
        # Simple topic extraction using keyword frequency
        words = re.findall(r'\b\w{4,}\b', thread_text.lower())  # Words 4+ chars
        
        # Remove common words
        stopwords = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'only', 'think', 'also', 'like', 'some', 'when', 'than', 'about', 'after', 'first', 'well', 'year', 'work', 'such', 'make', 'even', 'most', 'take', 'them', 'these', 'good', 'much', 'over', 'want', 'come', 'here', 'right', 'still', 'back', 'through', 'where', 'being', 'before', 'between', 'both', 'under', 'again', 'same', 'those', 'while', 'should', 'never', 'around', 'another', 'though', 'might', 'really', 'little', 'because', 'against', 'without', 'during', 'something', 'nothing', 'everything', 'anything', 'someone', 'everyone', 'anyone', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves'}
        
        filtered_words = [word for word in words if word not in stopwords and len(word) > 3]
        word_freq = Counter(filtered_words)
        
        # Get most common topics
        top_topics = word_freq.most_common(10)
        
        # Calculate thread statistics
        unique_senders = thread_messages['from_recipient_id'].nunique()
        message_count = len(thread_messages)
        time_span = (thread_messages['date_sent'].max() - thread_messages['date_sent'].min()) / 1000 / 3600  # hours
        
        thread_analysis[thread_id] = {
            'message_count': message_count,
            'unique_senders': unique_senders,
            'time_span_hours': time_span,
            'top_topics': top_topics,
            'avg_message_length': thread_messages['body'].str.len().mean(),
            'url_count': len(extract_urls_from_messages(thread_messages))
        }
    
    return thread_analysis

def generate_topic_report(messages_df: pd.DataFrame, recipients_df: pd.DataFrame, detailed: bool = False) -> Dict[str, Any]:
    """Generate comprehensive topic analysis report."""
    
    print("📋 Generating comprehensive topic analysis report...")
    
    # Run all analyses
    political_analysis = categorize_political_discussions(messages_df)
    personal_analysis = analyze_personal_interests(messages_df)
    humor_analysis = detect_humor_and_memes(messages_df)
    social_media_analysis = analyze_social_media_content(messages_df)
    thread_analysis = identify_conversation_threads(messages_df)
    
    # Overall statistics
    total_messages = len(messages_df[messages_df['body'].notna()])
    
    # Calculate topic coverage
    political_coverage = political_analysis['total_political_messages'] / total_messages if total_messages > 0 else 0
    personal_coverage = personal_analysis['total_personal_messages'] / total_messages if total_messages > 0 else 0
    humor_coverage = humor_analysis['total_humor_messages'] / total_messages if total_messages > 0 else 0
    
    report = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_messages_analyzed': total_messages,
            'analysis_coverage': {
                'political_discussions': political_coverage,
                'personal_interests': personal_coverage,
                'humor_content': humor_coverage
            }
        },
        'political_analysis': political_analysis,
        'personal_interests': personal_analysis,
        'humor_patterns': humor_analysis,
        'social_media_sharing': social_media_analysis,
        'conversation_threads': thread_analysis if detailed else {k: v for k, v in list(thread_analysis.items())[:5]},
        'topic_distribution_summary': {
            'political_theory': political_analysis['category_stats'],
            'personal_interests': personal_analysis['category_stats'],
            'humor_types': humor_analysis['category_stats'],
            'social_platforms': social_media_analysis['platform_sharing']
        }
    }
    
    return report

def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze topics and themes in Signal conversations')
    parser.add_argument('--min-messages', type=int, default=10, help='Minimum messages per topic category')
    parser.add_argument('--detailed', action='store_true', help='Include detailed thread analysis')
    
    args = parser.parse_args()
    
    print("📚 Starting topic analysis...")
    
    try:
        # Load data
        messages_df, recipients_df = load_signal_data()
        print(f"✅ Loaded {len(messages_df)} messages, {len(recipients_df)} recipients")
        
        # Generate report
        report = generate_topic_report(messages_df, recipients_df, detailed=args.detailed)
        
        # Save report
        output_path = Path("data/processed/topic_analysis.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📝 Analysis complete! Report saved to {output_path}")
        
        # Print summary
        meta = report['metadata']
        political = report['political_analysis']
        personal = report['personal_interests']
        humor = report['humor_patterns']
        social = report['social_media_sharing']
        
        print("\n" + "="*50)
        print("TOPIC ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"📊 Total Messages: {meta['total_messages_analyzed']:,}")
        print(f"🏛️  Political Discussions: {political['total_political_messages']:,} ({meta['analysis_coverage']['political_discussions']:.1%})")
        print(f"🎯 Personal Interests: {personal['total_personal_messages']:,} ({meta['analysis_coverage']['personal_interests']:.1%})")
        print(f"😂 Humor Content: {humor['total_humor_messages']:,} ({meta['analysis_coverage']['humor_content']:.1%})")
        print(f"🔗 URLs Shared: {social['total_urls']:,}")
        
        print(f"\n🏛️  Political Categories:")
        for category, count in sorted(political['category_stats'].items(), key=lambda x: x[1], reverse=True):
            if count >= args.min_messages:
                print(f"   {category}: {count:,}")
        
        print(f"\n🎯 Personal Interest Categories:")
        for category, count in sorted(personal['category_stats'].items(), key=lambda x: x[1], reverse=True):
            if count >= args.min_messages:
                print(f"   {category}: {count:,}")
        
        print(f"\n😂 Humor Categories:")
        for category, count in sorted(humor['category_stats'].items(), key=lambda x: x[1], reverse=True):
            if count >= args.min_messages:
                print(f"   {category}: {count:,}")
        
        print(f"\n🐦 Social Media Platforms:")
        for platform, count in sorted(social['platform_sharing'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                print(f"   {platform}: {count:,}")
        
        if social['twitter_analysis']:
            twitter = social['twitter_analysis']
            print(f"\n🐦 Twitter Analysis:")
            print(f"   Total shared: {twitter['total_shared']:,}")
            print(f"   With commentary: {twitter['with_commentary']:,}")
        
        print("\n✨ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()