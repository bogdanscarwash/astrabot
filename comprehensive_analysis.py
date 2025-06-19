#!/usr/bin/env python3
import csv
import re
from collections import Counter, defaultdict
import json

def analyze_comprehensive_patterns():
    messages = []
    emoji_counter = Counter()
    word_counter = Counter()
    message_lengths = []
    topics = Counter()
    emotional_words = Counter()
    conversation_starters = []
    
    # Define patterns
    emoji_regex = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]|[\U00002600-\U000027BF]|[\U0001F900-\U0001F9FF]|[\U00002700-\U000027BF]')
    
    # Topic keywords
    topic_keywords = {
        'politics': ['fascist', 'fascism', 'democrat', 'republican', 'election', 'vote', 'government', 'biden', 'trump', 'political', 'marxism', 'communist', 'capitalism', 'dsa', 'revolutionary', 'theory'],
        'relationships': ['love', 'relationship', 'dating', 'partner', 'romantic', 'boyfriend', 'girlfriend', 'crush', 'kiss', 'married'],
        'gender_trans': ['trans', 'transgender', 'gender', 'transition', 'hormones', 'dysphoria', 'pronouns', 'masculine', 'feminine', 'nonbinary'],
        'technology': ['phone', 'computer', 'app', 'software', 'code', 'programming', 'internet', 'website', 'tech', 'digital'],
        'gaming': ['game', 'gaming', 'magic', 'mtg', 'cards', 'deck', 'tournament', 'player'],
        'food': ['food', 'eat', 'cooking', 'recipe', 'restaurant', 'dinner', 'lunch', 'breakfast', 'hungry'],
        'work_money': ['work', 'job', 'money', 'expensive', 'cost', 'salary', 'career', 'office', 'boss']
    }
    
    # Emotional indicators
    emotional_patterns = {
        'excitement': ['omg', 'wow', 'amazing', 'awesome', 'great', 'love', 'excited', 'yay', 'hell yes', 'fuck yeah'],
        'frustration': ['fuck', 'shit', 'damn', 'annoying', 'frustrated', 'hate', 'stupid', 'ridiculous', 'ugh'],
        'affection': ['love', 'sweet', 'beautiful', 'gorgeous', 'cute', 'adorable', 'honey', 'babe', 'darling'],
        'humor': ['lol', 'lmao', 'haha', 'funny', 'hilarious', 'joke', 'lollllllll'],
        'uncertainty': ['maybe', 'probably', 'perhaps', 'might', 'unsure', 'not sure', 'idk', 'dunno']
    }
    
    # Conversation patterns
    conversation_patterns = []
    previous_sender = None
    burst_count = 0
    
    with open('data/raw/signal-flatfiles/signal.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            body = row.get('body', '').strip()
            sender = row.get('from_recipient_id', '')
            
            if body and body != '' and not body.startswith('http') and len(body) > 1:
                messages.append({
                    'body': body,
                    'sender': sender,
                    'length': len(body)
                })
                message_lengths.append(len(body))
                
                # Burst pattern analysis
                if sender == previous_sender:
                    burst_count += 1
                else:
                    if burst_count > 1:
                        conversation_patterns.append(f"Burst of {burst_count} messages")
                    burst_count = 1
                    previous_sender = sender
                
                # Find emojis
                emojis = emoji_regex.findall(body)
                emoji_counter.update(emojis)
                
                # Analyze topics
                body_lower = body.lower()
                for topic, keywords in topic_keywords.items():
                    if any(keyword in body_lower for keyword in keywords):
                        topics[topic] += 1
                
                # Analyze emotions
                for emotion, patterns in emotional_patterns.items():
                    if any(pattern in body_lower for pattern in patterns):
                        emotional_words[emotion] += 1
                
                # Extract words
                words = re.findall(r'\b\w+\b', body_lower)
                word_counter.update(words)
                
                # Conversation starters (short messages that might start convos)
                if len(body) < 30 and not any(x in body_lower for x in ['ok', 'yeah', 'no', 'lol', 'haha']):
                    conversation_starters.append(body)
                
                count += 1
                if count >= 2000:  # Analyze more messages
                    break
    
    # Analysis results
    print("=== COMPREHENSIVE SIGNAL CONVERSATION ANALYSIS ===\n")
    
    print(f"📊 BASIC STATS:")
    print(f"Total messages analyzed: {len(messages)}")
    print(f"Average message length: {sum(message_lengths)/len(message_lengths):.1f} characters")
    print(f"Message length range: {min(message_lengths)} - {max(message_lengths)}")
    print()
    
    print("😀 TOP 15 EMOJIS:")
    for emoji_char, count in emoji_counter.most_common(15):
        print(f"  {emoji_char} : {count} times")
    print()
    
    print("🗣️ TOP CONVERSATION TOPICS:")
    for topic, count in topics.most_common():
        print(f"  {topic.replace('_', ' ').title()}: {count} messages")
    print()
    
    print("💭 EMOTIONAL TONE ANALYSIS:")
    for emotion, count in emotional_words.most_common():
        print(f"  {emotion.title()}: {count} instances")
    print()
    
    print("📏 MESSAGE LENGTH PATTERNS:")
    short_msgs = [m for m in messages if m['length'] <= 20]
    medium_msgs = [m for m in messages if 20 < m['length'] <= 100]
    long_msgs = [m for m in messages if m['length'] > 100]
    
    print(f"  Short (≤20 chars): {len(short_msgs)} ({len(short_msgs)/len(messages)*100:.1f}%)")
    print(f"  Medium (21-100 chars): {len(medium_msgs)} ({len(medium_msgs)/len(messages)*100:.1f}%)")
    print(f"  Long (>100 chars): {len(long_msgs)} ({len(long_msgs)/len(messages)*100:.1f}%)")
    print()
    
    print("🔥 MOST COMMON WORDS (excluding stop words):")
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'it', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'her', 'us', 'them', 's', 't', 'm', 're', 've', 'll', 'd'}
    
    filtered_words = [(word, count) for word, count in word_counter.most_common(50) if word.lower() not in stop_words and len(word) > 2]
    for word, count in filtered_words[:25]:
        print(f"  {word}: {count}")
    print()
    
    print("🚀 CONVERSATION STARTER SAMPLES:")
    unique_starters = list(set(conversation_starters))[:15]
    for i, starter in enumerate(unique_starters, 1):
        print(f"  {i}. '{starter}'")
    print()
    
    print("💬 SAMPLE LONG MESSAGES (showing communication style):")
    long_samples = [msg['body'] for msg in messages if msg['length'] > 150][:10]
    for i, msg in enumerate(long_samples, 1):
        print(f"  {i}. {msg[:200]}{'...' if len(msg) > 200 else ''}")
        print()
    
    print("🎯 UNIQUE EXPRESSIONS AND QUIRKS:")
    quirks = []
    for msg in messages[:500]:
        body = msg['body'].lower()
        # Look for unique expressions
        if 'lollllllll' in body:
            quirks.append("Extended 'lol' - 'lollllllll'")
        if 'fuck with' in body and 'love' not in body:
            quirks.append("'I fuck with' meaning 'I like'")
        if 'bet' in body and len(body) < 10:
            quirks.append("'Bet' as agreement/confirmation")
        if 'bestie' in body:
            quirks.append("Uses 'bestie' as endearment")
        if re.search(r'[a-z]+ ass [a-z]+', body):
            quirks.append("Uses 'X ass Y' construction")
        if 'periodt' in body or 'period' in body:
            quirks.append("'Period' for emphasis")
    
    unique_quirks = list(set(quirks))[:10]
    for quirk in unique_quirks:
        print(f"  • {quirk}")

if __name__ == "__main__":
    analyze_comprehensive_patterns()