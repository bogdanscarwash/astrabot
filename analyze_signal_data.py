#!/usr/bin/env python3
import csv
import re
from collections import Counter
import sys

def analyze_signal_data():
    # Read a sample of messages with actual body content
    messages = []
    word_counter = Counter()
    message_lengths = []
    emoji_patterns = []
    
    # Common emoji patterns to look for
    emoji_regex = re.compile(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]|[\U00002600-\U000027BF]|[\U0001F900-\U0001F9FF]|[\U00002700-\U000027BF]')
    
    with open('data/raw/signal-flatfiles/signal.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            body = row.get('body', '').strip()
            if body and body != '' and not body.startswith('http') and len(body) > 1:
                messages.append(body)
                message_lengths.append(len(body))
                
                # Find emojis
                emojis = emoji_regex.findall(body)
                emoji_patterns.extend(emojis)
                
                # Extract words (basic tokenization)
                words = re.findall(r'\b\w+\b', body.lower())
                word_counter.update(words)
                
                count += 1
                if count >= 1000:  # Sample first 1000 non-empty messages
                    break
    
    # Count emojis
    emoji_counter = Counter(emoji_patterns)
    
    print(f'Analyzed {len(messages)} messages')
    print(f'Average message length: {sum(message_lengths)/len(message_lengths):.1f} characters')
    print(f'Message length range: {min(message_lengths)} - {max(message_lengths)}')
    print()
    
    print('TOP 20 EMOJIS:')
    for emoji_char, count in emoji_counter.most_common(20):
        print(f'{emoji_char}: {count}')
    print()
    
    print('TOP 30 WORDS:')
    for word, count in word_counter.most_common(30):
        print(f'{word}: {count}')
    print()
    
    print('SAMPLE MESSAGES:')
    for i, msg in enumerate(messages[:20]):
        print(f'{i+1}: {msg[:150]}{"..." if len(msg) > 150 else ""}')
    print()
    
    # Analyze message length patterns
    short_msgs = [m for m in messages if len(m) <= 20]
    medium_msgs = [m for m in messages if 20 < len(m) <= 100]
    long_msgs = [m for m in messages if len(m) > 100]
    
    print(f'Message length distribution:')
    print(f'Short (≤20 chars): {len(short_msgs)} ({len(short_msgs)/len(messages)*100:.2f}%)')
    print(f'Medium (21-100 chars): {len(medium_msgs)} ({len(medium_msgs)/len(messages)*100:.1f}%)')
    print(f'Long (>100 chars): {len(long_msgs)} ({len(long_msgs)/len(messages)*100:.1f}%)')

if __name__ == "__main__":
    analyze_signal_data()