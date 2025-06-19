#!/usr/bin/env python3
import csv
import re
from collections import Counter, defaultdict

def analyze_linguistic_patterns():
    """Analyze specific linguistic patterns and communication habits"""
    
    messages = []
    slang_usage = Counter()
    sentence_starters = Counter()
    punctuation_patterns = Counter()
    typos_and_corrections = []
    
    # Define slang and colloquial patterns
    slang_patterns = {
        'bestie': r'\bbestie\b',
        'periodt': r'\bperiodt?\b',
        'bet': r'\bbet\b(?!\w)',  # 'bet' as standalone word
        'fr/for real': r'\b(fr|for real)\b',
        'ngl': r'\bngl\b',
        'tbh': r'\btbh\b',
        'lowkey': r'\blowkey\b',
        'highkey': r'\bhighkey\b',
        'sus': r'\bsus\b',
        'cap/no cap': r'\b(cap|no cap)\b',
        'slaps': r'\bslaps\b',
        'hits different': r'hits different',
        'valid': r'\bvalid\b(?! \w)',
        'based': r'\bbased\b',
        'cringe': r'\bcringe\b',
        'bussin': r'\bbussin\b',
        'slay': r'\bslay\b',
        'queen': r'\bqueen\b',
        'king': r'\bking\b',
    }
    
    # Intensifiers and emphasis
    intensifiers = {
        'fucking': r'\bfucking\b',
        'literally': r'\bliterally\b',
        'actually': r'\bactually\b',
        'totally': r'\btotally\b',
        'super': r'\bsuper\b(?! \w)',
        'really': r'\breally\b',
        'so': r'\bso\b(?= \w)',
        'very': r'\bvery\b',
        'extremely': r'\bextremely\b',
        'absolutely': r'\babsolutely\b',
    }
    
    # Communication habits
    communication_patterns = {
        'all_caps': 0,
        'multiple_punctuation': 0,
        'ellipsis': 0,
        'questions': 0,
        'exclamations': 0,
        'typo_corrections': 0,
        'incomplete_thoughts': 0,
    }
    
    with open('data/raw/signal-flatfiles/signal.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        previous_message = ""
        
        for row in reader:
            body = row.get('body', '').strip()
            
            if body and body != '' and not body.startswith('http') and len(body) > 1:
                messages.append(body)
                body_lower = body.lower()
                
                # Check for slang usage
                for slang, pattern in slang_patterns.items():
                    if re.search(pattern, body_lower):
                        slang_usage[slang] += 1
                
                # Check for intensifiers
                for intensifier, pattern in intensifiers.items():
                    if re.search(pattern, body_lower):
                        slang_usage[f"intensifier_{intensifier}"] += 1
                
                # Sentence starters
                first_words = body.split()[:2]
                if first_words:
                    starter = ' '.join(first_words).lower()
                    if len(starter) > 2:
                        sentence_starters[starter] += 1
                
                # Communication pattern analysis
                if body.isupper() and len(body) > 3:
                    communication_patterns['all_caps'] += 1
                
                if re.search(r'[!.?]{2,}', body):
                    communication_patterns['multiple_punctuation'] += 1
                
                if '...' in body or '…' in body:
                    communication_patterns['ellipsis'] += 1
                
                if body.endswith('?'):
                    communication_patterns['questions'] += 1
                
                if body.endswith('!'):
                    communication_patterns['exclamations'] += 1
                
                # Look for corrections (similar to previous message but edited)
                if previous_message and len(body) > 10 and len(previous_message) > 10:
                    # Simple similarity check
                    words_current = set(body.lower().split())
                    words_previous = set(previous_message.lower().split())
                    similarity = len(words_current & words_previous) / max(len(words_current), len(words_previous))
                    if 0.6 < similarity < 0.95:  # Likely a correction
                        communication_patterns['typo_corrections'] += 1
                        typos_and_corrections.append((previous_message, body))
                
                # Incomplete thoughts (ending with commas, no punctuation, etc.)
                if body.endswith(',') or (not body[-1] in '.!?' and len(body) > 10):
                    communication_patterns['incomplete_thoughts'] += 1
                
                previous_message = body
                count += 1
                if count >= 2000:
                    break
    
    # Print analysis
    print("=== LINGUISTIC PATTERNS & COMMUNICATION STYLE ===\n")
    
    print("🗣️ SLANG AND COLLOQUIAL EXPRESSIONS:")
    for term, count in slang_usage.most_common(20):
        if not term.startswith('intensifier_'):
            print(f"  '{term}': {count} times")
    print()
    
    print("💥 INTENSIFIERS AND EMPHASIS:")
    for term, count in slang_usage.most_common():
        if term.startswith('intensifier_'):
            clean_term = term.replace('intensifier_', '')
            print(f"  '{clean_term}': {count} times")
    print()
    
    print("🎬 COMMON SENTENCE STARTERS:")
    filtered_starters = [(starter, count) for starter, count in sentence_starters.most_common(20) 
                        if count > 2 and not starter.startswith(('i ', 'the ', 'a ', 'and '))]
    for starter, count in filtered_starters[:15]:
        print(f"  '{starter}...': {count} times")
    print()
    
    print("📝 PUNCTUATION & STYLE PATTERNS:")
    total_messages = len(messages)
    for pattern, count in communication_patterns.items():
        percentage = (count / total_messages) * 100
        pattern_name = pattern.replace('_', ' ').title()
        print(f"  {pattern_name}: {count} times ({percentage:.1f}% of messages)")
    print()
    
    print("✏️ EXAMPLE CORRECTIONS/REVISIONS:")
    for i, (original, corrected) in enumerate(typos_and_corrections[:5], 1):
        print(f"  {i}. Original: '{original[:80]}{'...' if len(original) > 80 else ''}'")
        print(f"     Revised:  '{corrected[:80]}{'...' if len(corrected) > 80 else ''}'")
        print()
    
    print("🔤 UNIQUE SPELLINGS & VARIANTS:")
    unique_spellings = []
    for msg in messages[:500]:
        # Look for creative spellings or variants
        words = msg.lower().split()
        for word in words:
            if 'lollllll' in word:
                unique_spellings.append(word)
            elif re.search(r'(.)\1{2,}', word) and len(word) > 4:  # Repeated characters
                unique_spellings.append(word)
            elif word.endswith('ass') and len(word) > 3:
                unique_spellings.append(f"'{word}' (as intensifier)")
    
    unique_list = list(set(unique_spellings))[:10]
    for spelling in unique_list:
        print(f"  {spelling}")

if __name__ == "__main__":
    analyze_linguistic_patterns()