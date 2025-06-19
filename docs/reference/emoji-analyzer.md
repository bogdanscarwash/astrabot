# EmojiAnalyzer Reference

This document provides comprehensive reference for the EmojiAnalyzer module, which provides sophisticated emoji analysis capabilities for conversation data.

## Overview

The `EmojiAnalyzer` class analyzes emoji usage patterns, emotional correlations, and personal emoji signatures in Signal conversation data. It includes context-aware emoji interpretation, sentiment analysis, and emotional state tracking through emoji usage.

## Core Class

### `src.core.emoji_analyzer.EmojiAnalyzer`

Comprehensive emoji analysis for conversation data with emotional mapping and signature detection.

#### Initialization

```python
from src.core.emoji_analyzer import EmojiAnalyzer

analyzer = EmojiAnalyzer()
```

The analyzer initializes with predefined emotion mappings that categorize emojis into emotional categories like joy/laughter, love/affection, sadness/crying, anger/frustration, and others based on Signal conversation analysis.

## Core Methods

### `analyze_message_emoji_patterns(message, sender_id, timestamp)`

Analyze emoji patterns in a single message with comprehensive emotional and contextual analysis.

**Parameters:**
- `message` (str): Message text content to analyze
- `sender_id` (str): Sender's recipient ID for tracking patterns
- `timestamp` (datetime): Message timestamp for temporal analysis

**Returns:**
- Dict[str, Any]: Comprehensive emoji analysis containing:
  - `has_emojis` (bool): Whether message contains emojis
  - `emoji_count` (int): Total number of emojis in message
  - `unique_emojis` (int): Number of unique emoji types
  - `emojis_list` (List[str]): List of all emojis found
  - `emotional_intensity` (float): Emotional intensity score (0-1)
  - `avg_sentiment` (float): Average sentiment score (-1 to 1)
  - `dominant_emotion` (str): Most prevalent emotion category
  - `emotion_categories` (List[str]): All emotion categories present
  - `position_patterns` (Counter): Distribution of emoji positions
  - `sender_id` (str): Sender identifier
  - `timestamp` (datetime): Message timestamp

**Example:**
```python
from datetime import datetime
from src.core.emoji_analyzer import EmojiAnalyzer

analyzer = EmojiAnalyzer()
message = "This is amazing! 😊 😊 ❤️ So happy! 🎉"
analysis = analyzer.analyze_message_emoji_patterns(
    message, 
    "user123", 
    datetime.now()
)

print(analysis['has_emojis'])  # True
print(analysis['emoji_count'])  # 4
print(analysis['unique_emojis'])  # 3
print(analysis['dominant_emotion'])  # 'joy_laughter'
print(analysis['emotional_intensity'])  # 0.9 (high positive intensity)
```

**No Emoji Example:**
```python
message = "This is a regular message without any emojis"
analysis = analyzer.analyze_message_emoji_patterns(message, "user123", datetime.now())

print(analysis['has_emojis'])  # False
print(analysis['emoji_count'])  # 0
print(analysis['emotional_intensity'])  # 0.0
print(analysis['dominant_emotion'])  # None
```

### `detect_emoji_signatures(messages_df, min_usage=5)`

Detect personal emoji signatures for each sender based on usage patterns and frequency.

**Parameters:**
- `messages_df` (pd.DataFrame): DataFrame containing Signal messages with columns:
  - `body` (str): Message text content
  - `from_recipient_id` (int/str): Sender's recipient ID
- `min_usage` (int, optional): Minimum usage count to consider as signature. Defaults to 5.

**Returns:**
- Dict[str, List[EmojiUsagePattern]]: Dictionary mapping sender IDs to their emoji signature patterns:
  - Key: Sender ID as string
  - Value: List of `EmojiUsagePattern` objects containing:
    - `emoji` (str): The emoji character
    - `frequency` (int): Usage count
    - `emotional_category` (str): Emotion category classification
    - `usage_context` (str): Most common position context
    - `sender_signature` (bool): Whether this is a signature emoji (>5% usage rate)

**Example:**
```python
import pandas as pd
from src.core.emoji_analyzer import EmojiAnalyzer

# Sample messages DataFrame
messages_df = pd.DataFrame({
    'from_recipient_id': [2, 2, 3, 2, 3, 2],
    'body': [
        "Great job! 👍 😊",
        "I'm so happy! ❤️ 🎉", 
        "That's terrible 😢 😡",
        "Multiple same emoji! 😂 😂 😂",
        "Just kidding! 😜 😈",
        "Mixed emotions 😊 😢 🤔"
    ]
})

analyzer = EmojiAnalyzer()
signatures = analyzer.detect_emoji_signatures(messages_df, min_usage=2)

# Check user 2's signatures
if '2' in signatures:
    user2_sigs = signatures['2']
    for sig in user2_sigs:
        print(f"Emoji: {sig.emoji}, Frequency: {sig.frequency}, Signature: {sig.sender_signature}")
        # Output: Emoji: 😂, Frequency: 3, Signature: True
```

## Additional Methods

### `extract_emojis_from_text(text)`

Extract emojis with their position and contextual information.

**Parameters:**
- `text` (str): Text to analyze for emojis

**Returns:**
- List[Dict[str, Any]]: List of emoji data with position, context, and emotion mapping

**Example:**
```python
text = "Hello! 😊 How are you? 👋 Great day! 🌞"
emojis = analyzer.extract_emojis_from_text(text)

for emoji_data in emojis:
    print(f"Emoji: {emoji_data['emoji']}")
    print(f"Position: {emoji_data['position_type']}")
    print(f"Context: {emoji_data['context']}")
    print(f"Emotion: {emoji_data['emotion_category']}")
```

### `generate_emoji_insights(messages_df)`

Generate comprehensive emoji insights from conversation data.

**Parameters:**
- `messages_df` (pd.DataFrame): DataFrame containing conversation messages

**Returns:**
- Dict[str, Any]: Comprehensive insights including:
  - `usage_statistics`: Overall emoji usage metrics
  - `emoji_signatures`: Personal emoji signatures per sender
  - `emotional_patterns`: Emotional correlation analysis
  - `emotion_distribution`: Distribution of emotion categories
  - `sentiment_analysis`: Overall sentiment trends

**Example:**
```python
insights = analyzer.generate_emoji_insights(messages_df)

stats = insights['usage_statistics']
print(f"Emoji usage rate: {stats['emoji_usage_rate']:.2%}")
print(f"Most common emojis: {stats['most_common_emojis']}")

sentiment = insights['sentiment_analysis']
print(f"Average sentiment: {sentiment['avg_sentiment']:.2f}")
print(f"Positive ratio: {sentiment['positive_ratio']:.2%}")
```

### `analyze_emotional_state_correlation(messages_df, time_window_minutes=30)`

Analyze how emoji usage correlates with emotional states over time windows.

**Parameters:**
- `messages_df` (pd.DataFrame): DataFrame with messages and timestamps
- `time_window_minutes` (int, optional): Time window size in minutes. Defaults to 30.

**Returns:**
- Dict[str, Any]: Emotional correlation analysis with timeline data

### `interpret_emoji_in_context(emoji_char, message_context, sender_history=None)`

Provide context-aware interpretation of emoji usage including sarcasm and emphasis detection.

**Parameters:**
- `emoji_char` (str): The emoji to interpret
- `message_context` (str): Full message context
- `sender_history` (List[str], optional): Previous messages for context

**Returns:**
- Dict[str, Any]: Context-aware interpretation with modifiers

**Example:**
```python
# Sarcasm detection
interpretation = analyzer.interpret_emoji_in_context(
    '😊', 
    "Yeah right, that's totally going to work 😊"
)
print(interpretation['context_modifier'])  # 'sarcastic'

# Emphasis detection  
interpretation = analyzer.interpret_emoji_in_context(
    '😂',
    "That was so funny 😂😂😂"
)
print(interpretation['context_modifier'])  # 'emphasized'
```

## Emotion Categories

The EmojiAnalyzer uses predefined emotion categories based on Signal conversation analysis:

- **joy_laughter**: 😂, 🤣, 😄, 😃, 😁, 😊, 😀, 🙂, 😋, 😌, 😆
- **love_affection**: ❤️, 💖, 💕, 💗, 💓, 💘, 💝, 💜, 🧡, 💛, 💚, 💙, 🤍, 🖤, 🤎, 💯, 😍, 🥰, 😘, 😗, 😙, 😚
- **sadness_crying**: 😢, 😭, 😞, 😔, 😟, 😕, 🙁, ☹️, 😩, 😫
- **anger_frustration**: 😠, 😡, 🤬, 😤, 💢, 😾, 😖, 😣
- **surprise_shock**: 😮, 😯, 😲, 🤯, 😱, 🙀, 😳
- **thinking_contemplation**: 🤔, 🧐, 🤨, 🙄, 😐, 😑, 🤐
- **playful_teasing**: 😏, 😜, 😝, 😛, 🤪, 🤭, 😈, 👿, 🤡
- **support_encouragement**: 👍, 👌, ✌️, 🤝, 👏, 🙌, 💪, 🎉, 🎊, ✨
- **confusion_uncertainty**: 😵, 🤷, 🤦, 😅, 😬
- **cool_casual**: 😎, 🤓, 🥶, 🥵, 🤠

## Usage in Training Pipeline

The EmojiAnalyzer integrates with the training pipeline to preserve emoji usage patterns and emotional context in generated training data:

```python
from src.core.emoji_analyzer import EmojiAnalyzer
from src.llm.training_data_creator import TrainingDataCreator

analyzer = EmojiAnalyzer()
creator = TrainingDataCreator()

# Analyze emoji patterns before creating training data
emoji_insights = analyzer.generate_emoji_insights(messages_df)
signatures = analyzer.detect_emoji_signatures(messages_df)

# Use insights to inform training data creation
training_data = creator.create_training_examples(
    messages_df, 
    emoji_patterns=signatures
)
```

## Related Models

- `EmojiUsagePattern`: Pydantic model for emoji signature patterns
- `EmotionalTone`: Enum for emotional tone classification
- `SignalMessage`: Enhanced message model with emoji metadata
