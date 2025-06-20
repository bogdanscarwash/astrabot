# Conversation Schemas Reference

This document provides comprehensive reference for the conversation schemas module, which defines Pydantic models and enums for structured Signal conversation data processing.

## Overview

The `conversation_schemas` module provides data models for representing Signal conversations, relationship dynamics, message timing patterns, and enhanced message metadata. These schemas enable structured analysis and training data creation while preserving conversational context and relationships.

## Enumerations

### `RelationshipDynamic`

Types of relationship dynamics detected in conversations.

```python
from src.models.conversation_schemas import RelationshipDynamic
```

**Values:**
- `INTELLECTUAL_PEERS` = "intellectual_peers" - Conversations between intellectual equals discussing complex topics
- `MENTOR_STUDENT` = "mentor_student" - Teaching/learning relationship with knowledge transfer
- `CLOSE_FRIENDS` = "close_friends" - Intimate friendship with personal sharing and support
- `CASUAL_ACQUAINTANCES` = "casual_acquaintances" - Light, surface-level interactions
- `ROMANTIC_PARTNERS` = "romantic_partners" - Romantic relationship with emotional intimacy
- `POLITICAL_ALLIES` = "political_allies" - Shared political views and activism
- `DEBATE_PARTNERS` = "debate_partners" - Structured disagreement and argumentation

**Example:**
```python
from src.models.conversation_schemas import RelationshipDynamic

# Classify relationship based on conversation patterns
relationship = RelationshipDynamic.CLOSE_FRIENDS
print(relationship.value)  # "close_friends"

# Use in conversation analysis
if relationship == RelationshipDynamic.INTELLECTUAL_PEERS:
    # Apply academic language analysis
    pass
```

### `MessageTiming`

Message timing patterns for response analysis.

```python
from src.models.conversation_schemas import MessageTiming
```

**Values:**
- `IMMEDIATE` = "immediate" - Response within 30 seconds
- `QUICK` = "quick" - Response within 30 seconds to 2 minutes  
- `MODERATE` = "moderate" - Response within 2 to 15 minutes
- `DELAYED` = "delayed" - Response within 15 minutes to 1 hour
- `LATE` = "late" - Response after 1 hour

**Example:**
```python
from src.models.conversation_schemas import MessageTiming
from datetime import datetime, timedelta

def classify_response_timing(current_msg_time, previous_msg_time):
    """Classify message timing based on response delay."""
    delay = (current_msg_time - previous_msg_time).total_seconds()
    
    if delay < 30:
        return MessageTiming.IMMEDIATE
    elif delay < 120:  # 2 minutes
        return MessageTiming.QUICK
    elif delay < 900:  # 15 minutes
        return MessageTiming.MODERATE
    elif delay < 3600:  # 1 hour
        return MessageTiming.DELAYED
    else:
        return MessageTiming.LATE

# Usage example
timing = classify_response_timing(datetime.now(), datetime.now() - timedelta(minutes=5))
print(timing.value)  # "moderate"
```

## Core Classes

### `SignalMessage`

Enhanced Signal message with comprehensive analysis metadata.

```python
from src.models.conversation_schemas import SignalMessage
```

**Core Fields:**
- `message_id` (str): Unique message identifier
- `thread_id` (str): Thread/conversation identifier  
- `sender_id` (str): Sender's recipient ID
- `timestamp` (datetime): Message timestamp
- `body` (str): Message text content

**Analysis Metadata:**
- `message_type` (MessageType): Type of message in conversation flow
- `emotional_tone` (EmotionalTone): Emotional tone classification
- `topic_category` (Optional[TopicCategory]): Primary topic category
- `contains_emoji` (bool): Whether message contains emojis
- `emoji_list` (List[str]): List of emojis used in message
- `contains_url` (bool): Whether message contains URLs
- `url_list` (List[str]): List of URLs found in message

**Language Analysis:**
- `word_count` (int): Number of words in message
- `character_count` (int): Number of characters in message
- `contains_profanity` (bool): Whether message contains profanity
- `academic_language` (bool): Whether message uses academic language
- `internet_slang` (bool): Whether message uses internet slang

**Conversation Context:**
- `response_to_message_id` (Optional[str]): ID of message this responds to
- `time_since_previous` (Optional[float]): Seconds since previous message
- `is_correction` (bool): Whether this corrects a previous message
- `is_continuation` (bool): Whether this continues previous thought

**Example:**
```python
from datetime import datetime
from src.models.conversation_schemas import SignalMessage, MessageType, EmotionalTone

# Create enhanced message with metadata
message = SignalMessage(
    message_id="msg_001",
    thread_id="thread_123",
    sender_id="user_456",
    timestamp=datetime.now(),
    body="That's amazing! 😊 Check out this link: https://example.com",
    message_type=MessageType.RESPONSE,
    emotional_tone=EmotionalTone.HAPPY,
    contains_emoji=True,
    emoji_list=["😊"],
    contains_url=True,
    url_list=["https://example.com"],
    word_count=7,
    character_count=58,
    contains_profanity=False,
    academic_language=False,
    internet_slang=False,
    time_since_previous=45.0,
    is_correction=False,
    is_continuation=False
)

print(f"Message: {message.body}")
print(f"Emotional tone: {message.emotional_tone.value}")
print(f"Contains emojis: {message.contains_emoji}")
```

### `ConversationWindow`

Represents a sliding window of messages with context for training data creation.

```python
from src.models.conversation_schemas import ConversationWindow
```

**Fields:**
- `window_id` (str): Unique window identifier
- `messages` (List[SignalMessage]): Messages in this window
- `start_time` (datetime): Window start timestamp
- `end_time` (datetime): Window end timestamp
- `participants` (List[str]): Participant IDs in window
- `dominant_topic` (Optional[TopicCategory]): Primary topic discussed
- `relationship_dynamic` (Optional[RelationshipDynamic]): Detected relationship type
- `emotional_arc` (List[EmotionalTone]): Emotional progression through window

**Methods:**
- `to_training_format()`: Convert window to training data format
- `get_context_summary()`: Generate contextual summary
- `extract_dialogue_pairs()`: Extract question-answer pairs

**Example:**
```python
from src.models.conversation_schemas import ConversationWindow, RelationshipDynamic
from datetime import datetime

# Create conversation window
window = ConversationWindow(
    window_id="window_001",
    messages=[message1, message2, message3],
    start_time=datetime.now(),
    end_time=datetime.now(),
    participants=["user_123", "user_456"],
    relationship_dynamic=RelationshipDynamic.CLOSE_FRIENDS,
    emotional_arc=[EmotionalTone.CASUAL, EmotionalTone.HAPPY, EmotionalTone.EXCITED]
)

# Convert to training format
training_data = window.to_training_format()
print(training_data["instruction"])
print(training_data["response"])
```

### `ConversationThread`

Represents a complete conversation thread with metadata and analysis.

```python
from src.models.conversation_schemas import ConversationThread
```

**Fields:**
- `thread_id` (str): Unique thread identifier
- `participants` (List[str]): All participants in thread
- `start_time` (datetime): Thread start timestamp
- `end_time` (datetime): Thread end timestamp
- `message_count` (int): Total number of messages
- `windows` (List[ConversationWindow]): Conversation windows
- `primary_relationship` (RelationshipDynamic): Dominant relationship type
- `topic_evolution` (List[TopicCategory]): Topic progression
- `communication_patterns` (Dict[str, Any]): Detected patterns

**Methods:**
- `segment_into_windows()`: Create conversation windows
- `analyze_relationship_dynamics()`: Detect relationship patterns
- `extract_communication_style()`: Analyze communication styles
- `generate_thread_summary()`: Create thread summary

**Example:**
```python
from src.models.conversation_schemas import ConversationThread

# Create conversation thread
thread = ConversationThread(
    thread_id="thread_001",
    participants=["user_123", "user_456"],
    start_time=datetime.now(),
    end_time=datetime.now(),
    message_count=50,
    primary_relationship=RelationshipDynamic.INTELLECTUAL_PEERS,
    topic_evolution=[TopicCategory.TECHNOLOGY, TopicCategory.PHILOSOPHY]
)

# Segment into windows for analysis
windows = thread.segment_into_windows(window_size=10, overlap=2)
print(f"Created {len(windows)} conversation windows")

# Analyze communication patterns
patterns = thread.extract_communication_style()
print(f"Response timing: {patterns['avg_response_time']}")
print(f"Message length: {patterns['avg_message_length']}")
```

## Related Enums

### `MessageType`

Classification of message types in conversation flow.

**Values:**
- `INITIATION`: Conversation starter
- `RESPONSE`: Response to previous message
- `FOLLOW_UP`: Follow-up question or comment
- `TOPIC_SHIFT`: Change of conversation topic
- `CLARIFICATION`: Request for clarification
- `AGREEMENT`: Expression of agreement
- `DISAGREEMENT`: Expression of disagreement
- `EMOTIONAL_SUPPORT`: Providing emotional support
- `INFORMATION_SHARING`: Sharing factual information
- `HUMOR`: Joke or humorous comment

### `EmotionalTone`

Emotional tone classification for messages.

**Values:**
- `HAPPY`: Positive, joyful tone
- `SAD`: Negative, melancholic tone
- `ANGRY`: Frustrated or angry tone
- `EXCITED`: High-energy, enthusiastic tone
- `ANXIOUS`: Worried or nervous tone
- `CASUAL`: Relaxed, informal tone
- `SERIOUS`: Formal, serious tone
- `HUMOROUS`: Funny, playful tone
- `CONTEMPLATIVE`: Thoughtful, reflective tone
- `AFFECTIONATE`: Loving, caring tone

### `TopicCategory`

Primary topic categories for conversation classification.

**Values:**
- `PERSONAL_LIFE`: Personal experiences and life events
- `WORK_CAREER`: Professional and career topics
- `RELATIONSHIPS`: Relationship discussions
- `TECHNOLOGY`: Technology and digital topics
- `POLITICS`: Political discussions and opinions
- `ENTERTAINMENT`: Movies, music, games, etc.
- `FOOD`: Food, cooking, restaurants
- `TRAVEL`: Travel experiences and plans
- `HEALTH_FITNESS`: Health and fitness topics
- `PHILOSOPHY`: Philosophical discussions
- `CURRENT_EVENTS`: News and current events
- `EDUCATION`: Learning and educational topics

## Usage in Training Pipeline

The conversation schemas integrate with the training pipeline to create structured, context-aware training data:

```python
from src.models.conversation_schemas import (
    SignalMessage, ConversationWindow, ConversationThread,
    RelationshipDynamic, MessageTiming
)
from src.llm.training_data_creator import TrainingDataCreator

# Process Signal messages into structured format
messages = [
    SignalMessage.from_signal_data(row) 
    for _, row in signal_df.iterrows()
]

# Create conversation threads
thread = ConversationThread.from_messages(messages)

# Analyze relationship dynamics
thread.analyze_relationship_dynamics()

# Segment into training windows
windows = thread.segment_into_windows(
    window_size=10,
    overlap=2,
    preserve_context=True
)

# Generate training data with preserved context
creator = TrainingDataCreator()
training_examples = []

for window in windows:
    examples = creator.create_training_examples_from_window(
        window,
        preserve_relationship_context=True,
        include_timing_patterns=True
    )
    training_examples.extend(examples)

print(f"Generated {len(training_examples)} training examples")
```

## Schema Validation

All schemas include comprehensive validation to ensure data integrity:

```python
from pydantic import ValidationError
from src.models.conversation_schemas import SignalMessage

try:
    # Valid message
    message = SignalMessage(
        message_id="msg_001",
        thread_id="thread_001", 
        sender_id="user_001",
        timestamp=datetime.now(),
        body="Hello world!",
        word_count=2,
        character_count=12
    )
    print("Message created successfully")
    
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Integration with Analysis Modules

The conversation schemas work seamlessly with other analysis modules:

```python
from src.core.emoji_analyzer import EmojiAnalyzer
from src.core.style_analyzer import StyleAnalyzer
from src.models.conversation_schemas import SignalMessage

# Enhance message with emoji analysis
analyzer = EmojiAnalyzer()
style_analyzer = StyleAnalyzer()

def enhance_message_with_analysis(raw_message_data):
    """Enhance raw message with comprehensive analysis."""
    
    # Create base message
    message = SignalMessage.from_signal_data(raw_message_data)
    
    # Add emoji analysis
    emoji_analysis = analyzer.analyze_message_emoji_patterns(
        message.body, 
        message.sender_id, 
        message.timestamp
    )
    message.contains_emoji = emoji_analysis['has_emojis']
    message.emoji_list = emoji_analysis['emojis_list']
    
    # Add style analysis
    style_analysis = style_analyzer.analyze_message_style(message.body)
    message.academic_language = style_analysis['academic_language']
    message.internet_slang = style_analysis['internet_slang']
    
    return message
```
