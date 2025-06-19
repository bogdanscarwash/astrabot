# TrainingDataCreator API Reference

The `TrainingDataCreator` class provides methods for creating training data from Signal conversations.

## Class: TrainingDataCreator

```python
from src.llm import TrainingDataCreator

creator = TrainingDataCreator(your_recipient_id=2)
```

### Constructor Parameters

- `your_recipient_id` (int): Your recipient ID in the Signal database. Default: 2

### Methods

#### create_conversation_windows

Creates sliding conversation windows for training.

```python
windows = creator.create_conversation_windows(
    messages_df=messages_df,
    window_size=5
)
```

**Parameters:**
- `messages_df` (pd.DataFrame): DataFrame containing messages
- `window_size` (int): Number of context messages. Default: 5

**Returns:**
- List[Dict]: Conversation windows with metadata

**Example Window:**
```python
{
    'thread_id': 123,
    'context': [
        {'speaker': 'Other', 'text': 'Hey!', 'timestamp': 1234567890},
        {'speaker': 'You', 'text': 'Hi there!', 'timestamp': 1234567891}
    ],
    'response': {
        'text': 'How are you?',
        'timestamp': 1234567892
    },
    'metadata': {
        'momentum': 'rapid',
        'context_size': 2,
        'avg_time_gap': 1.0,
        'response_delay': 1.0
    }
}
```

#### segment_natural_dialogues

Segments conversations into natural episodes based on time gaps.

```python
episodes = creator.segment_natural_dialogues(
    messages_df=messages_df,
    time_gap_minutes=30
)
```

**Parameters:**
- `messages_df` (pd.DataFrame): DataFrame containing messages
- `time_gap_minutes` (int): Minutes of inactivity for new episode. Default: 30

**Returns:**
- List[Dict]: Dialogue episodes with complete conversation arcs

#### analyze_personal_texting_style

Analyzes your personal communication patterns.

```python
style = creator.analyze_personal_texting_style(messages_df)
```

**Returns:**
```python
{
    'avg_message_length': 45.2,
    'message_length_distribution': {...},
    'burst_patterns': {
        'total_bursts': 156,
        'avg_burst_size': 3.2,
        'burst_frequency': 0.34,
        'max_burst_size': 8
    },
    'preferred_length': 'concise',
    'emoji_usage': {
        'emoji_frequency': 0.23,
        'messages_with_emojis': 450,
        'emoji_usage_rate': '23.0%'
    },
    'total_messages': 1956
}
```

#### create_training_examples

Main method to create comprehensive training examples.

```python
examples, style = creator.create_training_examples(
    messages_df=messages_df,
    recipients_df=recipients_df,
    include_twitter_content=True,
    max_examples=10000
)
```

**Parameters:**
- `messages_df` (pd.DataFrame): Messages data
- `recipients_df` (pd.DataFrame): Recipients data
- `include_twitter_content` (bool): Process Twitter links. Default: True
- `max_examples` (int, optional): Maximum examples to create

**Returns:**
- Tuple[List[Dict], Dict]: (training_examples, style_analysis)

#### save_training_data

Saves training data to JSON file.

```python
creator.save_training_data(
    training_examples=examples,
    output_path='training_data.json',
    style_analysis=style
)
```

## Utility Function

### create_training_data_from_signal

Convenience function for the complete pipeline.

```python
from src.llm import create_training_data_from_signal

result = create_training_data_from_signal(
    messages_csv_path='data/raw/signal-flatfiles/signal.csv',
    recipients_csv_path='data/raw/signal-flatfiles/recipient.csv', 
    output_path='outputs/training_data.json',
    your_recipient_id=2,
    include_twitter=True,
    max_examples=None
)
```

**Returns:**
```python
{
    'success': True,
    'total_examples': 5432,
    'style_analysis': {...},
    'output_path': 'outputs/training_data.json'
}
```

## Training Example Format

Each training example follows this structure:

```python
{
    'instruction': 'Continue this rapid conversation naturally',
    'input': 'Other: Hey!\nYou: Hi there!',
    'output': 'How are you doing?',
    'metadata': {
        'type': 'conversation_window',
        'momentum': 'rapid',
        'response_delay': 2.5,
        'context_size': 3
    }
}
```

## Error Handling

The module uses logging for error reporting:

```python
import logging
from src.utils import get_logger

# Enable debug logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
```

Common errors:
- `FileNotFoundError`: Check CSV file paths
- `KeyError`: Ensure CSV has required columns
- `ValueError`: Check your_recipient_id is correct