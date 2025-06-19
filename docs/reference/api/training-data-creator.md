# TrainingDataCreator API Reference

The `TrainingDataCreator` class provides comprehensive methods for transforming Signal conversations into high-quality training data for language model fine-tuning. It preserves natural conversation flow, communication styles, and enriches content with external data.

## Table of Contents
- [Class Overview](#class-overview)
- [Constructor](#constructor)
- [Core Methods](#core-methods)
- [Advanced Methods](#advanced-methods)
- [Utility Functions](#utility-functions)
- [Usage Patterns](#usage-patterns)
- [Performance Optimization](#performance-optimization)
- [Integration Examples](#integration-examples)
- [Best Practices](#best-practices)

## Class Overview

```python
from src.llm.training_data_creator import TrainingDataCreator

# Basic initialization
creator = TrainingDataCreator(your_recipient_id=2)

# With custom configuration
creator = TrainingDataCreator(
    your_recipient_id=2,
    min_message_length=10,
    include_system_messages=False,
    preserve_message_order=True
)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `your_recipient_id` | int | 2 | Your recipient ID in the Signal database |
| `min_message_length` | int | 5 | Minimum character length for messages |
| `include_system_messages` | bool | False | Include Signal system messages |
| `preserve_message_order` | bool | True | Maintain chronological order |

## Core Methods

### create_conversation_windows

Creates sliding conversation windows that capture natural dialogue flow with surrounding context.

```python
windows = creator.create_conversation_windows(
    messages_df=messages_df,
    window_size=5,
    stride=1,
    min_window_messages=3
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages_df` | pd.DataFrame | required | DataFrame containing messages |
| `window_size` | int | 5 | Number of context messages |
| `stride` | int | 1 | Window sliding step size |
| `min_window_messages` | int | 3 | Minimum messages required in window |

**Returns:** List[Dict] - Conversation windows with metadata

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
        'momentum': 'rapid',  # rapid/moderate/slow
        'context_size': 2,
        'avg_time_gap': 1.0,  # seconds
        'response_delay': 1.0,
        'conversation_partner': 'Alice',
        'has_media': False
    }
}
```

**Advanced Example:**
```python
# Create windows with specific configuration
windows = creator.create_conversation_windows(
    messages_df,
    window_size=7,  # More context
    stride=2,  # Skip every other window
    min_window_messages=4  # Require substantial context
)

# Filter windows by momentum
rapid_windows = [w for w in windows if w['metadata']['momentum'] == 'rapid']
print(f"Found {len(rapid_windows)} rapid conversation windows")

# Analyze window characteristics
avg_delay = np.mean([w['metadata']['response_delay'] for w in windows])
print(f"Average response delay: {avg_delay:.1f} seconds")
```

### segment_natural_dialogues

Segments conversations into natural episodes based on time gaps, creating complete conversation arcs.

```python
episodes = creator.segment_natural_dialogues(
    messages_df=messages_df,
    time_gap_minutes=30,
    min_episode_length=5,
    max_episode_length=50
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages_df` | pd.DataFrame | required | DataFrame containing messages |
| `time_gap_minutes` | int | 30 | Minutes of inactivity for new episode |
| `min_episode_length` | int | 3 | Minimum messages per episode |
| `max_episode_length` | int | 100 | Maximum messages per episode |

**Returns:** List[Dict] - Dialogue episodes with complete conversation arcs

**Example Episode:**
```python
{
    'episode_id': 'ep_123_456',
    'thread_id': 123,
    'messages': [
        {'speaker': 'You', 'text': 'Just saw the news!', 'timestamp': 1234567890},
        {'speaker': 'Other', 'text': 'What happened?', 'timestamp': 1234567920},
        # ... more messages
    ],
    'metadata': {
        'duration_seconds': 1800,
        'message_count': 15,
        'participant_turns': {'You': 8, 'Other': 7},
        'initiatior': 'You',
        'avg_response_time': 45.3,
        'conversation_type': 'discussion'  # chat/discussion/planning
    }
}
```

**Usage Example:**
```python
# Segment with custom parameters
episodes = creator.segment_natural_dialogues(
    messages_df,
    time_gap_minutes=60,  # Longer gaps for episode breaks
    min_episode_length=10  # Only substantial conversations
)

# Analyze episode patterns
episode_lengths = [e['metadata']['message_count'] for e in episodes]
print(f"Episode statistics:")
print(f"  Total: {len(episodes)}")
print(f"  Average length: {np.mean(episode_lengths):.1f} messages")
print(f"  Longest: {max(episode_lengths)} messages")

# Find discussion episodes
discussions = [e for e in episodes 
              if e['metadata']['conversation_type'] == 'discussion']
```

### analyze_personal_texting_style

Comprehensively analyzes your personal communication patterns across all conversations.

```python
style = creator.analyze_personal_texting_style(
    messages_df,
    include_word_frequency=True,
    analyze_time_patterns=True
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages_df` | pd.DataFrame | required | Your messages |
| `include_word_frequency` | bool | False | Analyze word usage |
| `analyze_time_patterns` | bool | False | Analyze timing patterns |

**Returns:** Dict - Comprehensive style analysis

**Example Output:**
```python
{
    'avg_message_length': 45.2,
    'message_length_distribution': {
        'very_short': 234,  # <10 chars
        'short': 567,       # 10-30 chars
        'medium': 890,      # 30-100 chars
        'long': 345,        # 100-200 chars
        'very_long': 123    # >200 chars
    },
    'burst_patterns': {
        'total_bursts': 156,
        'avg_burst_size': 3.2,
        'burst_frequency': 0.34,
        'max_burst_size': 8,
        'burst_intervals': [1.2, 2.3, 1.8]  # avg seconds between burst messages
    },
    'preferred_length': 'concise',
    'emoji_usage': {
        'emoji_frequency': 0.23,
        'messages_with_emojis': 450,
        'emoji_usage_rate': '23.0%',
        'top_emojis': ['😂', '👍', '❤️', '🤔', '😊'],
        'emoji_positions': {'start': 0.2, 'middle': 0.3, 'end': 0.5}
    },
    'time_patterns': {
        'most_active_hours': [20, 21, 22],  # 8-11 PM
        'most_active_days': ['Friday', 'Saturday'],
        'avg_messages_per_day': 45.6,
        'response_time_percentiles': {
            'p50': 120,  # seconds
            'p75': 600,
            'p90': 3600
        }
    },
    'vocabulary': {
        'unique_words': 3456,
        'avg_words_per_message': 7.8,
        'most_common_words': [
            ('yeah', 234),
            ('haha', 189),
            ('sounds', 156)
        ],
        'question_frequency': 0.34
    },
    'total_messages': 1956
}
```

### create_training_examples

Main method to create comprehensive training examples with multiple strategies.

```python
examples, style = creator.create_training_examples(
    messages_df=messages_df,
    recipients_df=recipients_df,
    strategies=['conversational', 'burst', 'adaptive'],
    include_twitter_content=True,
    max_examples=10000
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages_df` | pd.DataFrame | required | Messages data |
| `recipients_df` | pd.DataFrame | required | Recipients data |
| `strategies` | List[str] | ['conversational'] | Training strategies to use |
| `include_twitter_content` | bool | True | Process Twitter links |
| `max_examples` | int | None | Maximum examples to create |
| `balance_examples` | bool | True | Balance example types |

**Available Strategies:**
- `'conversational'` - Natural dialogue windows
- `'burst'` - Multi-message sequences
- `'adaptive'` - Style-adapted responses
- `'completion'` - Message completion
- `'qa'` - Question-answer pairs

**Returns:** Tuple[List[Dict], Dict] - (training_examples, style_analysis)

**Complex Example:**
```python
# Create diverse training data
examples, style = creator.create_training_examples(
    messages_df,
    recipients_df,
    strategies=['conversational', 'burst', 'adaptive'],
    include_twitter_content=True,
    max_examples=10000
)

# Analyze example distribution
example_types = {}
for ex in examples:
    ex_type = ex.get('metadata', {}).get('type', 'unknown')
    example_types[ex_type] = example_types.get(ex_type, 0) + 1

print("Training data distribution:")
for ex_type, count in example_types.items():
    print(f"  {ex_type}: {count} ({count/len(examples)*100:.1f}%)")

# Sample examples by type
for ex_type in example_types:
    sample = next(e for e in examples if e.get('metadata', {}).get('type') == ex_type)
    print(f"\n{ex_type} example:")
    print(f"  Instruction: {sample['instruction']}")
    print(f"  Input: {sample['input'][:100]}...")
    print(f"  Output: {sample['output'][:100]}...")
```

### save_training_data

Saves training data with metadata and validation.

```python
creator.save_training_data(
    training_examples=examples,
    output_path='training_data.json',
    style_analysis=style,
    format='jsonl',
    validate=True
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `training_examples` | List[Dict] | required | Training examples |
| `output_path` | str | required | Output file path |
| `style_analysis` | Dict | None | Style analysis to include |
| `format` | str | 'json' | Output format (json/jsonl) |
| `validate` | bool | True | Validate before saving |
| `compress` | bool | False | Gzip compress output |

**Example with Validation:**
```python
# Save with full validation
try:
    creator.save_training_data(
        examples,
        'data/training/conversation_data.jsonl',
        style_analysis=style,
        format='jsonl',  # One example per line
        validate=True,
        compress=True  # Creates .jsonl.gz
    )
    print("✓ Training data saved successfully")
except ValidationError as e:
    print(f"✗ Validation failed: {e}")
    # Handle invalid examples
    valid_examples = [ex for ex in examples if validate_example(ex)]
    creator.save_training_data(valid_examples, 'data/training/valid_only.jsonl')
```

## Advanced Methods

### create_conversational_data

Creates training data focusing on natural conversation flow.

```python
conversational_data = creator.create_conversational_data(
    messages_df,
    window_size=5,
    include_metadata=True,
    momentum_filter='rapid'
)
```

**Example with Filtering:**
```python
# Create data for rapid conversations only
rapid_data = creator.create_conversational_data(
    messages_df,
    momentum_filter='rapid',
    min_context_messages=3
)

# Create data for specific conversation partners
partner_messages = messages_df[messages_df['thread_id'].isin(partner_threads)]
partner_data = creator.create_conversational_data(partner_messages)
```

### create_burst_sequence_data

Captures multi-message sequences sent in quick succession.

```python
burst_data = creator.create_burst_sequence_data(
    messages_df,
    max_time_between_messages=120,  # 2 minutes
    min_burst_size=2,
    max_burst_size=10
)
```

**Burst Example:**
```python
{
    'instruction': 'Continue this message burst naturally',
    'input': 'You: Hey I just realized something\nYou: Remember when we talked about that project?',
    'output': 'I think we should reconsider the timeline',
    'metadata': {
        'type': 'burst_sequence',
        'burst_size': 3,
        'avg_time_between': 15.2,
        'burst_momentum': 'rapid'
    }
}
```

### create_adaptive_training_data

Creates examples showing how you adapt to different conversation partners.

```python
adaptive_data = creator.create_adaptive_training_data(
    messages_df,
    recipients_df,
    min_style_difference=0.5,
    min_conversations_per_partner=20
)
```

**Adaptive Example:**
```python
{
    'instruction': 'Respond in a style adapted to this conversation partner who prefers lengthy messages',
    'input': 'Partner: I\'ve been thinking about our discussion yesterday...[long message]',
    'output': 'That\'s a really interesting perspective! I was also considering...[matching length]',
    'metadata': {
        'type': 'adaptive',
        'partner_style': 'lengthy_texter',
        'your_usual_style': 'concise',
        'adaptation_level': 0.7
    }
}
```

## Utility Functions

### create_training_data_from_signal

Complete pipeline convenience function with all options.

```python
from src.llm import create_training_data_from_signal

result = create_training_data_from_signal(
    messages_csv_path='data/raw/signal-flatfiles/signal.csv',
    recipients_csv_path='data/raw/signal-flatfiles/recipient.csv',
    output_path='outputs/training_data.json',
    your_recipient_id=2,
    include_twitter=True,
    include_images=True,
    strategies=['conversational', 'burst', 'adaptive'],
    max_examples=10000,
    filters={
        'min_message_length': 10,
        'exclude_blocked': True,
        'date_range': ('2023-01-01', '2024-01-01')
    }
)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages_csv_path` | str | required | Path to signal.csv |
| `recipients_csv_path` | str | required | Path to recipient.csv |
| `output_path` | str | required | Output JSON path |
| `your_recipient_id` | int | 2 | Your Signal recipient ID |
| `include_twitter` | bool | True | Extract Twitter content |
| `include_images` | bool | True | Describe images with AI |
| `strategies` | List[str] | ['conversational'] | Training strategies |
| `max_examples` | int | None | Maximum examples |
| `filters` | Dict | {} | Additional filters |

**Returns:**
```python
{
    'success': True,
    'total_examples': 5432,
    'examples_by_type': {
        'conversational': 3000,
        'burst': 1500,
        'adaptive': 932
    },
    'style_analysis': {
        'avg_message_length': 45.2,
        'preferred_length': 'concise',
        # ... full style analysis
    },
    'output_path': 'outputs/training_data.json',
    'processing_time': 124.5,
    'enhanced_messages': 234
}
```

### validate_training_data

Validates training data structure and quality.

```python
from src.llm.training_data_creator import validate_training_data

is_valid, errors = validate_training_data(training_examples)

if not is_valid:
    print(f"Found {len(errors)} validation errors:")
    for error in errors[:5]:
        print(f"  - {error}")
```

## Usage Patterns

### Pattern 1: Quick Start

```python
# Simplest usage - process everything with defaults
from src.llm import create_training_data_from_signal

result = create_training_data_from_signal(
    'data/raw/signal-flatfiles/signal.csv',
    'data/raw/signal-flatfiles/recipient.csv',
    'training_data.json'
)
```

### Pattern 2: Filtered Processing

```python
# Process only recent conversations with specific partners
creator = TrainingDataCreator()

# Load and filter data
messages_df = pd.read_csv('signal.csv')
recipients_df = pd.read_csv('recipient.csv')

# Filter by date
cutoff = datetime.now() - timedelta(days=180)
recent_messages = messages_df[
    messages_df['date_sent'] > int(cutoff.timestamp() * 1000)
]

# Filter by conversation activity
active_threads = messages_df.groupby('thread_id').size()
active_threads = active_threads[active_threads >= 50].index
filtered_messages = recent_messages[
    recent_messages['thread_id'].isin(active_threads)
]

# Create training data
examples, style = creator.create_training_examples(
    filtered_messages,
    recipients_df,
    strategies=['conversational', 'burst']
)
```

### Pattern 3: Style-Specific Training

```python
# Create different training sets for different aspects
creator = TrainingDataCreator()

# Conversational flow training
conv_examples = creator.create_conversational_data(
    messages_df,
    window_size=7,
    momentum_filter=None  # All conversation speeds
)

# Burst pattern training
burst_examples = creator.create_burst_sequence_data(
    messages_df,
    min_burst_size=3  # Only substantial bursts
)

# Adaptive style training
adaptive_examples = creator.create_adaptive_training_data(
    messages_df,
    recipients_df,
    min_style_difference=0.7  # Only significant adaptations
)

# Combine with weights
all_examples = (
    conv_examples * 3 +  # 3x weight on conversational
    burst_examples * 2 + # 2x weight on bursts
    adaptive_examples    # 1x weight on adaptive
)
random.shuffle(all_examples)
```

## Performance Optimization

### Memory-Efficient Processing

```python
def process_large_dataset(csv_path, chunk_size=10000):
    """Process large datasets in chunks"""
    creator = TrainingDataCreator()
    all_examples = []
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Process chunk
        examples, _ = creator.create_training_examples(
            chunk,
            recipients_df,
            max_examples=1000  # Limit per chunk
        )
        all_examples.extend(examples)
        
        # Free memory
        del chunk
        gc.collect()
        
        print(f"Processed {len(all_examples)} examples so far...")
    
    return all_examples
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def process_thread_group(args):
    """Process a single thread's messages"""
    thread_id, messages, creator = args
    return creator.create_conversational_data(messages)

def parallel_process(messages_df, num_workers=4):
    """Process threads in parallel"""
    creator = TrainingDataCreator()
    
    # Group by thread
    thread_groups = [
        (tid, group) 
        for tid, group in messages_df.groupby('thread_id')
    ]
    
    # Create partial function with creator
    process_func = partial(process_thread_group, creator=creator)
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_func, thread_groups)
    
    # Flatten results
    return [ex for thread_examples in results for ex in thread_examples]
```

### Caching for Repeated Processing

```python
import pickle
from pathlib import Path

class CachedTrainingDataCreator(TrainingDataCreator):
    def __init__(self, cache_dir='cache/training_data', **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def create_training_examples(self, messages_df, recipients_df, **kwargs):
        # Create cache key
        cache_key = f"{len(messages_df)}_{kwargs.get('strategies', ['conv'])}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Process normally
        result = super().create_training_examples(
            messages_df, recipients_df, **kwargs
        )
        
        # Cache result
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

## Integration Examples

### Integration with ConversationProcessor

```python
from src.core.conversation_processor import ConversationProcessor
from src.llm.training_data_creator import TrainingDataCreator

# Process conversations with enhancement
processor = ConversationProcessor(
    your_recipient_id=2,
    enhance_with_twitter=True,
    include_images=True
)

# Enhance messages
enhanced_messages = processor.process_conversations(
    messages_df,
    recipients_df
)

# Create training data from enhanced messages
creator = TrainingDataCreator()
training_data = creator.create_training_examples(
    enhanced_messages,
    recipients_df,
    include_twitter_content=False  # Already enhanced
)
```

### Integration with Style Analysis

```python
from src.core.style_analyzer import StyleAnalyzer

# Analyze style first
analyzer = StyleAnalyzer()
style_profiles = analyzer.analyze_all_communication_styles(
    messages_df,
    your_recipient_id=2
)

# Create style-specific training data
creator = TrainingDataCreator()

for partner_id, style_profile in style_profiles.items():
    # Get messages for this partner
    partner_messages = messages_df[
        (messages_df['from_recipient_id'] == partner_id) |
        (messages_df['to_recipient_id'] == partner_id)
    ]
    
    # Create adapted training data
    if style_profile['classification'] == 'lengthy_texter':
        examples = creator.create_training_examples(
            partner_messages,
            recipients_df,
            strategies=['adaptive'],
            filters={'min_message_length': 50}
        )
```

## Best Practices

### Data Quality

1. **Filter aggressively** - Quality over quantity
   ```python
   # Remove low-quality messages
   messages_df = messages_df[
       (messages_df['body'].str.len() > 10) &
       (~messages_df['body'].str.match(r'^(ok|yes|no|yeah)$', case=False))
   ]
   ```

2. **Balance your dataset**
   ```python
   # Ensure balanced representation
   examples_per_thread = messages_df.groupby('thread_id').size()
   balanced_threads = examples_per_thread[
       examples_per_thread.between(50, 500)
   ].index
   ```

3. **Validate continuously**
   ```python
   # Validate during creation
   valid_examples = []
   for ex in examples:
       if len(ex['output']) > 5 and len(ex['input']) > 10:
           valid_examples.append(ex)
   ```

### Privacy Protection

1. **Anonymize before processing**
   ```python
   # Replace names and numbers
   messages_df['body'] = messages_df['body'].apply(anonymize_text)
   ```

2. **Filter sensitive content**
   ```python
   # Remove messages with sensitive keywords
   sensitive_keywords = ['password', 'ssn', 'credit card']
   mask = ~messages_df['body'].str.contains('|'.join(sensitive_keywords), case=False)
   messages_df = messages_df[mask]
   ```

### Performance

1. **Process incrementally**
   ```python
   # Save checkpoints
   for i in range(0, len(examples), 1000):
       checkpoint = examples[:i+1000]
       save_checkpoint(checkpoint, f'checkpoint_{i}.json')
   ```

2. **Monitor memory usage**
   ```python
   import psutil
   
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   ```

## Error Handling

### Common Errors and Solutions

```python
# FileNotFoundError
try:
    messages_df = pd.read_csv('signal.csv')
except FileNotFoundError:
    print("Signal CSV not found. Run extraction first.")
    sys.exit(1)

# KeyError - Missing columns
required_columns = ['_id', 'thread_id', 'from_recipient_id', 'body']
missing = set(required_columns) - set(messages_df.columns)
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# ValueError - Invalid recipient ID
if your_recipient_id not in messages_df['from_recipient_id'].unique():
    raise ValueError(f"Recipient ID {your_recipient_id} not found in messages")

# Memory Error
try:
    examples = creator.create_training_examples(messages_df)
except MemoryError:
    print("Out of memory. Try processing in chunks.")
    examples = process_in_chunks(messages_df, chunk_size=5000)
```

### Debug Mode

```python
# Enable detailed debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create with debug info
creator = TrainingDataCreator(debug=True)
examples, style = creator.create_training_examples(
    messages_df[:100],  # Small sample
    recipients_df
)

# Inspect intermediate results
print(f"Debug info: {creator.debug_info}")
```