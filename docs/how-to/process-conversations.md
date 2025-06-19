# How to Process Conversations

This guide explains how to transform your Signal conversation data into high-quality training data for personalizing language models. The conversation processing pipeline preserves your natural communication style while enriching messages with external content.

## Overview

Conversation processing in Astrabot:
1. **Extracts natural dialogue flows** from your Signal messages
2. **Preserves communication patterns** like message bursts and timing
3. **Enriches content** by extracting Twitter/X posts and describing images
4. **Analyzes your style** to capture unique communication traits
5. **Creates training examples** that teach models your texting patterns

## Prerequisites

Before starting, ensure you have:
- Extracted Signal data in CSV format (see [Process Signal Backup](process-signal-backup.md))
- Python environment set up with Astrabot installed
- (Optional) API keys for content enhancement:
  - OpenAI API key for image descriptions
  - Or Anthropic API key as alternative

## Basic Processing with Command Line

### Step 1: Verify Your Data

First, check that your Signal data was extracted correctly:

```bash
# List extracted files
ls -la data/raw/signal-flatfiles/

# Check message count
wc -l data/raw/signal-flatfiles/signal.csv

# Find your recipient ID (usually 2)
python scripts/find-my-id.py
```

### Step 2: Run Basic Processing

Process conversations with default settings:

```bash
python scripts/process_signal_data.py \
    --input data/raw/signal-flatfiles \
    --output data/processed/training_data.json
```

This will:
- Load your messages and contacts
- Create conversation windows with context
- Analyze your communication style
- Generate training examples
- Save results to JSON

### Step 3: Monitor Progress

Watch the output for:
```
✅ Processing complete!
Created 5432 training examples
Output saved to: data/processed/training_data.json

📊 Your Communication Style:
  Average message length: 45.2 chars
  Preferred style: concise
  Burst frequency: 34.00%
  Emoji usage: 23.0%
```

## Advanced Processing with Python API

### Basic Usage

```python
from src.core.conversation_processor import ConversationProcessor
from src.llm.training_data_creator import TrainingDataCreator
import pandas as pd

# Load your data
messages_df = pd.read_csv('data/raw/signal-flatfiles/signal.csv')
recipients_df = pd.read_csv('data/raw/signal-flatfiles/recipient.csv')

# Initialize processor
processor = ConversationProcessor(
    your_recipient_id=2,
    enhance_with_twitter=True,
    include_images=True
)

# Process conversations
enhanced_messages = processor.process_conversations(
    messages_df, 
    recipients_df
)

# Create training data
creator = TrainingDataCreator()
training_data = creator.create_conversational_data(
    enhanced_messages,
    max_examples=10000
)

print(f"Created {len(training_data)} training examples")
```

### Filtering Conversations

Process only specific conversations:

```python
# Filter by contact
contact_name = "Alice"
contact_id = recipients_df[
    recipients_df['profile_given_name'] == contact_name
]['_id'].iloc[0]

filtered_messages = messages_df[
    (messages_df['from_recipient_id'] == contact_id) |
    (messages_df['to_recipient_id'] == contact_id)
]

# Filter by date range
from datetime import datetime, timedelta

cutoff_date = datetime.now() - timedelta(days=365)
cutoff_timestamp = int(cutoff_date.timestamp() * 1000)

recent_messages = messages_df[
    messages_df['date_sent'] > cutoff_timestamp
]

# Filter by minimum conversation length
thread_counts = messages_df.groupby('thread_id').size()
active_threads = thread_counts[thread_counts >= 50].index
active_messages = messages_df[messages_df['thread_id'].isin(active_threads)]
```

### Custom Enhancement Options

```python
# Configure enhancement settings
processor = ConversationProcessor(
    your_recipient_id=2,
    enhance_with_twitter=True,
    include_images=True,
    twitter_config={
        'batch_size': 10,
        'timeout': 30,
        'nitter_instances': [
            'nitter.net',
            'nitter.it',
            'nitter.42l.fr'
        ]
    }
)

# Process with custom style analysis
from src.core.style_analyzer import StyleAnalyzer

analyzer = StyleAnalyzer()
style_analysis = analyzer.analyze_communication_patterns(
    messages_df,
    your_recipient_id=2,
    min_messages_threshold=100  # Require more messages for analysis
)

# Apply style-based filtering
if style_analysis['preferred_length'] == 'lengthy':
    # Keep longer messages for lengthy texters
    messages_df = messages_df[messages_df['body'].str.len() > 50]
```

## Understanding the Output

### Training Data Format

Each training example contains:

```json
{
  "instruction": "Continue this conversation naturally",
  "input": "Friend: Hey, did you see that tweet about AI?\nYou: Which one?",
  "output": "The one about GPT-4 being multimodal now",
  "metadata": {
    "type": "conversation_window",
    "momentum": "rapid",
    "response_delay": 45.2,
    "context_size": 2,
    "has_enhanced_content": true
  }
}
```

### Enhanced Message Format

When Twitter content is found:

```json
{
  "original": "Check out this tweet: https://twitter.com/user/status/123",
  "enhanced": "Check out this tweet: https://twitter.com/user/status/123\n\n[TWEET: @user]\nThis is the actual tweet content with preserved formatting\n[/TWEET]",
  "metadata": {
    "has_twitter": true,
    "tweet_count": 1,
    "has_images": false
  }
}
```

### Style Analysis Output

```json
{
  "avg_message_length": 45.2,
  "preferred_length": "concise",
  "burst_patterns": {
    "total_bursts": 234,
    "avg_burst_size": 3.2,
    "burst_frequency": 0.34
  },
  "emoji_usage": {
    "emoji_frequency": 0.23,
    "top_emojis": ["😂", "👍", "❤️"]
  },
  "conversation_roles": {
    "balanced_conversationalist": 45,
    "conversation_driver": 12,
    "responsive_participant": 23
  }
}
```

## Customization Options

### Command Line Arguments

```bash
python scripts/process_signal_data.py \
    --input data/raw/signal-flatfiles \
    --output data/processed/training_data.json \
    --recipient-id 2 \
    --max-examples 5000 \
    --min-messages 10 \
    --no-twitter \
    --exclude-blocked \
    --date-from "2023-01-01" \
    --date-to "2024-01-01" \
    --verbose
```

| Option | Description | Default |
|--------|-------------|---------|
| `--recipient-id` | Your Signal recipient ID | 2 |
| `--max-examples` | Limit training examples | None |
| `--min-messages` | Minimum messages per conversation | 10 |
| `--no-twitter` | Disable Twitter enhancement | False |
| `--exclude-blocked` | Skip blocked contacts | False |
| `--date-from` | Start date (YYYY-MM-DD) | None |
| `--date-to` | End date (YYYY-MM-DD) | None |
| `--window-size` | Conversation context window | 5 |
| `--episode-gap` | Minutes for episode breaks | 30 |

### Python API Parameters

```python
# Conversation window creation
windows = creator.create_conversation_windows(
    messages_df,
    window_size=7,  # More context
    min_window_messages=3,  # Require at least 3 messages
    include_timestamps=True
)

# Episode segmentation
episodes = creator.segment_natural_dialogues(
    messages_df,
    time_gap_minutes=60,  # Longer gaps for episodes
    min_episode_length=5,  # Minimum messages per episode
    max_episode_length=50  # Cap episode size
)

# Style-based processing
training_data = creator.create_adaptive_training_data(
    messages_df,
    recipients_df,
    adaptation_threshold=0.7,  # How different styles must be
    min_examples_per_style=50  # Ensure style coverage
)
```

## Performance Optimization

### Processing Large Datasets

For datasets with 100k+ messages:

```python
# Process in chunks
chunk_size = 10000
chunks = []

for i in range(0, len(messages_df), chunk_size):
    chunk = messages_df.iloc[i:i+chunk_size]
    processed = processor.process_conversations(chunk, recipients_df)
    chunks.append(processed)

# Combine results
all_messages = pd.concat(chunks)
```

### Memory Management

```python
# Use generators for large datasets
def process_conversations_generator(messages_df, batch_size=1000):
    for i in range(0, len(messages_df), batch_size):
        batch = messages_df.iloc[i:i+batch_size]
        yield processor.process_batch(batch)

# Process with limited memory
training_examples = []
for batch_results in process_conversations_generator(messages_df):
    training_examples.extend(batch_results)
    
    # Save periodically
    if len(training_examples) >= 5000:
        save_checkpoint(training_examples)
        training_examples = []
```

### Parallel Processing

```python
from multiprocessing import Pool
import numpy as np

# Split by thread for parallel processing
thread_groups = messages_df.groupby('thread_id')

def process_thread(thread_data):
    thread_id, messages = thread_data
    return processor.process_single_thread(messages)

# Process threads in parallel
with Pool(processes=4) as pool:
    results = pool.map(process_thread, thread_groups)
```

## Troubleshooting

### Common Issues

**Issue**: "No messages found for recipient"
```python
# Debug recipient IDs
print("Unique from_recipient_ids:", messages_df['from_recipient_id'].unique()[:10])
print("Unique to_recipient_ids:", messages_df['to_recipient_id'].unique()[:10])

# Find your messages
your_messages = messages_df[
    (messages_df['from_recipient_id'] == 2) | 
    (messages_df['to_recipient_id'] == 2)
]
print(f"Found {len(your_messages)} messages for recipient 2")
```

**Issue**: "Twitter extraction failing"
```python
# Test Twitter extraction directly
from src.extractors.twitter_extractor import TwitterExtractor

extractor = TwitterExtractor()
test_url = "https://twitter.com/user/status/123"

try:
    content = extractor.extract_tweet_content(test_url)
    print("Extraction successful:", content)
except Exception as e:
    print("Extraction failed:", str(e))
    # Try different Nitter instance
    extractor.nitter_base_url = "https://nitter.it"
```

**Issue**: "Out of memory during processing"
```python
# Reduce memory usage
import gc

# Process in smaller chunks
chunk_size = 5000
for chunk in pd.read_csv('signal.csv', chunksize=chunk_size):
    process_chunk(chunk)
    gc.collect()  # Force garbage collection

# Use data types efficiently
messages_df = pd.read_csv('signal.csv', dtype={
    '_id': 'int32',
    'thread_id': 'int32',
    'from_recipient_id': 'int16',
    'to_recipient_id': 'int16',
    'body': 'string'
})
```

### Debug Mode

Enable detailed logging:

```python
import logging
from src.utils import get_logger

# Set debug level
logger = get_logger()
logger.setLevel(logging.DEBUG)

# Add console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Now processing will show detailed logs
processor.process_conversations(messages_df, recipients_df)
```

## Best Practices

### Data Quality

1. **Clean your data first**:
   ```python
   # Remove empty messages
   messages_df = messages_df[messages_df['body'].notna()]
   messages_df = messages_df[messages_df['body'].str.len() > 5]
   ```

2. **Handle system messages**:
   ```python
   # Filter out system messages
   messages_df = messages_df[
       ~messages_df['from_recipient_id'].isin([1, 3])
   ]
   ```

3. **Validate timestamps**:
   ```python
   # Ensure chronological order
   messages_df = messages_df.sort_values('date_sent')
   
   # Remove future timestamps
   current_time = int(datetime.now().timestamp() * 1000)
   messages_df = messages_df[messages_df['date_sent'] <= current_time]
   ```

### Privacy Preservation

1. **Anonymize phone numbers**:
   ```python
   # Replace phone numbers in message content
   import re
   phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
   messages_df['body'] = messages_df['body'].str.replace(
       phone_pattern, '[PHONE]', regex=True
   )
   ```

2. **Remove personally identifiable information**:
   ```python
   # List of names to redact
   names_to_redact = ['John Doe', 'Jane Smith']
   for name in names_to_redact:
       messages_df['body'] = messages_df['body'].str.replace(
           name, '[NAME]', case=False
       )
   ```

### Output Validation

Always validate your training data:

```python
# Load and validate
import json

with open('training_data.json', 'r') as f:
    data = json.load(f)

# Check structure
assert all('instruction' in ex for ex in data)
assert all('input' in ex for ex in data)
assert all('output' in ex for ex in data)

# Check quality
print(f"Total examples: {len(data)}")
print(f"Average input length: {np.mean([len(ex['input']) for ex in data]):.1f}")
print(f"Average output length: {np.mean([len(ex['output']) for ex in data]):.1f}")

# Sample examples
for i in range(min(5, len(data))):
    print(f"\nExample {i+1}:")
    print(f"Instruction: {data[i]['instruction']}")
    print(f"Input: {data[i]['input'][:100]}...")
    print(f"Output: {data[i]['output'][:100]}...")
```

## Next Steps

After processing your conversations:

1. **Review the output** - Check training data quality
2. **Fine-tune your model** - See [Training Guide](../tutorials/first-training-run.md)
3. **Iterate on quality** - Adjust parameters based on results
4. **Consider privacy** - Review [Privacy Best Practices](../explanation/privacy-architecture.md)

## See Also

- [Signal Data Schema Reference](../reference/signal-data-schema.md)
- [TrainingDataCreator API](../reference/api/training-data-creator.md)
- [Architecture Overview](../explanation/architecture.md)                                                       │