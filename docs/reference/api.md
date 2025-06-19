# API Reference

This document provides a comprehensive reference for all Astrabot modules and functions.

## Core Modules

### `src.core.conversation_processor`

Main module for processing Signal conversations into training data.

#### Functions

##### `create_conversation_windows(messages_df, window_size=5, your_recipient_id=2)`

Create sliding windows of conversation context.

**Parameters:**
- `messages_df` (DataFrame): Signal messages dataframe
- `window_size` (int): Number of messages to include for context
- `your_recipient_id` (int): Your recipient ID in the database

**Returns:**
- List[Dict]: Conversation windows with metadata

**Example:**
```python
windows = create_conversation_windows(messages_df, window_size=5)
```

##### `segment_natural_dialogues(messages_df, time_gap_minutes=30, your_recipient_id=2)`

Segment conversations into natural dialogue episodes based on time gaps.

**Parameters:**
- `messages_df` (DataFrame): Signal messages dataframe
- `time_gap_minutes` (int): Minutes of inactivity to consider new episode
- `your_recipient_id` (int): Your recipient ID

**Returns:**
- List[Dict]: Conversation episodes with complete dialogue arcs

##### `preserve_conversation_dynamics(messages_df, your_recipient_id=2)`

Capture and preserve conversation modes and dynamics including burst texting.

**Parameters:**
- `messages_df` (DataFrame): Signal messages dataframe
- `your_recipient_id` (int): Your recipient ID

**Returns:**
- List[Dict]: Conversation data with preserved dynamics

## Extractor Modules

### `src.extractors.twitter_extractor`

Extract and process Twitter/X content from messages.

#### Functions

##### `extract_tweet_text(url, return_structured=False)`

Extract tweet content from a Twitter/X URL.

**Parameters:**
- `url` (str): Twitter/X URL
- `return_structured` (bool): Return TweetContent object if True

**Returns:**
- Dict or TweetContent: Tweet data

**Example:**
```python
tweet_data = extract_tweet_text("https://twitter.com/user/status/123")
```

##### `extract_tweet_images(url)`

Extract image URLs from a tweet.

**Parameters:**
- `url` (str): Twitter/X URL

**Returns:**
- List[str]: Image URLs

##### `describe_tweet_images(image_urls, api='openai', batch_process=True)`

Describe images using vision AI.

**Parameters:**
- `image_urls` (List[str]): URLs of images to describe
- `api` (str): 'openai' or 'anthropic'
- `batch_process` (bool): Process in batch for efficiency

**Returns:**
- List[str]: Image descriptions

##### `process_message_with_twitter_content(message, use_images=True, image_api='openai')`

Process a message and enhance with Twitter content.

**Parameters:**
- `message` (str): Message text potentially containing Twitter URLs
- `use_images` (bool): Whether to process images
- `image_api` (str): Which API to use for image description

**Returns:**
- str: Enhanced message with injected content

## Model Schemas

### `src.models.conversation_schemas`

Pydantic models for structured data.

#### Classes

##### `TweetContent`

Structured representation of tweet data.

**Attributes:**
- `text` (str): Tweet text content
- `author` (str): Tweet author username
- `tweet_id` (str): Tweet ID
- `mentioned_users` (List[str]): Mentioned usernames
- `hashtags` (List[str]): Hashtags in tweet
- `urls` (List[str]): URLs in tweet
- `sentiment` (Sentiment): Detected sentiment
- `is_reply` (bool): Whether tweet is a reply
- `is_retweet` (bool): Whether tweet is a retweet

**Methods:**
- `to_training_format()`: Convert to training-friendly format

##### `ImageDescription`

Structured image description.

**Attributes:**
- `description` (str): Main description
- `main_subjects` (List[str]): Key subjects/objects
- `detected_text` (Optional[str]): Any text in image
- `emotional_tone` (Optional[str]): Emotional context
- `colors` (List[str]): Dominant colors
- `setting` (Optional[str]): Scene setting

##### `EnhancedMessage`

Complete message with all extracted content.

**Attributes:**
- `original_message` (str): Original message text
- `conversation_id` (str): Conversation thread ID
- `message_id` (str): Unique message ID
- `sender_id` (str): Sender's ID
- `timestamp` (datetime): Message timestamp
- `tweet_contents` (List[TweetContent]): Extracted tweets
- `image_descriptions` (List[ImageDescription]): Described images

## Utility Modules

### `src.utils.config`

Configuration management.

#### Class `Config`

Centralized configuration loaded from environment variables.

**Attributes:**
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `MODEL_NAME`: Base model name
- `MAX_EPOCHS`: Training epochs
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Learning rate
- `DATA_PATH`: Data directory path
- `OUTPUT_PATH`: Output directory path

**Usage:**
```python
from src.utils.config import config
print(config.MODEL_NAME)
```

### `src.utils.logging`

Structured logging with privacy features.

#### Functions

##### `get_logger(name)`

Get a configured logger instance.

**Parameters:**
- `name` (str): Logger name (usually `__name__`)

**Returns:**
- `AstrabotLogger`: Configured logger

**Example:**
```python
from src.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing messages")
```

##### `log_performance(func)`

Decorator to log function performance.

**Example:**
```python
@log_performance
def process_data():
    # Function implementation
    pass
```

##### `log_api_call(service, endpoint, response_time)`

Log API call metrics.

**Parameters:**
- `service` (str): API service name
- `endpoint` (str): API endpoint
- `response_time` (float): Response time in seconds

## LLM Modules

### `src.llm.training_data_creator`

Create training data in various formats.

#### Class `TrainingDataCreator`

Main class for creating training datasets.

**Methods:**

##### `create_conversational_training_data(messages_df, recipients_df)`

Create conversational training data preserving natural flow.

**Returns:**
- List[Dict]: Training examples with conversation context

##### `create_adaptive_training_data(messages_df, recipients_df)`

Create training data that captures style adaptation.

**Returns:**
- List[Dict]: Training examples with style metadata

##### `format_for_training(training_data, tokenizer)`

Format training data for model consumption.

**Parameters:**
- `training_data` (List[Dict]): Raw training examples
- `tokenizer`: Model tokenizer

**Returns:**
- Dataset: Formatted dataset ready for training

## Script Interfaces

### `scripts/train.py`

Command-line training interface.

**Arguments:**
- `--data-path`: Path to Signal flatfiles
- `--output-path`: Output directory
- `--model-name`: Base model name
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--training-mode`: 'conversational', 'qa', or 'adaptive'
- `--use-twitter-enhancement`: Enable Twitter content extraction

### `scripts/evaluate.py`

Model evaluation interface.

**Arguments:**
- `--model-path`: Path to trained model
- `--test-data`: Test data file
- `--eval-mode`: 'perplexity', 'style', 'interactive', or 'all'
- `--output-file`: Results output file

### `scripts/export.py`

Model export interface.

**Arguments:**
- `--model-path`: Path to trained model
- `--export-format`: 'hf', 'gguf', 'onnx', 'merged', or 'all'
- `--output-path`: Export directory
- `--quantization`: Quantization method for GGUF
- `--push-to-hub`: Push to Hugging Face Hub
- `--hub-repo`: Hub repository name

## Testing Utilities

### Test Fixtures

Located in `tests/fixtures/`:
- Sample messages
- Mock API responses
- Test conversations

### Test Helpers

Common test utilities in `tests/conftest.py`:
- Mock Signal data generators
- API response mocks
- Temporary file handlers