# Utilities Reference

This document provides comprehensive reference documentation for all utility modules in the Astrabot project. These utilities provide essential functionality for configuration management, logging, and common operations throughout the codebase.

## Table of Contents
- [Configuration (config.py)](#configuration-configpy)
- [Logging (logging.py)](#logging-loggingpy)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Configuration (config.py)

The configuration module provides centralized management of all application settings, environment variables, and feature flags.

### Class: Config

Central configuration management with secure environment variable handling.

#### Methods

##### `get(key: str, default: Optional[Any] = None) -> Optional[Any]`
Get a configuration value with optional default.
```python
# Example
api_key = config.get('OPENAI_API_KEY', 'default-key')
```

##### `require(key: str) -> Any`
Get a required configuration value. Raises `ValueError` if not found.
```python
# Example
recipient_id = config.require('YOUR_RECIPIENT_ID')
```

##### `has_openai() -> bool`
Check if OpenAI API is configured.
```python
if config.has_openai():
    # Use OpenAI for image descriptions
```

##### `has_anthropic() -> bool`
Check if Anthropic API is configured.
```python
if config.has_anthropic():
    # Use Claude for image descriptions
```

##### `validate() -> None`
Validate configuration and create required directories.
- Creates DATA_DIR and OUTPUT_DIR if they don't exist
- Logs configuration status
- Called automatically on module import

##### `print_status() -> None`
Display configuration status with sensitive values masked.
```python
config.print_status()
# Output:
# Configuration Status:
# ✓ OPENAI_API_KEY: sk-...ged
# ✗ ANTHROPIC_API_KEY: Not configured
```

#### Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **API Keys** |
| OPENAI_API_KEY | str | None | OpenAI API key for GPT-4 vision |
| ANTHROPIC_API_KEY | str | None | Anthropic API key for Claude |
| HF_TOKEN | str | None | Hugging Face token for model uploads |
| **Model Settings** |
| ANTHROPIC_MODEL | str | claude-3-haiku-20240307 | Anthropic model to use |
| OPENAI_MODEL | str | gpt-4o-mini | OpenAI model to use |
| **Signal Data** |
| YOUR_RECIPIENT_ID | int | 2 | Your Signal recipient ID |
| SIGNAL_BACKUP_PATH | str | None | Path to Signal backup file |
| SIGNAL_PASSWORD | str | None | Signal backup password |
| **Paths** |
| DATA_DIR | str | ./data | Data directory path |
| OUTPUT_DIR | str | ./output | Output directory path |
| CACHE_DIR | str | ./data/cache | Cache directory path |
| LOG_DIR | str | ./data/logs | Log directory path |
| **Features** |
| ENABLE_IMAGE_PROCESSING | bool | True | Enable image description |
| ENABLE_BATCH_PROCESSING | bool | True | Enable batch API calls |
| MAX_BATCH_SIZE | int | 10 | Maximum batch size |
| **Development** |
| DEBUG | bool | False | Debug mode flag |
| LOG_LEVEL | str | INFO | Logging level |
| **Rate Limiting** |
| API_RATE_LIMIT | int | 60 | API calls per window |
| API_RATE_WINDOW | int | 60 | Rate limit window (seconds) |

### Module Instance

The module provides a pre-configured singleton instance:
```python
from src.utils import config
```

## Logging (logging.py)

Advanced logging system with security features, performance tracking, and structured logging support.

### Functions

#### `get_logger(name: Optional[str] = None) -> AstrabotLogger`
Get or create a logger instance (singleton pattern).
```python
logger = get_logger(__name__)
logger.info("Processing started")
```

#### `setup_logging(level: str = "INFO", log_file: Optional[str] = None, json_format: bool = False) -> None`
Configure the logging system.
```python
setup_logging(
    level="DEBUG",
    log_file="app.log",
    json_format=True  # Enable structured JSON logging
)
```

Parameters:
- `level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_file`: Optional log file path (enables file logging with rotation)
- `json_format`: Enable JSON structured logging

#### `mask_sensitive_data(text: str) -> str`
Mask sensitive data in text using regex patterns.
```python
masked = mask_sensitive_data("API key: sk-1234567890abcdef")
# Returns: "API key: sk-...masked"
```

Automatically masks:
- OpenAI API keys (sk-...)
- Google API keys (AIza...)
- Bearer tokens
- Password fields
- Generic "token" or "key" patterns

### Decorators

#### `@log_performance(operation_name: Optional[str] = None)`
Log function execution time and performance metrics.
```python
@log_performance("data_processing")
def process_data(df):
    # Function implementation
    return result
```

Logs:
- Start time
- End time
- Duration
- Success/failure status
- Exception details (if any)

#### `@log_api_call(api_name: str)`
Log API calls with automatic timing and error tracking.
```python
@log_api_call("openai")
def call_gpt4_vision(image_url):
    # API call implementation
    return response
```

Logs:
- API name
- Call timing
- Success/failure
- Response status
- Error details

### Context Managers

#### `log_data_processing(operation: str, record_count: Optional[int] = None)`
Track data processing operations with context.
```python
with log_data_processing("conversation_extraction", record_count=1000):
    # Process conversations
    processed_data = transform_messages(data)
```

Logs:
- Operation start/end
- Record count
- Processing duration
- Success/failure status

### Classes

#### Class: AstrabotLogger

Enhanced logger with context management capabilities.

##### Methods

###### `add_context(**kwargs) -> ContextManager`
Add temporary context to logs within a scope.
```python
with logger.add_context(user_id=123, session="abc"):
    logger.info("Processing user request")
    # Logs include user_id and session in extra fields
```

###### `set_context(**kwargs) -> None`
Set persistent context for all subsequent logs.
```python
logger.set_context(environment="production", version="1.0.0")
```

###### `clear_context() -> None`
Clear all persistent context.
```python
logger.clear_context()
```

#### Class: SensitiveDataFilter

Logging filter that masks sensitive data patterns.

Features:
- Regex-based pattern matching
- Configurable masking patterns
- Thread-safe operation
- Automatic application to all log records

#### Class: JSONFormatter

Structured JSON logging formatter for machine-readable logs.

Output format:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "src.core.processor",
  "message": "Processing completed",
  "extra": {
    "duration": 1.234,
    "record_count": 1000
  }
}
```

## Usage Examples

### Basic Configuration Usage
```python
from src.utils import config

# Get configuration values
api_key = config.get('OPENAI_API_KEY')
debug_mode = config.get('DEBUG', False)

# Require configuration values
recipient_id = config.require('YOUR_RECIPIENT_ID')

# Check API availability
if config.has_openai():
    model = config.get('OPENAI_MODEL', 'gpt-4o-mini')
```

### Basic Logging Usage
```python
from src.utils import get_logger

logger = get_logger(__name__)

# Simple logging
logger.info("Starting processing")
logger.warning("API rate limit approaching")
logger.error("Failed to process", exc_info=True)

# With extra context
logger.info("Message processed", extra={
    "message_id": 12345,
    "thread_id": 67890
})
```

### Advanced Logging with Context
```python
from src.utils import get_logger, log_performance, log_data_processing

logger = get_logger(__name__)

@log_performance("conversation_processing")
def process_conversations(data):
    with logger.add_context(stage="preprocessing"):
        logger.info("Starting preprocessing")
        # Preprocessing logic
    
    with log_data_processing("transformation", len(data)):
        # Transform data
        return transformed_data
```

### Secure API Integration
```python
from src.utils import config, get_logger, log_api_call

logger = get_logger(__name__)

@log_api_call("openai")
def describe_image(image_url):
    if not config.has_openai():
        raise ValueError("OpenAI not configured")
    
    api_key = config.require('OPENAI_API_KEY')
    # API call implementation
```

## Best Practices

### Configuration
1. **Never commit .env files** - Use .env.example as template
2. **Use require() for critical values** - Fail fast if configuration is missing
3. **Check API availability** - Use has_openai() before API calls
4. **Validate early** - Call config.validate() in main entry points

### Logging
1. **Use appropriate log levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: General informational messages
   - WARNING: Warning messages for recoverable issues
   - ERROR: Error messages for failures
   - CRITICAL: Critical failures requiring immediate attention

2. **Include context in logs**:
   ```python
   logger.info("Processing message", extra={
       "message_id": msg_id,
       "thread_id": thread_id
   })
   ```

3. **Use decorators for consistent metrics**:
   ```python
   @log_performance()
   @log_api_call("external_api")
   def make_api_request():
       pass
   ```

4. **Never log sensitive data directly**:
   ```python
   # Bad
   logger.info(f"Using API key: {api_key}")
   
   # Good
   logger.info("API configured successfully")
   ```

5. **Use structured logging for production**:
   ```python
   setup_logging(json_format=True)  # Machine-readable logs
   ```

### Security
1. **Sensitive data is automatically masked** in logs
2. **Use mask_sensitive_data()** for user-facing output
3. **Store secrets in environment variables** only
4. **Never log full API responses** without sanitization

### Performance
1. **Use log_performance decorator** for timing-critical functions
2. **Track data processing** with record counts
3. **Monitor API calls** with log_api_call decorator
4. **Use batch processing** when configured

## See Also
- [Environment Variables Guide](environment-variables.md)
- [Security Best Practices](../explanation/security.md)
- [Testing Documentation](../../tests/README.md)