# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Astrabot is a personal AI fine-tuning project that creates language models mimicking your communication style by analyzing Signal messenger conversation history. It uses Unsloth and Hugging Face transformers to fine-tune Qwen3 models on personal chat data.

## Common Development Commands

### Testing
```bash
# Run all tests
just test

# Run specific test file
just test-file
# Then enter: tests/unit/test_conversation_processor.py

# Run tests with coverage
just test-coverage

# Run a specific test by name
just test-one
# Then enter: test_extract_tweet_text

# Run only unit tests
just test-unit

# Run integration tests (requires API keys in .env)
just test-integration
```

### Code Quality
```bash
# Format code with black
just format

# Run linting (flake8 + mypy)
just lint

# Type checking
just type-check

# Run all quality checks
just all

# Clean up cache and generated files
just clean
```

### Signal Backup Processing
```bash
# Extract Signal backup using Docker
cd docker/signalbackup-tools
docker build -t signalbackup-tools .
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools --input /backup/signal-backup.backup \
  --password "your-backup-password" --csv /output
```

### Training Pipeline
```bash
# Run the main training notebook
jupyter notebook notebooks/03_training_pipeline.ipynb

# Or use the command-line training script
python scripts/train.py --config configs/training_config.yaml

# Process Signal data into training format
python scripts/process_signal_data.py --input data/raw/signal-flatfiles --output data/processed
```

### Environment Setup
```bash
# Run the bootstrap script (now in scripts/setup/)
bash scripts/setup/bootstrap.sh

# This will:
# - Check Python 3.9+ requirement
# - Install uv package manager
# - Install all dependencies in .venv
# - Set up pre-commit hooks

# Set up environment variables
python scripts/setup/setup-secrets.py
# Or manually:
cp .env.example .env
# Edit .env with your API keys
```

## Architecture and Key Components

### Project Structure
```
src/
├── core/                    # Core processing logic
│   ├── conversation_processor.py  # Main conversation processing with Twitter extraction
│   ├── conversation_analyzer.py   # Pattern analysis
│   └── style_analyzer.py         # Communication style detection
├── extractors/             # Content extraction
│   └── twitter_extractor.py      # Twitter/X content and image extraction
├── llm/                    # LLM training utilities
│   ├── training_data_creator.py  # Creates training examples
│   ├── adaptive_trainer.py       # Style-adaptive training
│   └── prompts/qwen3-chat       # Jinja2 chat template
├── models/                 # Data models
│   ├── schemas.py               # Pydantic models (TweetContent, ImageDescription)
│   └── conversation_schemas.py   # Conversation-specific models
└── utils/                  # Utilities
    ├── config.py                # Environment-based configuration
    └── logging.py               # Security-aware logging with masking

scripts/
├── setup/                  # Setup and installation scripts
│   ├── bootstrap.sh            # Main setup script using uv
│   └── setup-secrets.py        # Interactive API key configuration
└── old-analysis/          # Archived analysis scripts
```

### Data Processing Flow
1. **Signal Backup** → CSV files in `data/raw/signal-flatfiles/`
2. **CSV Processing** → Conversation windows with context
3. **Enhancement** → Twitter content extraction, image descriptions
4. **Training Data** → Multiple formats (conversational, burst, adaptive)
5. **Fine-tuning** → Unsloth with LoRA on Qwen3 models

### Key Classes and Functions

#### Core Processing
- `ConversationProcessor`: Main class handling conversation transformation
  - `process_conversations()`: Converts Signal data to training format
  - `extract_twitter_content()`: Enriches messages with tweet data
- `StyleAnalyzer`: Detects personal communication patterns
  - `analyze_burst_patterns()`: Identifies multi-message sequences
  - `calculate_style_metrics()`: Message length, emoji usage, timing

#### Twitter Integration
- `TwitterExtractor`: Extracts content from Twitter/X URLs
  - Supports multiple Nitter instances for privacy
  - Falls back between instances on failure
  - Caches responses to minimize API calls
- Image description via OpenAI GPT-4o-mini or Anthropic Claude

#### Training Data Creation
- `TrainingDataCreator`: Generates various training formats
  - `create_conversational_data()`: Natural dialogue with context
  - `create_adaptive_data()`: Adjusts to conversation partners
  - `create_burst_sequence_data()`: Preserves message sequences

### Model Configuration
- **Base Model**: Qwen3-14B (default), supports 3B-14B variants
- **Quantization**: 4-bit with bitsandbytes
- **Fine-tuning**: LoRA with r=8, alpha=16
- **Max Length**: 4096 tokens
- **Template**: Custom Jinja2 template at `src/llm/prompts/qwen3-chat`

### Important Implementation Details

#### Security and Privacy
- Sensitive data (API keys, phone numbers) automatically masked in logs
- Blocked contacts handled with appropriate context flags
- API responses cached locally to minimize external calls
- All Signal data stays local unless explicitly uploaded

#### Testing Approach
- Strong TDD foundation - tests written before implementation
- Test structure mirrors source structure (`tests/unit/`, `tests/integration/`)
- Fixtures in `tests/fixtures/` for consistent test data
- Mock external API calls for unit tests

#### Configuration
- **Single dependency source**: `pyproject.toml` (no requirements.txt)
- **Package management**: `uv` for fast, reliable dependency resolution
- Environment variables via `.env` file (never commit!)
- Key variables:
  - `OPENAI_API_KEY`: GPT-4o-mini for images
  - `ANTHROPIC_API_KEY`: Alternative vision API
  - `YOUR_RECIPIENT_ID`: Your Signal ID (usually 2)
  - `HF_TOKEN`: For model uploads

#### Current Architecture Notes
- Migrated from monolithic notebooks to modular `src/` structure
- Command-line interface via `scripts/train.py`
- Documentation follows Diátaxis framework
- All setup consolidated to single bootstrap script
- Uses modern Python packaging (pyproject.toml + uv)
