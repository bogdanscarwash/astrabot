# Astrabot

A personal AI fine-tuning project that creates language models mimicking your communication style by analyzing Signal messenger conversation history.

## Overview

Astrabot uses your Signal messenger backup to create a personalized AI that communicates like you do. It analyzes your conversation patterns, preserves your unique communication style (including burst texting, emoji usage, and response patterns), and fine-tunes modern language models using Unsloth and Hugging Face transformers.

## Features

- **Signal Backup Processing**: Docker-based extraction of Signal backup files into analyzable data
- **Conversational AI Training**: Preserves natural dialogue flow rather than forcing Q&A format
- **Style Preservation**: Maintains your unique communication patterns (burst texting, message length, timing)
- **Twitter/X Integration**: Automatically extracts and includes tweet content and images in training data
- **Multi-Person Adaptation**: Learns how you adapt your communication style to different people
- **Privacy-Focused**: Handles blocked contacts appropriately and masks sensitive data
- **Comprehensive Development Tools**: Full Makefile automation for testing, code quality, and training

## Quick Start

### Prerequisites

- Python 3.8+
- Docker (for Signal backup processing)
- CUDA-capable GPU (recommended for training)
- Signal backup file (.backup)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astrabot.git
cd astrabot
```

2. Set up the development environment:
```bash
make setup-env
```

3. Install dependencies:
```bash
make install-dev
```

4. Configure environment variables:
```bash
# Edit .env with your API keys and settings
nano .env
```

## Development Workflow

### Available Commands

Run `make help` to see all available commands:

```bash
# Setup
make install         # Install production dependencies
make install-dev     # Install all dependencies including dev tools
make setup-env       # Set up development environment

# Testing
make test           # Run all tests
make test-unit      # Run only unit tests
make test-integration # Run integration tests (requires API keys)
make test-coverage  # Run tests with coverage report
make test-file      # Run specific test file interactively
make test-one       # Run specific test by name

# Code Quality
make lint           # Run flake8 linting
make format         # Format code with black
make type-check     # Run mypy type checking
make all            # Run format, lint, and type-check

# Docker & Data Processing
make docker-build   # Build Signal backup tools Docker image
make process-signal # Process Signal backup data

# Training
make train          # Run training pipeline

# Utilities
make clean          # Clean up generated files
make notebook       # Run Jupyter notebook server
make docs           # Generate documentation
make pre-commit     # Run pre-commit hooks
```

### Processing Signal Backup

1. Build the Signal backup tools Docker image:
```bash
make docker-build
```

2. Extract your Signal backup:
```bash
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools --input /backup/signal-backup.backup \
  --password "your-backup-password" --csv /output
```

3. Process the extracted data:
```bash
make process-signal
```

### Training Your Model

#### Option 1: Interactive Notebook
```bash
make notebook
# Navigate to notebooks/03_training_pipeline.ipynb
```

#### Option 2: Command Line
```bash
make train
```

The training pipeline will:
- Load and analyze your conversation data
- Create conversational training examples
- Fine-tune a Qwen model with your communication style
- Export the trained model

## Project Structure

```
astrabot/
├── src/                    # Source code modules
│   ├── core/              # Core processing logic
│   │   ├── conversation_processor.py  # Main conversation processing
│   │   ├── conversation_analyzer.py   # Pattern analysis
│   │   └── style_analyzer.py         # Communication style detection
│   ├── extractors/        # Twitter/media extraction
│   │   └── twitter_extractor.py      # Twitter/X content extraction
│   ├── llm/               # LLM training utilities
│   │   ├── training_data_creator.py  # Training example creation
│   │   ├── adaptive_trainer.py       # Style-adaptive training
│   │   └── prompts/qwen3-chat       # Chat template
│   ├── models/            # Data models and schemas
│   │   ├── schemas.py               # Pydantic models
│   │   └── conversation_schemas.py   # Conversation models
│   └── utils/             # Utilities (logging, config)
├── scripts/               # Command-line scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data
├── docker/                # Docker configurations
├── data/                  # Data directory (not in git)
│   ├── raw/              # Raw Signal data
│   ├── processed/        # Processed datasets
│   └── cache/            # API response cache
└── outputs/               # Generated models and reports
```

## Key Components

### Conversation Processing
- **ConversationProcessor**: Main class handling conversation transformation
  - Preserves natural dialogue flow and context
  - Handles burst texting and multi-message sequences
  - Extracts Twitter content and enriches messages

### Twitter Integration
- **TwitterExtractor**: Extracts tweet content and images from shared links
- Uses GPT-4o-mini for cost-effective image descriptions
- Supports multiple Nitter instances for privacy
- Caches responses to minimize API calls

### Training Pipeline
- **TrainingDataCreator**: Generates various training formats
  - Conversational data with natural dialogue
  - Adaptive data that adjusts to conversation partners
  - Burst sequence data preserving message sequences
- Uses Unsloth for efficient fine-tuning
- Supports multiple model sizes (3B to 14B parameters)
- LoRA-based fine-tuning for efficiency

## Configuration

Key environment variables (see `.env.example`):
- `OPENAI_API_KEY`: For GPT-4o-mini image descriptions
- `ANTHROPIC_API_KEY`: Alternative for Claude vision
- `YOUR_RECIPIENT_ID`: Your ID in Signal data (usually 2)
- `HF_TOKEN`: Hugging Face token for model uploads

## Testing

The project follows a strong TDD approach with comprehensive test coverage:

```bash
# Run all tests
make test

# Run specific test file
make test-file
# Enter: tests/unit/test_conversation_processor.py

# Run with coverage
make test-coverage

# Run specific test
make test-one
# Enter: test_extract_tweet_text
```

## Code Quality

Maintain high code quality with automated tools:

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all quality checks
make all
```

## Privacy & Security

- Sensitive data (API keys, phone numbers) are automatically masked in logs
- Blocked contacts are included with appropriate context
- All data stays local unless you explicitly upload models
- Consider the privacy implications before sharing trained models

## Contributing

This is a personal project, but if you'd like to contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `make all` to ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [signal-backup-decode](https://github.com/bepaald/signalbackup-tools) for Signal backup processing
- Hugging Face for transformer models and infrastructure