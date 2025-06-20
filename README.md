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

- Python 3.9+ (required for dependencies)
- Docker (for Signal backup processing)
- CUDA-capable GPU (recommended for training)
- Signal backup file (.backup)
- `just` task runner (optional but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astrabot.git
cd astrabot
```

2. **Run the bootstrap script**:
```bash
bash scripts/setup/bootstrap.sh
```

This bootstrap script will:
- Check Python version (3.9+ required)
- Install `uv` package manager if not present
- Create and configure a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Verify the installation

3. **Configure environment variables**:
```bash
# Copy example and edit with your API keys
cp .env.example .env
nano .env
```

4. **Install just task runner** (optional but recommended):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
```

That's it! The project uses modern Python packaging with `pyproject.toml` and the `uv` package manager for fast, reliable dependency management.

## Development Workflow

### Available Commands

**With just task runner** (recommended - run `just help` to see all commands):

```bash
# Setup
just install         # Install production dependencies
just install-dev     # Install all dependencies including dev tools

# Testing
just test           # Run all tests
just test-unit      # Run only unit tests
just test-integration # Run integration tests (requires API keys)
just test-coverage  # Run tests with coverage report
just test-file      # Run specific test file interactively
just test-one       # Run specific test by name
just test-quick     # Quick test run without coverage
just test-watch     # Watch tests and rerun on changes

# Code Quality
just lint           # Run flake8 linting
just format         # Format code with black and isort
just type-check     # Run mypy type checking
just all            # Run format, lint, and type-check
just pre-commit     # Run pre-commit hooks

# UV Package Management
just uv-sync        # Sync dependencies with uv
just uv-add PKG     # Add a new dependency
just uv-remove PKG  # Remove a dependency
just uv-update      # Update all dependencies

# Docker & Data Processing
just docker-build   # Build Signal backup tools Docker image
just process-signal # Process Signal backup data

# Training
just train          # Run training pipeline
just train-qwen3    # Train Qwen3 model with full pipeline
just train-qwen3-small  # Train small Qwen3 model for testing

# Utilities
just clean          # Clean up generated files
just notebook       # Run Jupyter notebook server
just docs           # Generate documentation
```

**Without just** (direct commands):

```bash
# Activate virtual environment first
source .venv/bin/activate

# Then use standard tools
pytest              # Run tests
black src/ tests/   # Format code
flake8 src/ tests/  # Lint code
mypy src/           # Type check
uv sync             # Sync dependencies
```

### Processing Signal Backup

1. Build the Signal backup tools Docker image:
```bash
just docker-build
```

2. Extract your Signal backup:
```bash
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools --input /backup/signal-backup.backup \
  --password "your-backup-password" --csv /output
```

3. Process the extracted data:
```bash
just process-signal
```

### Training Your Model

#### Option 1: Interactive Notebook
```bash
just notebook
# Navigate to notebooks/03_training_pipeline.ipynb
```

#### Option 2: Command Line
```bash
# Basic training
just train

# Qwen3 model training options
just train-qwen3        # Full training pipeline
just train-qwen3-small  # Small model for testing
just train-qwen3-test   # Test training with debug output
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
│   ├── setup/            # Setup and installation scripts
│   │   ├── bootstrap.sh           # Main setup script
│   │   └── setup-secrets.py       # API key configuration
│   └── old-analysis/     # Legacy analysis scripts
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
just test

# Run specific test file
just test-file
# Enter: tests/unit/test_conversation_processor.py

# Run with coverage
just test-coverage

# Run specific test
just test-one
# Enter: test_extract_tweet_text

# Quick testing (no coverage)
just test-quick

# Watch mode (reruns tests on file changes)
just test-watch
```

## Code Quality

Maintain high code quality with automated tools:

```bash
# Format code (black + isort)
just format

# Run linting (flake8)
just lint

# Type checking (mypy)
just type-check

# Run all quality checks
just all

# Run pre-commit hooks
just pre-commit
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
3. Run `bash scripts/setup/bootstrap.sh` to set up your environment
4. Make your changes with tests
5. Run `just all` to ensure code quality
6. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [signal-backup-decode](https://github.com/bepaald/signalbackup-tools) for Signal backup processing
- Hugging Face for transformer models and infrastructure
