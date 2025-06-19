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

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Processing Signal Backup

1. Build the Signal backup tools Docker image:
```bash
cd docker/signalbackup-tools
docker build -t signalbackup-tools .
```

2. Extract your Signal backup:
```bash
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools --input /backup/signal-backup.backup \
  --password "your-backup-password" --csv /output
```

### Training Your Model

1. Launch the training notebook:
```bash
jupyter notebook notebooks/03_training_pipeline.ipynb
```

2. Follow the notebook to:
   - Load and analyze your conversation data
   - Create conversational training examples
   - Fine-tune a Qwen model with your communication style
   - Export the trained model

## Project Structure

```
astrabot/
├── src/                    # Source code modules
│   ├── core/              # Core processing logic
│   ├── extractors/        # Twitter/media extraction
│   ├── llm/               # LLM training utilities
│   ├── models/            # Data models and schemas
│   └── utils/             # Utilities (logging, config)
├── notebooks/             # Jupyter notebooks
├── tests/                 # Test suite
├── docker/                # Docker configurations
├── data/                  # Data directory (not in git)
│   ├── raw/              # Raw Signal data
│   ├── processed/        # Processed datasets
│   └── cache/            # API response cache
└── outputs/               # Generated models and reports
```

## Key Components

### Conversation Processing
- `conversation_utilities.py`: Core functions for processing conversations
- Preserves natural dialogue flow and context
- Handles burst texting and multi-message sequences

### Twitter Integration
- Extracts tweet content and images from shared links
- Uses GPT-4o-mini for cost-effective image descriptions
- Enriches training data with social media context

### Training Pipeline
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

Run the test suite:
```bash
make test
```

Run specific tests:
```bash
make test-file
# Enter: test_conversation_utilities.py
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
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- [signal-backup-decode](https://github.com/bepaald/signalbackup-tools) for Signal backup processing
- Hugging Face for transformer models and infrastructure