# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Astrabot is a personal AI fine-tuning project that creates a language model mimicking your communication style by analyzing Signal messenger conversation history. It uses Unsloth and Hugging Face transformers to fine-tune Qwen3 models on personal chat data.

## Common Development Commands

### Signal Backup Processing
```bash
# Process Signal backup files using Docker
cd cntn-signalbackup-tools
docker build -t signalbackup-tools .
docker run -v /path/to/backup:/backup signalbackup-tools
```

### Running the Main Notebook
```bash
# Launch Jupyter notebook
jupyter notebook notebook.ipynb
```

### Testing Conversation Utilities
```bash
# Test Twitter extraction and enhancement functions
python test_conversation_utilities.py
```

### Dependencies Installation
```bash
# Install all required packages
pip install -r requirements.txt
```

## Architecture and Key Components

### Data Flow
1. **Signal Backup Extraction**: `cntn-signalbackup-tools/` Docker container processes Signal backup files into CSV flatfiles
2. **Data Processing**: `notebook.ipynb` reads CSVs from `signal-flatfiles/` and transforms them into training data
3. **Style Analysis**: Analyzes personal communication patterns (burst texting, message length, emoji usage)
4. **Training Data Creation**: Multiple transformation strategies:
   - Conversation format with context
   - Question-answer pair extraction
   - Persona-based training with style preservation
   - Adaptive training based on conversation partner
5. **Model Fine-tuning**: Uses Unsloth with LoRA for efficient fine-tuning of Qwen3 models

### Key Processing Functions
- `transform_to_conversations()`: Converts messages to conversation format
- `analyze_personal_texting_style()`: Detects communication patterns
- `create_adaptive_training_data()`: Creates style-adaptive training examples
- `extract_twitter_links()`: Processes shared Twitter/X links for context
- `extract_qa_pairs_enhanced()`: Improved Q&A extraction with better patterns and filtering
- `extract_tweet_text()`: Reliably extracts tweet text from URLs
- `extract_tweet_images()`: Gets image URLs from tweets
- `describe_tweet_images()`: Sends images to vision APIs for description

### Model Configuration
- Base model: Qwen3-14B (4-bit quantized)
- Fine-tuning method: LoRA (r=8, alpha=16)
- Max sequence length: 4096 tokens
- Template: `templates/qwen3-chat` (Jinja2 format)

## Important Notes
- The project handles sensitive personal data (Signal messages)
- Includes privacy features for blocked contacts
- Twitter/X link extraction requires network access
- Training data is saved as JSON files for inspection