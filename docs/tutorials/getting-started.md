# Getting Started with Astrabot

Welcome to Astrabot! This tutorial will guide you through setting up Astrabot and creating your first personalized AI model that mimics your communication style from Signal messenger conversations.

## What You'll Learn

By the end of this tutorial, you will:
- Set up Astrabot on your system
- Extract and process your Signal backup data
- Analyze your communication patterns
- Train a personalized AI model
- Test your model with sample conversations

**Time Required**: 30-60 minutes (plus training time)

## Prerequisites

Before starting, ensure you have:

### Required
- **Python 3.8 or higher** ([install guide](https://www.python.org/downloads/))
- **Signal backup file** (.backup format from Android/iOS)
- **Signal backup password** (30-digit code shown when creating backup)
- **At least 8GB RAM** (16GB+ recommended for larger models)
- **50GB free disk space** for models and data

### Optional but Recommended
- **NVIDIA GPU** with CUDA support (10x faster training)
- **OpenAI API key** for image descriptions in conversations
- **Hugging Face account** for sharing your trained models

## Step 1: Installation

### 1.1 Clone the Repository

```bash
git clone https://github.com/yourusername/astrabot.git
cd astrabot
```

### 1.2 Set Up Python Environment

We recommend using pyenv for Python version management:

```bash
# Install Python 3.11 (recommended)
pyenv install 3.11.7
pyenv local 3.11.7

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 1.3 Install Astrabot

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Or minimal installation:

```bash
pip install -e .
```

### 1.4 Verify Installation

```bash
# Check installation
python -c "from src.utils import get_logger; print('Astrabot installed successfully!')"

# Run tests to ensure everything works
make test-unit
```

## Step 2: Configuration

### 2.1 Set Up Environment Variables

Create your configuration file:

```bash
cp .env.example .env
```

### 2.2 Configure API Keys

Edit `.env` with your preferred editor:

```bash
# Core Settings
YOUR_RECIPIENT_ID=2  # Your Signal ID (usually 2, we'll verify this later)

# API Keys (optional but recommended)
OPENAI_API_KEY=sk-...        # For image descriptions
ANTHROPIC_API_KEY=sk-ant-... # Alternative to OpenAI
HF_TOKEN=hf_...              # For uploading models to Hugging Face

# Advanced Settings (defaults are usually fine)
ENABLE_IMAGE_PROCESSING=true
ENABLE_BATCH_PROCESSING=true
MAX_BATCH_SIZE=10
DEBUG=false
```

### 2.3 Verify Configuration

```bash
python scripts/check-config.py
```

Expected output:
```
✓ Environment file loaded
✓ Data directories created
✓ OpenAI API configured (or ✗ if not set)
✓ Configuration valid
```

## Step 3: Process Your Signal Backup

### 3.1 Prepare Your Backup File

First, locate your Signal backup:
- **Android**: Usually in `/sdcard/Signal/Backups/` or internal storage
- **iOS**: Export via Signal Settings → Chats → Chat Backup

### 3.2 Build the Extraction Tool

```bash
cd docker/signalbackup-tools
docker build -t signalbackup-tools .
cd ../..
```

### 3.3 Extract Your Signal Data

Run the extraction (replace paths and password):

```bash
# Create output directory
mkdir -p data/raw/signal-flatfiles

# Extract backup to CSV files
docker run -v /path/to/your/backup:/backup \
           -v $(pwd)/data/raw/signal-flatfiles:/output \
           signalbackup-tools \
           /backup/signal-2024-01-01-00-00-00.backup \
           --password "12345678901234567890123456789012345678901234567890" \
           --output /output --csv
```

**Note**: The password is the 30-digit code shown when you created the backup.

### 3.4 Verify Extraction

Check that CSV files were created:

```bash
ls -la data/raw/signal-flatfiles/
```

You should see files like:
- `signal.csv` (messages)
- `recipient.csv` (contacts)
- `thread.csv` (conversations)
- And several others

### 3.5 Find Your Recipient ID

Your recipient ID is needed to identify your messages:

```bash
# This will help identify your ID (usually 2)
python scripts/find-my-id.py
```

## Step 4: Explore Your Data

Before training, let's understand your communication patterns:

### 4.1 Basic Data Analysis

```bash
# Get conversation statistics
python scripts/analyze-conversations.py

# Output example:
# Total messages: 45,832
# Your messages: 22,451 (49%)
# Active conversations: 127
# Date range: 2021-01-15 to 2024-01-15
# Most active hours: 20:00-23:00
```

### 4.2 Communication Style Analysis

```python
# Interactive Python session
python

from src.core.style_analyzer import StyleAnalyzer
from src.utils import get_logger

analyzer = StyleAnalyzer()
style = analyzer.analyze_from_csv('data/raw/signal-flatfiles/signal.csv')

print(f"Average message length: {style['avg_length']} chars")
print(f"Emoji usage: {style['emoji_frequency']:.1%}")
print(f"Question frequency: {style['question_ratio']:.1%}")
```

## Step 5: Create Training Data

### Option 1: Using the Notebook (Recommended)

Start Jupyter and follow the guided notebook:

```bash
jupyter notebook notebooks/03_training_pipeline.ipynb
```

The notebook will walk you through:
1. Loading and exploring your data
2. Filtering conversations
3. Creating training examples
4. Visualizing your communication patterns

### Option 2: Using Command Line

For automated processing:

```bash
python scripts/create-training-data.py \
    --input data/raw/signal-flatfiles \
    --output data/processed/training_data.json \
    --your-id 2 \
    --min-messages 10 \
    --include-images
```

Options explained:
- `--min-messages`: Only include conversations with at least N messages
- `--include-images`: Process image descriptions (requires API key)
- `--exclude-blocked`: Skip blocked contacts

### Option 3: Using Python API

```python
from src.llm.training_data_creator import TrainingDataCreator
from src.core.conversation_processor import ConversationProcessor

# Initialize processor
processor = ConversationProcessor(
    your_recipient_id=2,
    enhance_with_twitter=True,
    include_images=True
)

# Process conversations
conversations = processor.process_from_csv(
    'data/raw/signal-flatfiles/signal.csv',
    'data/raw/signal-flatfiles/recipient.csv'
)

# Create training data
creator = TrainingDataCreator()
training_data = creator.create_conversational_data(
    conversations,
    max_examples=10000
)

print(f"Created {len(training_data)} training examples")
```

## Step 6: Fine-tune Your Model

Now we'll train a model on your communication style:

### 6.1 Choose Your Model Size

| Model | VRAM Required | Training Time | Quality |
|-------|---------------|---------------|---------|
| Qwen3-3B | 6GB | 30-60 min | Good for basic style |
| Qwen3-7B | 12GB | 1-2 hours | Balanced performance |
| Qwen3-14B | 24GB | 2-4 hours | Best quality |

### 6.2 Start Training

Using the notebook (recommended):
```bash
# The notebook handles everything
jupyter notebook notebooks/03_training_pipeline.ipynb
```

Or using the command line:
```bash
python scripts/train.py \
    --model "Qwen/Qwen3-7B" \
    --data data/processed/training_data.json \
    --output models/my-style \
    --epochs 2 \
    --batch-size 4
```

### 6.3 Monitor Training

Watch for:
- **Loss decreasing**: Should drop from ~2.0 to ~0.5
- **Examples generated**: Check if they match your style
- **Memory usage**: Reduce batch size if OOM errors

## Step 7: Test Your Model

### 7.1 Interactive Testing

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your fine-tuned model
model = AutoModelForCausalLM.from_pretrained("models/my-style")
tokenizer = AutoTokenizer.from_pretrained("models/my-style")

# Test with prompts
def chat(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Try different conversation starters
print(chat("Hey, how's your day going?"))
print(chat("Did you see the game last night?"))
print(chat("I'm thinking about learning Python"))
```

### 7.2 Evaluation Metrics

Check if your model captures:
- **Message length**: Similar to your average
- **Emoji usage**: Matches your patterns
- **Vocabulary**: Uses your common phrases
- **Response timing**: Multi-message bursts if that's your style

### 7.3 Save and Share

```bash
# Save locally
python scripts/export-model.py --input models/my-style --output my-astrabot-v1

# Upload to Hugging Face (requires HF_TOKEN)
python scripts/upload-to-hf.py --model models/my-style --name "my-communication-style"
```

## What's Next?

### Immediate Next Steps
1. **Fine-tune further**: Try different hyperparameters
2. **Test extensively**: Chat with your model in various contexts
3. **Share safely**: Only share with trusted parties

### Learn More
- [Advanced Training Techniques](advanced-training.md) - Multi-stage training, style adaptation
- [Privacy and Security](../explanation/privacy-considerations.md) - Protecting your data
- [Model Deployment](../how-to/deploy-model.md) - Using your model in applications

### Get Involved
- Star the repository on GitHub
- Share your experience (without sharing personal data!)
- Contribute improvements

## Troubleshooting

### Installation Issues

**Problem**: `pip install` fails
```bash
# Solution: Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
```

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Data Processing Issues

**Problem**: Signal backup extraction fails
- Check password is exactly 30 digits
- Ensure backup file isn't corrupted
- Try with a newer backup

**Problem**: No conversations found
- Verify YOUR_RECIPIENT_ID is correct (use find-my-id.py)
- Check CSV files were created in data/raw/signal-flatfiles
- Ensure you have message history in the backup

### Training Issues

**Problem**: Out of memory (OOM)
```python
# Reduce batch size
trainer.batch_size = 2  # or even 1

# Use gradient accumulation
trainer.gradient_accumulation_steps = 4

# Use smaller model
model_name = "Qwen/Qwen3-3B"  # Instead of 7B or 14B
```

**Problem**: Loss not decreasing
- Check training data quality
- Increase learning rate slightly
- Train for more epochs
- Ensure data has your messages (not just others')

### Getting Help

1. **Check logs**: `data/logs/astrabot.log`
2. **Run diagnostics**: `python scripts/diagnose.py`
3. **Community support**: GitHub Discussions
4. **Issue tracking**: GitHub Issues

Remember: Never share your Signal backup, training data, or personal conversation content when seeking help!