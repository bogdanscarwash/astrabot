# Getting Started with Astrabot

This tutorial will walk you through setting up Astrabot and creating your first personalized AI model.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher installed
- A Signal backup file (.backup)
- At least 8GB of RAM (16GB+ recommended)
- A CUDA-capable GPU (optional but recommended)

## Step 1: Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astrabot.git
cd astrabot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Step 2: Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# Required for image descriptions
OPENAI_API_KEY=sk-...

# Your Signal recipient ID (usually 2)
YOUR_RECIPIENT_ID=2

# Optional: Hugging Face token for model uploads
HF_TOKEN=hf_...
```

## Step 3: Process Your Signal Backup

1. Build the Docker image for Signal backup processing:
```bash
cd docker/signalbackup-tools
docker build -t signalbackup-tools .
cd ../..
```

2. Extract your Signal backup:
```bash
# Replace with your actual backup path and password
docker run -v /path/to/backup:/backup -v $(pwd)/data/raw/signal-flatfiles:/output \
  signalbackup-tools /backup/signal-2024-01-01-00-00-00.backup \
  --password "your-30-digit-password" \
  --output /output --csv
```

This will create CSV files in `data/raw/signal-flatfiles/`.

## Step 4: Create Training Data

### Option 1: Using the Notebook (Recommended for beginners)

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `notebooks/03_training_pipeline.ipynb`

3. Follow the notebook cells to:
   - Load your Signal data
   - Analyze your communication style
   - Create training examples
   - Fine-tune a model

### Option 2: Using Python Scripts

```python
from src.llm import create_training_data_from_signal

# Create training data
result = create_training_data_from_signal(
    messages_csv_path='data/raw/signal-flatfiles/signal.csv',
    recipients_csv_path='data/raw/signal-flatfiles/recipient.csv',
    output_path='outputs/training_data.json',
    your_recipient_id=2,
    include_twitter=True
)

print(f"Created {result['total_examples']} training examples")
print(f"Your communication style: {result['style_analysis']['preferred_length']}")
```

## Step 5: Fine-tune Your Model

The notebook handles model fine-tuning, but here's what happens:

1. A base model (Qwen-3B to 14B) is loaded
2. Your training data is formatted for the model
3. LoRA adapters are trained on your communication style
4. The model learns your patterns over 1-3 epochs

## Step 6: Test Your Model

After training, test your personalized AI:

```python
# Example prompt
prompt = "Hey, did you see that article about AI safety?"

# Your model will respond in your communication style
response = model.generate(prompt)
print(response)
```

## Next Steps

- Read [Understanding Your Communication Style](../explanation/communication-styles.md)
- Learn about [Advanced Training Options](advanced-training.md)
- Explore [Privacy Best Practices](../explanation/privacy-considerations.md)

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use a smaller model
2. **Signal Backup Errors**: Ensure your password is correct and file isn't corrupted
3. **No Twitter Content**: Check that your API keys are set correctly

### Getting Help

- Check the [FAQ](../reference/faq.md)
- Review the [API Reference](../reference/api/)
- Open an issue on GitHub