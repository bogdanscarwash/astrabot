# Your First Training Run

This tutorial will guide you through training your first personalized AI model with Astrabot.

## Prerequisites

Before starting, make sure you have:
- Completed the [Getting Started](getting-started.md) guide
- Processed your Signal backup (see [How to Process Signal Backup](../how-to/process-signal-backup.md))
- Set up your environment variables

## Step 1: Explore Your Data

First, let's understand what data we're working with.

### Using Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/03_training_pipeline.ipynb`

3. Run the data exploration cells to see:
   - Number of messages
   - Conversation participants
   - Message frequency over time

### Using the Command Line

```bash
# Get basic statistics about your data
python -c "
import pandas as pd
messages = pd.read_csv('data/raw/signal-flatfiles/signal.csv')
print(f'Total messages: {len(messages)}')
print(f'Date range: {messages[\"date_sent\"].min()} - {messages[\"date_sent\"].max()}')
print(f'Unique threads: {messages[\"thread_id\"].nunique()}')
"
```

## Step 2: Create Training Data

Astrabot offers three training modes:

### Conversational Mode (Recommended)

This mode preserves natural conversation flow:

```bash
python scripts/train.py \
    --training-mode conversational \
    --your-recipient-id 2 \
    --debug
```

This creates training examples that capture:
- Multi-turn conversations
- Your response patterns
- Burst texting style
- Context-aware responses

### Adaptive Mode

This mode learns how you adapt to different people:

```bash
python scripts/train.py \
    --training-mode adaptive \
    --your-recipient-id 2 \
    --debug
```

### With Twitter Enhancement

If you share Twitter/X links, enable content extraction:

```bash
python scripts/train.py \
    --training-mode conversational \
    --use-twitter-enhancement \
    --debug
```

## Step 3: Review Training Data

After creating training data, review it:

```bash
# Check the generated training data
python -c "
import json
with open('outputs/training_data.json', 'r') as f:
    data = json.load(f)
print(f'Total training examples: {len(data)}')
print('\nFirst example:')
print(json.dumps(data[0], indent=2))
"
```

Look for:
- Appropriate conversation context
- Accurate responses
- Preserved communication style

## Step 4: Start Training

Once you're satisfied with the training data, start the actual training:

```bash
python scripts/train.py \
    --training-mode conversational \
    --your-recipient-id 2 \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4
```

### Training Parameters

- **epochs**: Number of times to train on the dataset (3-5 is usually good)
- **batch-size**: Number of examples per training step (4-8 for consumer GPUs)
- **learning-rate**: How fast the model learns (2e-4 is a safe default)

### Monitoring Training

Watch for:
- Loss decreasing over time
- No out-of-memory errors
- Checkpoint saves every 500 steps

## Step 5: Test Your Model

After training completes, test it interactively:

```bash
python scripts/evaluate.py \
    --model-path outputs/model \
    --eval-mode interactive
```

Try conversations similar to your training data:
- Casual greetings
- Questions you typically answer
- Topics you often discuss

## Step 6: Evaluate Quality

Run a full evaluation:

```bash
python scripts/evaluate.py \
    --model-path outputs/model \
    --test-data outputs/training_data.json \
    --eval-mode all \
    --max-samples 100
```

This measures:
- **Perplexity**: How well the model predicts text
- **Style similarity**: How closely it matches your style
- **Response quality**: Coherence and relevance

## Step 7: Export Your Model

Export for different uses:

### For Sharing on Hugging Face

```bash
python scripts/export.py \
    --model-path outputs/model \
    --export-format hf \
    --push-to-hub \
    --hub-repo your-username/your-model-name \
    --private
```

### For Local Use with llama.cpp

```bash
python scripts/export.py \
    --model-path outputs/model \
    --export-format gguf \
    --quantization q4_0
```

## Troubleshooting

### Out of Memory

If you run out of GPU memory:
1. Reduce batch size to 2 or 1
2. Use gradient accumulation
3. Enable CPU offloading

### Poor Quality Results

If the model doesn't capture your style:
1. Check you have enough training data (>1000 examples)
2. Try training for more epochs
3. Ensure your recipient ID is correct
4. Review the training data quality

### Slow Training

To speed up training:
1. Use a smaller model (3B instead of 14B)
2. Reduce max sequence length
3. Enable mixed precision training

## Next Steps

Now that you've trained your first model:

1. **Experiment with Settings**: Try different training modes and parameters
2. **Fine-tune Further**: Continue training on specific types of conversations
3. **Deploy Your Model**: Set up inference for real-time chat
4. **Share with Community**: Contribute improvements back to Astrabot

## Tips for Best Results

1. **Data Quality**: More diverse conversations lead to better models
2. **Privacy First**: Always review training data before sharing models
3. **Iterative Improvement**: Train multiple versions and compare
4. **Regular Updates**: Retrain periodically with new conversations

Congratulations! You've successfully trained your first personalized AI model with Astrabot.