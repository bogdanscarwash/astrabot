# How to Fine-Tune Qwen3 Models

This guide walks you through fine-tuning Qwen3 models on your personal conversation data using Astrabot's advanced training pipeline.

## Overview

The Qwen3 training pipeline provides:
- Memory-efficient 4-bit quantization with Unsloth
- Multiple training data formats (conversational, adaptive, burst, Q&A)
- Reasoning capability with thinking tags
- Partner-specific style adaptation
- Multi-stage training support

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- Signal conversation data extracted using Astrabot
- Required packages installed (`pip install -e ".[dev]"`)

## Basic Usage

### 1. Quick Start

Train a Qwen3 model with default settings:

```bash
python scripts/train_qwen3.py \
  --messages data/raw/signal-flatfiles/signal.csv \
  --recipients data/raw/signal-flatfiles/recipient.csv \
  --output ./models/my-qwen3-model
```

### 2. Configuration

The training pipeline uses `configs/training_config.yaml`. Key settings:

```yaml
model:
  name: "unsloth/Qwen3-14B"  # Options: 3B, 8B, 14B
  max_seq_length: 4096
  load_in_4bit: true

lora:
  r: 32  # LoRA rank
  alpha: 32  # Alpha (typically = rank)
  
training:
  num_train_epochs: 3
  learning_rate: 2e-5
  per_device_train_batch_size: 2
```

### 3. Model Selection

Choose based on your hardware:

| Model | VRAM Required | Quality | Speed |
|-------|--------------|---------|-------|
| Qwen3-3B | ~6GB | Good | Fast |
| Qwen3-8B | ~16GB | Better | Medium |
| Qwen3-14B | ~28GB | Best | Slower |

## Advanced Usage

### Multi-Stage Training

Train with progressive refinement:

```bash
python scripts/train_qwen3.py \
  --config configs/training_config.yaml \
  --multi-stage \
  --output ./models/qwen3-multistage
```

Stages:
1. **Base**: General conversation patterns
2. **Style Adaptation**: Personal communication style
3. **Partner-Specific**: Adapt to different people

### Custom Configuration

Create a custom config file:

```yaml
# my_config.yaml
model:
  name: "unsloth/Qwen3-8B"
  
dataset:
  modes:
    conversational:
      weight: 0.6  # 60% conversational
    adaptive:
      weight: 0.3  # 30% adaptive
    burst_sequence:
      weight: 0.1  # 10% burst patterns
      
reasoning:
  enabled: true
  ratio: 0.2  # Add 20% reasoning data
```

Use it:

```bash
python scripts/train_qwen3.py --config my_config.yaml
```

### Twitter Enhancement

Enable Twitter content extraction:

```yaml
enhancement:
  twitter:
    enabled: true
    use_nitter: true
```

Set environment variables:
```bash
export OPENAI_API_KEY="your-key"  # For image descriptions
```

### Reasoning Mode

Qwen3 supports reasoning with thinking tags:

```python
messages = [
    {"role": "user", "content": "Solve: If x + 5 = 12, what is x?"}
]

# Model responds with:
# <think>
# I need to solve for x...
# x + 5 = 12
# x = 12 - 5
# x = 7
# </think>
# 
# To solve for x: x = 7
```

## Understanding Output

### Training Metrics

During training, monitor:
- **Loss**: Should decrease over time
- **Memory Usage**: Peak GPU memory
- **Steps/Second**: Training speed

Example output:
```
Epoch 1/3: 100%|████| 1000/1000 [15:23<00:00, 1.08it/s, loss=1.234]
Peak memory: 14.5 GB (98% of max)
Training completed in 2476.63 seconds
```

### Model Files

After training:
```
output/
├── lora_model/          # LoRA adapters
│   ├── adapter_config.json
│   └── adapter_model.bin
├── training_info.json   # Training metadata
└── logs/               # Training logs
```

## Customization

### Data Filtering

Filter messages before training:

```python
# In your script
messages_df = messages_df[
    (messages_df['body'].str.len() > 10) &  # Min length
    (messages_df['body'].str.len() < 1000)  # Max length
]
```

### Style Weights

Adjust how much to adapt to different people:

```yaml
dataset:
  modes:
    adaptive:
      weight: 0.5  # Increase for more adaptation
      analyze_styles: true
```

### Memory Optimization

For limited VRAM:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  
lora:
  r: 16  # Lower rank uses less memory
```

## Performance

### Training Time Estimates

| Model | Examples | Time (V100) | Time (RTX 3090) |
|-------|----------|-------------|-----------------|
| 3B | 10,000 | ~30 min | ~45 min |
| 8B | 10,000 | ~60 min | ~90 min |
| 14B | 10,000 | ~120 min | ~180 min |

### Optimization Tips

1. **Batch Size**: Larger = faster, but uses more memory
2. **Gradient Accumulation**: Simulate larger batches
3. **Mixed Precision**: Use fp16/bf16 for speed
4. **Data Loading**: Use `dataloader_pin_memory: true` if RAM allows

## Troubleshooting

### Out of Memory Errors

```
CUDA out of memory. Tried to allocate...
```

Solutions:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use smaller model
4. Reduce max sequence length

### Slow Training

If training is very slow:
1. Check GPU utilization: `nvidia-smi`
2. Enable mixed precision
3. Reduce logging frequency
4. Use data loading optimizations

### Poor Quality Output

If model outputs are poor:
1. Train for more epochs
2. Increase training data
3. Check data quality
4. Adjust learning rate

### Loading Errors

```
Error loading checkpoint...
```

Ensure:
1. Model name is correct
2. Internet connection for downloading
3. Sufficient disk space
4. Compatible transformers version

## Best Practices

### Data Preparation

1. **Quality over Quantity**: Clean, meaningful conversations
2. **Diversity**: Include various conversation types
3. **Privacy**: Remove sensitive information
4. **Balance**: Mix different data formats

### Training Strategy

1. **Start Small**: Test with subset first
2. **Monitor Progress**: Watch loss curves
3. **Save Checkpoints**: Enable periodic saving
4. **Validate Results**: Test on held-out data

### Deployment

1. **Test Thoroughly**: Try various prompts
2. **Set Boundaries**: Use system prompts
3. **Version Control**: Tag model versions
4. **Document Changes**: Track what each version learned

## Debug Mode

Enable detailed logging:

```bash
python scripts/train_qwen3.py --debug
```

This shows:
- Data loading progress
- Training step details
- Memory allocation
- Model architecture

## See Also

- [Process Signal Data](process-signal-data.md) - Extract conversation data
- [Adaptive Training Guide](adaptive-training.md) - Advanced style adaptation
- [Model Deployment](deploy-model.md) - Using your trained model
- [Privacy Guide](../reference/privacy-guide.md) - Protecting sensitive data