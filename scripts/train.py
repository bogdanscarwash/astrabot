#!/usr/bin/env python3
"""
Training orchestration script for Astrabot models.

This script provides a command-line interface for training personalized
language models using your Signal conversation data.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm.training_data_creator import TrainingDataCreator
from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a personalized language model on your conversation data"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=config.FLATFILES_PATH,
        help="Path to Signal flatfiles directory",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Path to save trained model",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_NAME,
        help="Base model name from Hugging Face",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=config.MAX_EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=config.MAX_SEQ_LENGTH,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--your-recipient-id",
        type=int,
        default=2,
        help="Your recipient ID in the Signal database",
    )

    parser.add_argument(
        "--training-mode",
        choices=["conversational", "qa", "adaptive"],
        default="conversational",
        help="Training data creation mode",
    )

    parser.add_argument(
        "--use-twitter-enhancement",
        action="store_true",
        help="Enable Twitter content extraction and enhancement",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def setup_logging(debug=False):
    """Configure logging for the training script."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(config.LOG_FILE_PATH).parent / "training.log"),
        ],
    )


def load_data(data_path):
    """Load Signal data from CSV files."""
    import pandas as pd

    logger.info(f"Loading data from {data_path}")

    try:
        messages_df = pd.read_csv(Path(data_path) / "signal.csv")
        recipients_df = pd.read_csv(Path(data_path) / "recipient.csv")

        logger.info(f"Loaded {len(messages_df)} messages and {len(recipients_df)} recipients")
        return messages_df, recipients_df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)


def create_training_data(messages_df, recipients_df, args):
    """Create training data based on selected mode."""
    logger.info(f"Creating training data in {args.training_mode} mode")

    creator = TrainingDataCreator(
        your_recipient_id=args.your_recipient_id,
        use_twitter_enhancement=args.use_twitter_enhancement,
    )

    if args.training_mode == "conversational":
        training_data = creator.create_conversational_training_data(messages_df, recipients_df)
    elif args.training_mode == "qa":
        # Legacy Q&A mode (not recommended)
        logger.warning("Q&A mode is deprecated. Consider using conversational mode.")
        training_data = creator.create_qa_training_data(messages_df, recipients_df)
    elif args.training_mode == "adaptive":
        training_data = creator.create_adaptive_training_data(messages_df, recipients_df)
    else:
        raise ValueError(f"Unknown training mode: {args.training_mode}")

    logger.info(f"Created {len(training_data)} training examples")
    return training_data


def save_training_data(training_data, output_path):
    """Save training data to disk."""
    output_file = Path(output_path) / "training_data.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved training data to {output_file}")
    return output_file


def train_model(training_data_path, args):
    """Train the model using Unsloth."""
    logger.info("Starting model training")

    # Import here to avoid loading heavy dependencies if just creating data
    try:
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        logger.error(f"Missing required training dependencies: {e}")
        logger.error("Please install: pip install unsloth transformers trl torch")
        sys.exit(1)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
    )

    # Load training data
    with open(training_data_path) as f:
        training_data = json.load(f)

    # Format for training
    from datasets import Dataset

    dataset = Dataset.from_list(training_data)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_path) / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="no",
        logging_dir=str(Path(args.output_path) / "logs"),
        report_to="none",  # Can be "wandb" for Weights & Biases
        fp16=torch.cuda.is_available(),
        optim="adamw_8bit",
        seed=3407,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    output_model_path = Path(args.output_path) / "model"
    logger.info(f"Saving model to {output_model_path}")
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    logger.info("Training complete!")
    return output_model_path


def main():
    """Main training pipeline."""
    args = parse_args()
    setup_logging(args.debug)

    logger.info("Starting Astrabot training pipeline")
    logger.debug(f"Arguments: {args}")

    # Load data
    messages_df, recipients_df = load_data(args.data_path)

    # Create training data
    training_data = create_training_data(messages_df, recipients_df, args)

    # Save training data
    training_data_path = save_training_data(training_data, args.output_path)

    # Train model
    if not args.debug:  # Skip actual training in debug mode
        model_path = train_model(training_data_path, args)
        logger.info(f"Model saved to: {model_path}")
    else:
        logger.info("Debug mode: Skipping actual training")

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()
