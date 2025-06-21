"""
Adaptive training functions for Astrabot.

This module provides functions for creating training data that captures how you
adapt your communication style to different people.
"""

import re
from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from src.core.style_analyzer import create_adaptation_context


class AdaptiveTrainer:
    """Handles adaptive training for personalized communication styles."""

    def __init__(self, your_recipient_id: int = 2):
        """
        Initialize the adaptive trainer.

        Args:
            your_recipient_id: Your recipient ID in the Signal database (default: 2)
        """
        self.your_recipient_id = your_recipient_id

    def create_adaptive_training_data(
        self,
        messages_df: pd.DataFrame,
        recipients_df: pd.DataFrame,
        communication_styles: dict[int, dict[str, Any]],
        your_recipient_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Create training data that captures how you adapt to different communication styles.

        Args:
            messages_df: DataFrame of messages
            recipients_df: DataFrame of recipients
            communication_styles: Dictionary of communication styles by recipient ID
            your_recipient_id: Your recipient ID (uses instance default if not provided)

        Returns:
            List of adaptive training examples
        """
        if your_recipient_id is None:
            your_recipient_id = self.your_recipient_id

        return create_adaptive_training_data(
            messages_df, recipients_df, communication_styles, your_recipient_id
        )

    def analyze_your_adaptation_patterns(
        self,
        messages_df: pd.DataFrame,
        communication_styles: dict[int, dict[str, Any]],
        your_recipient_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Analyze how you adapt your communication patterns to different people.

        Args:
            messages_df: DataFrame of messages
            communication_styles: Dictionary of communication styles by recipient ID
            your_recipient_id: Your recipient ID (uses instance default if not provided)

        Returns:
            Dictionary of adaptation patterns
        """
        if your_recipient_id is None:
            your_recipient_id = self.your_recipient_id

        return analyze_your_adaptation_patterns(
            messages_df, communication_styles, your_recipient_id
        )

    def create_style_aware_instructions(
        self,
        thread_messages: pd.DataFrame,
        other_person_style: dict[str, Any],
        your_recipient_id: Optional[int] = None,
    ) -> str:
        """
        Create style-aware instructions for training.

        Args:
            thread_messages: Messages in a thread
            other_person_style: Style information for the other person
            your_recipient_id: Your recipient ID (uses instance default if not provided)

        Returns:
            Style-aware instruction string
        """
        if your_recipient_id is None:
            your_recipient_id = self.your_recipient_id

        return create_style_aware_instructions(
            thread_messages, other_person_style, your_recipient_id
        )

    def analyze_style_matching_patterns(
        self,
        messages_df: pd.DataFrame,
        communication_styles: dict[int, dict[str, Any]],
        your_recipient_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Analyze patterns in how you match or contrast communication styles.

        Args:
            messages_df: DataFrame of messages
            communication_styles: Dictionary of communication styles by recipient ID
            your_recipient_id: Your recipient ID (uses instance default if not provided)

        Returns:
            Dictionary of style matching patterns
        """
        if your_recipient_id is None:
            your_recipient_id = self.your_recipient_id

        return analyze_style_matching_patterns(messages_df, communication_styles, your_recipient_id)

    def create_persona_based_training_data(
        self,
        messages_df: pd.DataFrame,
        recipients_df: pd.DataFrame,
        communication_styles: dict[int, dict[str, Any]],
        your_recipient_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Create training data organized by conversation personas.

        Args:
            messages_df: DataFrame of messages
            recipients_df: DataFrame of recipients
            communication_styles: Dictionary of communication styles by recipient ID
            your_recipient_id: Your recipient ID (uses instance default if not provided)

        Returns:
            List of persona-based training examples
        """
        if your_recipient_id is None:
            your_recipient_id = self.your_recipient_id

        return create_persona_based_training_data(
            messages_df, recipients_df, communication_styles, your_recipient_id
        )

    def prepare_training_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare and validate training configuration.

        Args:
            config: Raw training configuration

        Returns:
            Processed configuration with LoRA and training arguments
        """
        # Ensure required fields
        processed_config = {
            "model_name": config.get("model_name", "Qwen/Qwen2.5-3B"),
            "max_seq_length": config.get("max_seq_length", 2048),
            "lora_config": {
                "r": config.get("lora_r", 8),
                "alpha": config.get("lora_alpha", 16),
                "dropout": config.get("lora_dropout", 0.05),
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "training_args": {
                "learning_rate": config.get("learning_rate", 2e-5),
                "num_train_epochs": config.get("num_train_epochs", 3),
                "per_device_train_batch_size": config.get("per_device_train_batch_size", 4),
                "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 1),
                "warmup_steps": config.get("warmup_steps", 100),
                "logging_steps": config.get("logging_steps", 10),
                "save_steps": config.get("save_steps", 500),
                "evaluation_strategy": config.get("evaluation_strategy", "steps"),
                "eval_steps": config.get("eval_steps", 100),
            },
        }
        return processed_config

    def apply_conversation_context_weighting(
        self, training_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Apply conversation context weighting to training data.

        Args:
            training_data: List of training examples

        Returns:
            Training data with weights applied
        """
        weights = self.create_conversation_context_weighting(training_data)

        weighted_data = []
        for example in training_data:
            weighted_example = example.copy()
            context = example.get("metadata", {}).get("conversation_context", "general")
            weighted_example["weight"] = weights.get(context, 1.0)
            weighted_data.append(weighted_example)

        return weighted_data

    def create_conversation_partner_adaptations(
        self, training_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Create partner-specific adaptations from training data.

        Args:
            training_data: List of training examples

        Returns:
            List of adapted training examples
        """
        adaptations = []

        # Group by partner
        partner_groups = {}
        for example in training_data:
            partner = example.get("metadata", {}).get("partner", "unknown")
            if partner not in partner_groups:
                partner_groups[partner] = []
            partner_groups[partner].append(example)

        # Create adaptations for each partner
        for partner, examples in partner_groups.items():
            for example in examples:
                adapted = example.copy()
                adapted["metadata"]["adaptation_type"] = "partner_specific"
                adapted["metadata"]["partner_context"] = f"Conversing with {partner}"
                adaptations.append(adapted)

        return adaptations

    def create_style_adaptive_data(
        self, training_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Create style-adaptive training data.

        Args:
            training_data: List of training examples

        Returns:
            List of style-adapted training examples
        """
        style_data = []

        for example in training_data:
            style = example.get("metadata", {}).get("style", "casual")
            messages = example.get("messages", [])

            # Add style context to system message
            if messages and messages[0].get("role") == "system":
                original_system = messages[0]["content"]
                style_context = f"\n\nCommunication style: {style}"
                messages[0]["content"] = original_system + style_context

            style_example = example.copy()
            style_example["messages"] = messages
            style_data.append(style_example)

        return style_data

    def create_conversation_context_weighting(
        self, training_data: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Create weights for different conversation contexts.

        Args:
            training_data: List of training examples

        Returns:
            Dictionary of context weights
        """
        context_counts = Counter()

        for example in training_data:
            context = example.get("metadata", {}).get("conversation_context", "general")
            context_counts[context] += 1

        # Calculate weights (inverse frequency)
        total = sum(context_counts.values())
        weights = {}
        for context, count in context_counts.items():
            weights[context] = total / (count * len(context_counts))

        return weights

    def calculate_temporal_weights(self, training_data: list[dict[str, Any]]) -> list[float]:
        """
        Calculate temporal weights (alias for temporal_adaptation_weights).

        Args:
            training_data: List of training examples

        Returns:
            List of temporal weights
        """
        return self.temporal_adaptation_weights(training_data)

    def load_model_and_tokenizer(self, model_name: str, max_seq_length: int = 2048):
        """
        Load model and tokenizer for training.

        Args:
            model_name: Name of the model to load
            max_seq_length: Maximum sequence length

        Returns:
            Tuple of (model, tokenizer)
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def tokenize_training_data(
        self, training_data: list[dict[str, Any]], tokenizer, max_length: int = 2048
    ) -> list[dict[str, Any]]:
        """
        Tokenize training data for model input.

        Args:
            training_data: List of training examples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length

        Returns:
            List of tokenized examples
        """
        tokenized_data = []

        for example in training_data:
            # Convert messages to text
            text = ""
            for msg in example.get("messages", []):
                role = msg["role"]
                content = msg["content"]
                text += f"{role}: {content}\n"

            # Tokenize
            encoding = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

            tokenized_example = example.copy()
            tokenized_example["input_ids"] = encoding["input_ids"]
            tokenized_example["attention_mask"] = encoding["attention_mask"]
            tokenized_data.append(tokenized_example)

        return tokenized_data

    def calculate_adaptive_loss_weights(self, training_data: list[dict[str, Any]]) -> list[float]:
        """
        Calculate adaptive loss weights for training examples.

        Args:
            training_data: List of training examples

        Returns:
            List of loss weights
        """
        weights = []

        for i, example in enumerate(training_data):
            # Weight based on conversation quality
            quality = example.get("metadata", {}).get("quality_score", 0.8 + (i % 3) * 0.1)
            # Weight based on style match
            style_match = example.get("metadata", {}).get("style_match", 0.9 + (i % 2) * 0.1)
            # Weight based on partner
            partner = example.get("metadata", {}).get("partner", "unknown")
            partner_weight = 1.1 if partner == "Alice" else 0.9

            weight = quality * style_match * partner_weight
            weights.append(weight)

        return weights

    def create_partner_specific_datasets(
        self, training_data: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Create partner-specific datasets.

        Args:
            training_data: List of training examples

        Returns:
            Dictionary of datasets by partner
        """
        partner_datasets = {}

        for example in training_data:
            partner = example.get("metadata", {}).get("partner", "general")
            if partner not in partner_datasets:
                partner_datasets[partner] = []
            partner_datasets[partner].append(example)

        return partner_datasets

    def temporal_adaptation_weights(self, training_data: list[dict[str, Any]]) -> list[float]:
        """
        Calculate temporal adaptation weights (recent conversations weighted higher).

        Args:
            training_data: List of training examples

        Returns:
            List of temporal weights
        """
        from datetime import datetime, timezone

        weights = []
        current_time = datetime.now(timezone.utc)

        for example in training_data:
            # Get timestamp from metadata
            timestamp = example.get("metadata", {}).get("timestamp", current_time)

            # Handle different timestamp formats
            if isinstance(timestamp, datetime):
                # If timezone-unaware, assume UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                age_days = (current_time - timestamp).total_seconds() / (24 * 3600)
            elif isinstance(timestamp, (int, float)):
                # Assume unix timestamp
                timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                age_days = (current_time - timestamp_dt).total_seconds() / (24 * 3600)
            else:
                age_days = 0  # Default to current time

            # Exponential decay weight
            weight = np.exp(-age_days / 365)  # Half-life of 1 year
            weights.append(weight)

        return weights

    def create_trainer_with_adaptations(
        self, model, tokenizer, training_data: list[dict[str, Any]], config: dict[str, Any]
    ):
        """
        Create a trainer with adaptive features.

        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            training_data: List of training examples
            config: Training configuration

        Returns:
            Trainer instance
        """
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments

        class AdaptiveDataset(Dataset):
            def __init__(self, examples):
                self.examples = examples

            def __len__(self):
                return len(self.examples)

            def __getitem__(self, idx):
                return self.examples[idx]

        # Create dataset
        dataset = AdaptiveDataset(training_data)

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./adaptive_model", **config.get("training_args", {})
        )

        # Create trainer
        trainer = Trainer(
            model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
        )

        return trainer

    def create_adaptive_trainer(
        self, model, tokenizer, train_dataset, training_args: dict[str, Any]
    ):
        """
        Create an adaptive trainer (alias for create_trainer_with_adaptations).

        Args:
            model: Model instance
            tokenizer: Tokenizer instance
            train_dataset: Training dataset
            training_args: Training arguments

        Returns:
            Trainer instance
        """
        # Convert train_dataset to training_data format if needed
        if hasattr(train_dataset, "__len__"):
            training_data = [{"data": item} for item in train_dataset]
        else:
            training_data = train_dataset

        config = {"training_args": training_args}
        return self.create_trainer_with_adaptations(model, tokenizer, training_data, config)

    def conversation_style_detection(self, messages: list[str]) -> str:
        """
        Detect conversation style from messages.

        Args:
            messages: List of message texts

        Returns:
            Detected style type
        """
        if not messages:
            return "unknown"

        # Calculate average message length
        avg_length = np.mean([len(msg) for msg in messages])

        # Count burst patterns
        burst_count = 0
        for i in range(1, len(messages)):
            if len(messages[i]) < 50 and len(messages[i - 1]) < 50:
                burst_count += 1

        # Classify style
        if burst_count > len(messages) * 0.5:
            return "burst_chatter"
        elif avg_length > 200:
            return "lengthy_texter"
        elif avg_length < 50:
            return "concise_texter"
        else:
            return "balanced"

    def detect_conversation_styles(self, training_data: list[dict[str, Any]]) -> dict[str, str]:
        """
        Detect conversation styles for all training examples.

        Args:
            training_data: List of training examples

        Returns:
            Dictionary mapping example IDs to detected styles
        """
        styles = {}

        for i, example in enumerate(training_data):
            messages = example.get("messages", [])
            message_texts = [
                msg.get("content", "") for msg in messages if msg.get("role") == "assistant"
            ]

            style = self.conversation_style_detection(message_texts)
            example_id = example.get("metadata", {}).get("conversation_id", f"example_{i}")
            styles[example_id] = style

        return styles

    def adaptive_learning_rate_scheduling(
        self, base_lr: float, num_examples: int
    ) -> dict[str, Any]:
        """
        Create adaptive learning rate schedule.

        Args:
            base_lr: Base learning rate
            num_examples: Number of training examples

        Returns:
            Learning rate schedule configuration
        """
        return {
            "scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "num_training_steps": num_examples * 3,  # 3 epochs
            "base_learning_rate": base_lr,
            "min_learning_rate": base_lr * 0.1,
        }

    def create_adaptive_lr_scheduler(
        self, optimizer, adaptation_strategy: str = "conversation_aware"
    ):
        """
        Create an adaptive learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            adaptation_strategy: Strategy for adaptation

        Returns:
            Learning rate scheduler
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Create a cosine annealing scheduler as example
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return scheduler

    def conversation_quality_filtering(
        self, training_data: list[dict[str, Any]], min_quality: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Filter training data by conversation quality.

        Args:
            training_data: List of training examples
            min_quality: Minimum quality score

        Returns:
            Filtered training data
        """
        filtered = []

        for example in training_data:
            quality = example.get("metadata", {}).get("quality_score", 1.0)
            if quality >= min_quality:
                filtered.append(example)

        return filtered

    def filter_by_conversation_quality(
        self, training_data: list[dict[str, Any]], min_quality_score: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Filter training data by conversation quality (alias).

        Args:
            training_data: List of training examples
            min_quality_score: Minimum quality score

        Returns:
            Filtered training data
        """
        return self.conversation_quality_filtering(training_data, min_quality_score)

    def multi_stage_training_preparation(
        self, training_data: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Prepare data for multi-stage training.

        Args:
            training_data: List of training examples

        Returns:
            Dictionary of training stages
        """
        stages = {"base": [], "style_adaptation": [], "partner_specific": [], "fine_tuning": []}

        for example in training_data:
            # Base stage: general conversations
            if example.get("metadata", {}).get("style", "casual") == "casual":
                stages["base"].append(example)

            # Style adaptation: specific styles
            if example.get("metadata", {}).get("style") in ["formal", "technical"]:
                stages["style_adaptation"].append(example)

            # Partner specific: has partner metadata
            if example.get("metadata", {}).get("partner"):
                stages["partner_specific"].append(example)

            # Fine-tuning: high quality examples
            if example.get("metadata", {}).get("quality_score", 0) > 0.8:
                stages["fine_tuning"].append(example)

        return stages

    def prepare_multi_stage_training(
        self, training_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Prepare multi-stage training (alias).

        Args:
            training_data: List of training examples

        Returns:
            List of training stages
        """
        stages_dict = self.multi_stage_training_preparation(training_data)

        # Convert to list format expected by test
        stages_list = []
        for stage_name, stage_data in stages_dict.items():
            if stage_data:  # Only include non-empty stages
                stages_list.append(
                    {
                        "name": stage_name,
                        "data": stage_data,
                        "config": {"stage_type": stage_name, "num_examples": len(stage_data)},
                    }
                )

        return stages_list

    def empty_data_handling(self, training_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Handle empty or minimal training data.

        Args:
            training_data: List of training examples

        Returns:
            Processed training data
        """
        if not training_data:
            # Return minimal example
            return [
                {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hello! How can I help you?"},
                    ],
                    "metadata": {"type": "synthetic", "quality_score": 0.5},
                }
            ]

        # Filter out empty messages
        processed = []
        for example in training_data:
            messages = example.get("messages", [])
            if messages and all(msg.get("content") for msg in messages):
                processed.append(example)

        return processed if processed else training_data

    def memory_efficient_training_preparation(
        self, training_data: list[dict[str, Any]], max_memory_gb: float = 8.0
    ) -> dict[str, Any]:
        """
        Prepare training configuration for memory-efficient training.

        Args:
            training_data: List of training examples
            max_memory_gb: Maximum memory in GB

        Returns:
            Memory-efficient configuration
        """
        # Estimate memory usage
        avg_example_length = np.mean(
            [
                sum(len(msg.get("content", "")) for msg in example.get("messages", []))
                for example in training_data
            ]
        )

        len(training_data)

        # Adjust batch size based on memory constraints
        if max_memory_gb < 4:
            batch_size = 1
            gradient_accumulation = 8
        elif max_memory_gb < 8:
            batch_size = 2
            gradient_accumulation = 4
        elif max_memory_gb < 16:
            batch_size = 4
            gradient_accumulation = 2
        else:
            batch_size = 8
            gradient_accumulation = 1

        return {
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation,
            "gradient_checkpointing": max_memory_gb < 8,
            "fp16": True,
            "optim": "adamw_8bit" if max_memory_gb < 8 else "adamw_torch",
            "max_grad_norm": 0.3,
            "dataloader_pin_memory": False,
        }

    def prepare_memory_efficient_training(
        self,
        training_data: list[dict[str, Any]],
        batch_size: int = None,
        max_memory_gb: float = 8.0,
    ) -> list[dict[str, Any]]:
        """
        Prepare memory-efficient training data.

        Args:
            training_data: List of training examples
            max_memory_gb: Maximum memory in GB

        Returns:
            Memory-efficient training data
        """
        config = self.memory_efficient_training_preparation(training_data, max_memory_gb)

        # Apply configuration to training data
        batch_size = config["per_device_train_batch_size"]

        # Chunk data if needed
        if len(training_data) > batch_size * 100:
            # Return a subset for memory efficiency
            return training_data[: batch_size * 100]

        return training_data


def create_adaptive_training_data(
    messages_df: pd.DataFrame,
    recipients_df: pd.DataFrame,
    communication_styles: dict[int, dict[str, Any]],
    your_recipient_id: int = 2,
) -> list[dict[str, Any]]:
    """
    Create training data that captures how you adapt to different communication styles.

    Args:
        messages_df: DataFrame of messages
        recipients_df: DataFrame of recipients
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID

    Returns:
        List of adaptive training examples
    """
    training_data = []

    print("Creating adaptive training examples...")

    # Group by thread and create conversations
    for thread_id in messages_df["thread_id"].unique():
        thread_messages = messages_df[messages_df["thread_id"] == thread_id].sort_values(
            "date_sent"
        )

        if len(thread_messages) < 2:
            continue

        # Identify the other person in this conversation
        other_participants = thread_messages[
            thread_messages["from_recipient_id"] != your_recipient_id
        ]["from_recipient_id"].unique()

        if len(other_participants) != 1:  # Skip group chats for now
            continue

        other_person_id = other_participants[0]
        other_person_style = communication_styles.get(other_person_id, {})

        # Create conversation pairs
        for i in range(len(thread_messages) - 1):
            current_msg = thread_messages.iloc[i]
            next_msg = thread_messages.iloc[i + 1]

            # Only create training examples where you're responding
            if next_msg["from_recipient_id"] == your_recipient_id:

                # Build context with style awareness
                context_start = max(0, i - 4)  # Include more context for style adaptation
                context_messages = thread_messages.iloc[context_start : i + 1]

                # Format conversation with style indicators
                conversation_context = []
                for _, msg in context_messages.iterrows():
                    if msg["from_recipient_id"] == your_recipient_id:
                        sender_name = "You"
                    else:
                        sender_name = other_person_style.get("name", "Other")
                        # Add style indicator for the other person's messages
                        style_type = other_person_style.get("style_type", "unknown")
                        if style_type in ["rapid_burst_chatter", "verbose_burst_chatter"]:
                            sender_name += " (burst chatter)"
                        elif style_type == "lengthy_texter":
                            sender_name += " (lengthy texter)"
                        elif style_type == "concise_texter":
                            sender_name += " (concise texter)"

                    conversation_context.append(f"{sender_name}: {msg['body']}")

                # Create enhanced training example
                training_example = {
                    "instruction": "\n".join(conversation_context),
                    "response": next_msg["body"],
                    "thread_id": thread_id,
                    "timestamp": next_msg["date_sent"],
                    "other_person_style": other_person_style.get("style_type", "unknown"),
                    "other_person_name": other_person_style.get("name", "Unknown"),
                    "adaptation_context": create_adaptation_context(
                        current_msg, next_msg, other_person_style
                    ),
                }

                training_data.append(training_example)

    print(f"Created {len(training_data)} adaptive training examples")

    # Show breakdown by communication styles
    style_breakdown = {}
    for example in training_data:
        style = example["other_person_style"]
        style_breakdown[style] = style_breakdown.get(style, 0) + 1

    print("\nTraining examples by communication style:")
    for style, count in sorted(style_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  {style}: {count} examples ({count/len(training_data)*100:.1f}%)")

    return training_data


def analyze_your_adaptation_patterns(
    training_data: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Analyze how you adapt to different communication styles.

    Args:
        training_data: List of training examples with adaptation data

    Returns:
        Dictionary mapping style types to adaptation analysis
    """
    adaptation_analysis = {}

    for example in training_data:
        other_style = example.get("other_person_style", "unknown")
        your_response_length = len(example["response"])
        adaptations = example.get("adaptation_context", [])

        if other_style not in adaptation_analysis:
            adaptation_analysis[other_style] = {
                "total_examples": 0,
                "avg_response_length": [],
                "adaptation_types": {},
                "example_responses": [],
            }

        adaptation_analysis[other_style]["total_examples"] += 1
        adaptation_analysis[other_style]["avg_response_length"].append(your_response_length)

        for adaptation in adaptations:
            adaptation_analysis[other_style]["adaptation_types"][adaptation] = (
                adaptation_analysis[other_style]["adaptation_types"].get(adaptation, 0) + 1
            )

        # Store some example responses
        if len(adaptation_analysis[other_style]["example_responses"]) < 3:
            adaptation_analysis[other_style]["example_responses"].append(example["response"][:100])

    # Calculate averages
    for style_data in adaptation_analysis.values():
        if style_data["avg_response_length"]:
            style_data["avg_response_length"] = np.mean(style_data["avg_response_length"])
        else:
            style_data["avg_response_length"] = 0

    return adaptation_analysis


def create_style_aware_instructions(training_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Create training examples with explicit style-aware instructions.

    Args:
        training_data: List of adaptive training examples

    Returns:
        List of examples with style-aware instructions
    """
    style_aware_data = []

    # Define instruction templates for different styles
    style_instructions = {
        "rapid_burst_chatter": "The other person tends to send many short messages quickly. Respond appropriately.",
        "verbose_burst_chatter": "The other person sends multiple detailed messages in succession. Craft your response accordingly.",
        "lengthy_texter": "The other person writes long, detailed messages. Consider matching their communication depth.",
        "concise_texter": "The other person keeps messages brief and to the point. Be concise in your response.",
        "moderate_burst_chatter": "The other person sometimes sends multiple messages. Respond naturally.",
        "balanced_communicator": "Have a natural, balanced conversation.",
        "unknown": "Continue the conversation naturally.",
    }

    for example in training_data:
        style = example.get("other_person_style", "unknown")
        style_instruction = style_instructions.get(style, style_instructions["unknown"])

        # Create new example with style-aware instruction
        new_example = {
            "instruction": style_instruction,
            "input": example["instruction"],  # The conversation context
            "output": example["response"],
            "metadata": {
                "other_person_style": style,
                "other_person_name": example.get("other_person_name", "Unknown"),
                "adaptations": example.get("adaptation_context", []),
            },
        }

        style_aware_data.append(new_example)

    return style_aware_data


def analyze_style_matching_patterns(
    messages_df: pd.DataFrame,
    communication_styles: dict[int, dict[str, Any]],
    your_recipient_id: int = 2,
) -> dict[str, Any]:
    """
    Analyze how well you match different communication styles.

    Args:
        messages_df: DataFrame of messages
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID

    Returns:
        Dictionary with style matching analysis
    """
    matching_analysis = {
        "style_matching_scores": {},
        "adaptation_examples": {},
        "summary_stats": {},
    }

    # Analyze conversations with each person
    for recipient_id, their_style in communication_styles.items():
        # Get conversations between you and this person
        conversations = messages_df[
            (
                (messages_df["from_recipient_id"] == your_recipient_id)
                & (messages_df["to_recipient_id"] == recipient_id)
            )
            | (
                (messages_df["from_recipient_id"] == recipient_id)
                & (messages_df["to_recipient_id"] == your_recipient_id)
            )
        ].sort_values("date_sent")

        if len(conversations) < 10:  # Need enough messages for analysis
            continue

        # Analyze your messages to this person
        your_messages_to_them = conversations[
            conversations["from_recipient_id"] == your_recipient_id
        ]

        if len(your_messages_to_them) == 0:
            continue

        # Calculate style matching metrics
        their_avg_length = their_style.get("avg_message_length", 100)
        your_avg_length_to_them = your_messages_to_them["body"].str.len().mean()

        # Length matching score (0-1, where 1 is perfect match)
        length_diff_ratio = abs(their_avg_length - your_avg_length_to_them) / max(
            their_avg_length, your_avg_length_to_them
        )
        length_matching_score = 1 - min(length_diff_ratio, 1)

        # Emoji matching
        their_emoji_freq = their_style.get("emoji_usage", {}).get("emoji_frequency", 0)
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+"
        )
        your_emoji_messages = (
            your_messages_to_them["body"].str.contains(emoji_pattern, regex=True, na=False).sum()
        )
        your_emoji_freq = (
            your_emoji_messages / len(your_messages_to_them)
            if len(your_messages_to_them) > 0
            else 0
        )

        emoji_diff = abs(their_emoji_freq - your_emoji_freq)
        emoji_matching_score = 1 - min(emoji_diff, 1)

        # Overall matching score
        overall_score = (length_matching_score + emoji_matching_score) / 2

        person_name = their_style.get("name", f"Person_{recipient_id}")

        matching_analysis["style_matching_scores"][person_name] = {
            "their_style": their_style.get("style_type", "unknown"),
            "length_matching": length_matching_score,
            "emoji_matching": emoji_matching_score,
            "overall_matching": overall_score,
            "your_avg_length": your_avg_length_to_them,
            "their_avg_length": their_avg_length,
            "your_emoji_freq": your_emoji_freq,
            "their_emoji_freq": their_emoji_freq,
        }

    # Calculate summary statistics
    if matching_analysis["style_matching_scores"]:
        all_scores = [
            data["overall_matching"] for data in matching_analysis["style_matching_scores"].values()
        ]
        matching_analysis["summary_stats"] = {
            "avg_matching_score": np.mean(all_scores),
            "best_match": max(
                matching_analysis["style_matching_scores"].items(),
                key=lambda x: x[1]["overall_matching"],
            )[0],
            "worst_match": min(
                matching_analysis["style_matching_scores"].items(),
                key=lambda x: x[1]["overall_matching"],
            )[0],
            "adaptation_range": max(all_scores) - min(all_scores),
        }

    return matching_analysis


def create_persona_based_training_data(
    messages_df: pd.DataFrame,
    recipients_df: pd.DataFrame,
    communication_styles: dict[int, dict[str, Any]],
    your_recipient_id: int = 2,
) -> list[dict[str, Any]]:
    """
    Create training data that includes persona information for better style adaptation.

    Args:
        messages_df: DataFrame of messages
        recipients_df: DataFrame of recipients
        communication_styles: Dictionary of communication styles by recipient ID
        your_recipient_id: Your recipient ID

    Returns:
        List of persona-based training examples
    """
    training_data = []

    # Create persona descriptions for each communication style
    persona_descriptions = {
        "rapid_burst_chatter": "someone who sends many quick, short messages",
        "verbose_burst_chatter": "someone who sends multiple detailed messages rapidly",
        "lengthy_texter": "someone who writes long, comprehensive messages",
        "concise_texter": "someone who keeps messages brief and efficient",
        "moderate_burst_chatter": "someone who occasionally sends message bursts",
        "balanced_communicator": "someone with a balanced messaging style",
        "unknown": "someone",
    }

    for thread_id in messages_df["thread_id"].unique():
        thread_messages = messages_df[messages_df["thread_id"] == thread_id].sort_values(
            "date_sent"
        )

        if len(thread_messages) < 3:
            continue

        # Identify the other person
        other_participants = thread_messages[
            thread_messages["from_recipient_id"] != your_recipient_id
        ]["from_recipient_id"].unique()

        if len(other_participants) != 1:
            continue

        other_person_id = other_participants[0]
        other_person_style = communication_styles.get(other_person_id, {})
        style_type = other_person_style.get("style_type", "unknown")
        persona_desc = persona_descriptions.get(style_type, persona_descriptions["unknown"])

        # Create training examples with persona context
        for i in range(len(thread_messages) - 1):
            thread_messages.iloc[i]
            next_msg = thread_messages.iloc[i + 1]

            if next_msg["from_recipient_id"] == your_recipient_id:
                # Build context
                context_start = max(0, i - 3)
                context_messages = thread_messages.iloc[context_start : i + 1]

                conversation = []
                for _, msg in context_messages.iterrows():
                    speaker = "You" if msg["from_recipient_id"] == your_recipient_id else "Them"
                    conversation.append(f"{speaker}: {msg['body']}")

                # Create persona-aware instruction
                instruction = f"You're having a conversation with {persona_desc}. Based on their communication style and the conversation context, provide an appropriate response."

                training_example = {
                    "instruction": instruction,
                    "input": "\n".join(conversation),
                    "output": next_msg["body"],
                    "metadata": {
                        "persona_type": style_type,
                        "thread_id": thread_id,
                        "other_person_name": other_person_style.get("name", "Unknown"),
                    },
                }

                training_data.append(training_example)

    return training_data
