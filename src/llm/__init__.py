"""
Language Model training utilities for Astrabot.

This package provides tools for creating training data and fine-tuning models.
"""

from src.llm.training_data_creator import (
    TrainingDataCreator,
    create_training_data_from_signal,
    create_conversational_training_data,
)

from src.llm.training_formatter import (
    format_conversational_for_training,
    format_for_alpaca,
    format_for_chat_completion,
    split_burst_sequences,
    filter_by_quality,
    create_weighted_dataset,
    prepare_for_unsloth,
)

from src.llm.adaptive_trainer import (
    create_adaptive_training_data,
    analyze_your_adaptation_patterns,
    create_style_aware_instructions,
    analyze_style_matching_patterns,
    create_persona_based_training_data,
)

__all__ = [
    # Training data creator
    "TrainingDataCreator",
    "create_training_data_from_signal",
    "create_conversational_training_data",
    
    # Training formatter
    "format_conversational_for_training",
    "format_for_alpaca",
    "format_for_chat_completion",
    "split_burst_sequences",
    "filter_by_quality",
    "create_weighted_dataset",
    "prepare_for_unsloth",
    
    # Adaptive trainer
    "create_adaptive_training_data",
    "analyze_your_adaptation_patterns",
    "create_style_aware_instructions",
    "analyze_style_matching_patterns",
    "create_persona_based_training_data",
]