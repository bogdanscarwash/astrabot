"""
Astrabot - Personal AI Fine-tuning from Signal Conversations

This package provides tools for creating personalized AI models that mimic your
communication style by analyzing Signal messenger conversation history.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easier access
from src.llm.training_data_creator import TrainingDataCreator, create_training_data_from_signal
from src.core.conversation_processor import (
    extract_tweet_text,
    extract_tweet_images,
    process_message_with_twitter_content
)
from src.utils.logging import get_logger
from src.utils.config import config

__all__ = [
    "TrainingDataCreator",
    "create_training_data_from_signal",
    "extract_tweet_text", 
    "extract_tweet_images",
    "process_message_with_twitter_content",
    "get_logger",
    "config",
]