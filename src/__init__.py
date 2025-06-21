"""
Astrabot - Personal AI Fine-tuning from Signal Conversations

This package provides tools for creating personalized AI models that mimic your
communication style by analyzing Signal messenger conversation history.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from src.utils.config import config

# Import main components for easier access
from src.utils.logging import get_logger

__all__ = [
    "get_logger",
    "config",
]
