"""
Structured schemas for Astrabot training data processing.

This module defines Pydantic models and JSON schemas for consistent
data structures when processing tweets and images for training data.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    """Emotional tone categories for images and content."""
    ROMANTIC = "romantic"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    SAD = "sad"
    ANGRY = "angry"
    CONTEMPLATIVE = "contemplative"
    SLEEPY = "sleepy"
    HAPPY = "happy"
    ANXIOUS = "anxious"
    BORED = "bored"
    SENTIMENTAL = "sentimental"
    SENSITIVE = "sensitive"



class Sentiment(str, Enum):
    """Sentiment categories for text content."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ImageDescription(BaseModel):
    """Structured description of an image from AI vision API."""
    description: str = Field(description="Concise 1-2 sentence description of the image")
    detected_text: Optional[str] = Field(None, description="Any text visible in the image")
    main_subjects: List[str] = Field(description="Primary subjects/objects in the image")
    emotional_tone: EmotionalTone = Field(description="Overall emotional tone of the image")
    
    def to_training_format(self) -> str:
        """Convert to a format suitable for training data injection."""
        parts = [self.description]
        if self.detected_text:
            parts.append(f"Text in image: {self.detected_text}")
        if self.main_subjects:
            parts.append(f"Shows: {', '.join(self.main_subjects)}")
        return " | ".join(parts)


class TweetContent(BaseModel):
    """Structured representation of extracted tweet content."""
    text: str = Field(description="The main text content of the tweet")
    author: str = Field(description="Twitter username of the author")
    tweet_id: str = Field(description="Tweet ID")
    mentioned_users: List[str] = Field(default_factory=list, description="Mentioned @usernames")
    hashtags: List[str] = Field(default_factory=list, description="Hashtags in the tweet")
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL, description="Overall sentiment")
    
    def to_training_format(self) -> str:
        """Convert to a format suitable for training data injection."""
        base = f"@{self.author}: {self.text}"
        if self.hashtags:
            base += f" (tags: {', '.join(self.hashtags)})"
        return base


class ImageWithContext(BaseModel):
    """Image URL with its conversation context."""
    image_url: str = Field(description="URL of the image")
    conversation_id: str = Field(description="ID of the conversation/thread")
    message_id: str = Field(description="ID of the message containing the image")
    sender_id: str = Field(description="ID of the person who shared the image")
    timestamp: datetime = Field(description="When the image was shared")
    tweet_url: Optional[str] = Field(None, description="Source tweet URL if from Twitter")


class BatchImageDescription(BaseModel):
    """Image description with its conversation context preserved."""
    image_context: ImageWithContext = Field(description="Original image and its context")
    description: ImageDescription = Field(description="AI-generated description")
    
    def to_dict_with_context(self) -> Dict[str, Any]:
        """Convert to dictionary preserving all context for training data."""
        return {
            "conversation_id": self.image_context.conversation_id,
            "message_id": self.image_context.message_id,
            "sender_id": self.image_context.sender_id,
            "timestamp": self.image_context.timestamp.isoformat(),
            "image_url": self.image_context.image_url,
            "tweet_url": self.image_context.tweet_url,
            "description": self.description.description,
            "detected_text": self.description.detected_text,
            "main_subjects": self.description.main_subjects,
            "emotional_tone": self.description.emotional_tone,
            "formatted_description": self.description.to_training_format()
        }


class EnhancedMessage(BaseModel):
    """A message enhanced with extracted tweet and image content."""
    original_message: str = Field(description="Original message text")
    conversation_id: str = Field(description="Conversation/thread ID")
    message_id: str = Field(description="Unique message ID")
    sender_id: str = Field(description="Message sender ID")
    timestamp: datetime = Field(description="Message timestamp")
    tweet_contents: List[TweetContent] = Field(default_factory=list, description="Extracted tweets")
    image_descriptions: List[ImageDescription] = Field(default_factory=list, description="Image descriptions")
    
    def to_training_format(self) -> str:
        """Convert to training data format with all enhancements."""
        enhanced = self.original_message
        
        # Add tweet content
        for tweet in self.tweet_contents:
            enhanced += f"\n\n[TWEET: {tweet.to_training_format()}]"
        
        # Add image descriptions
        if self.image_descriptions:
            image_parts = [desc.to_training_format() for desc in self.image_descriptions]
            enhanced += f"\n[IMAGES: {' | '.join(image_parts)}]"
        
        return enhanced


def generate_json_schema(model: BaseModel) -> Dict[str, Any]:
    """
    Generate OpenAI-compatible JSON schema from a Pydantic model.
    
    Args:
        model: Pydantic model class
        
    Returns:
        Dict containing the JSON schema in OpenAI's expected format
    """
    schema = model.model_json_schema()
    
    # Remove the title field if present (OpenAI doesn't need it)
    schema.pop('title', None)
    
    # Ensure all objects have additionalProperties: false for strict mode
    def add_additional_properties(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'object' and 'properties' in obj:
                obj['additionalProperties'] = False
            for value in obj.values():
                add_additional_properties(value)
        elif isinstance(obj, list):
            for item in obj:
                add_additional_properties(item)
    
    add_additional_properties(schema)
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__.lower(),
            "schema": schema,
            "strict": True
        }
    }


# Pre-generated schemas for common use cases
IMAGE_DESCRIPTION_SCHEMA = generate_json_schema(ImageDescription)
TWEET_CONTENT_SCHEMA = generate_json_schema(TweetContent)
BATCH_IMAGE_DESCRIPTION_SCHEMA = generate_json_schema(BatchImageDescription)