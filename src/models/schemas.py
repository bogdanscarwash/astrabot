"""
Structured schemas for Astrabot training data processing.

This module defines Pydantic models and JSON schemas for consistent
data structures when processing tweets and images for training data.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EmotionalTone(str, Enum):
    """Emotional tone categories based on actual Signal conversation patterns."""

    # Core emotions from analysis
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    CONTEMPLATIVE = "contemplative"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"

    # Additional tones found in Signal data
    SARCASTIC = "sarcastic"
    PHILOSOPHICAL = "philosophical"
    PLAYFUL = "playful"
    TEASING = "teasing"
    INTELLECTUAL = "intellectual"
    MOCKING = "mocking"
    AFFECTIONATE = "affectionate"
    POLITICAL = "political"
    ANALYTICAL = "analytical"
    CASUAL = "casual"
    FLIRTATIOUS = "flirtatious"

    # Legacy tones (keep for backward compatibility)
    ROMANTIC = "romantic"
    SLEEPY = "sleepy"
    BORED = "bored"
    SENTIMENTAL = "sentimental"
    SENSITIVE = "sensitive"


class Sentiment(str, Enum):
    """Sentiment categories for text content."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    SARCASTIC = "sarcastic"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"


class ConversationMood(str, Enum):
    """Overall mood/energy of conversation based on Signal data patterns."""

    RELAXED = "relaxed"
    INTENSE = "intense"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    FLIRTY = "flirty"
    PHILOSOPHICAL = "philosophical"
    CASUAL = "casual"
    HEATED = "heated"
    SUPPORTIVE = "supportive"
    HUMOROUS = "humorous"


class MessageType(str, Enum):
    """Types of messages based on conversation patterns."""

    STANDALONE = "standalone"
    CONTINUATION = "continuation"
    CORRECTION = "correction"
    ELABORATION = "elaboration"
    RESPONSE = "response"
    BURST_START = "burst_start"
    BURST_MIDDLE = "burst_middle"
    BURST_END = "burst_end"
    MEDIA_SHARE = "media_share"
    TANGENT = "tangent"


class TopicCategory(str, Enum):
    """Topic categories based on actual Signal conversation analysis."""

    POLITICS = "politics"
    POLITICAL_THEORY = "political_theory"
    CURRENT_EVENTS = "current_events"
    FOOD = "food"
    PERSONAL_LIFE = "personal_life"
    MEMES = "memes"
    ACADEMIC = "academic"
    TECHNOLOGY = "technology"
    RELATIONSHIPS = "relationships"
    WORK = "work"
    HUMOR = "humor"
    PHILOSOPHY = "philosophy"
    SOCIAL_MEDIA = "social_media"
    NEWS = "news"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


class ImageDescription(BaseModel):
    """Structured description of an image from AI vision API."""

    description: str = Field(description="Concise 1-2 sentence description of the image")
    detected_text: Optional[str] = Field(None, description="Any text visible in the image")
    main_subjects: list[str] = Field(description="Primary subjects/objects in the image")
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
    mentioned_users: list[str] = Field(default_factory=list, description="Mentioned @usernames")
    hashtags: list[str] = Field(default_factory=list, description="Hashtags in the tweet")
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL, description="Overall sentiment")
    is_thread: bool = Field(default=False, description="Whether this tweet is part of a thread")
    is_retweet: bool = Field(default=False, description="Whether this is a retweet")

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

    def to_dict_with_context(self) -> dict[str, Any]:
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
            "formatted_description": self.description.to_training_format(),
        }


class EnhancedMessage(BaseModel):
    """A message enhanced with extracted tweet and image content."""

    original_message: str = Field(description="Original message text")
    conversation_id: str = Field(description="Conversation/thread ID")
    message_id: str = Field(description="Unique message ID")
    sender_id: str = Field(description="Message sender ID")
    timestamp: datetime = Field(description="Message timestamp")
    tweet_contents: list[TweetContent] = Field(default_factory=list, description="Extracted tweets")
    image_descriptions: list[ImageDescription] = Field(
        default_factory=list, description="Image descriptions"
    )

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


class EmojiUsagePattern(BaseModel):
    """Pattern for emoji usage and emotional expression."""

    emoji: str = Field(description="The emoji character")
    frequency: int = Field(description="How often this emoji is used")
    emotional_category: str = Field(description="Emotional category (joy, love, frustration, etc.)")
    usage_context: str = Field(
        description="Typical usage context (standalone, end_message, emphasis, etc.)"
    )
    sender_signature: bool = Field(
        default=False, description="Whether this is a signature emoji for the sender"
    )


class BurstSequence(BaseModel):
    """Sequence of rapid-fire messages sent close together."""

    messages: list[str] = Field(description="Messages in the burst sequence")
    duration_seconds: float = Field(description="Time span of the burst sequence")
    message_count: int = Field(description="Number of messages in burst")
    avg_message_length: float = Field(description="Average length of messages in burst")
    contains_corrections: bool = Field(
        default=False, description="Whether burst contains message corrections"
    )
    topic_category: Optional[TopicCategory] = Field(None, description="Primary topic of the burst")
    emotional_tone: EmotionalTone = Field(description="Overall emotional tone of burst")


class TopicTransition(BaseModel):
    """Tracking how conversations shift between topics."""

    from_topic: TopicCategory = Field(description="Previous topic")
    to_topic: TopicCategory = Field(description="New topic")
    transition_method: str = Field(
        description="How topic changed (abrupt, gradual, media_triggered, etc.)"
    )
    trigger_message: Optional[str] = Field(
        None, description="Message that triggered the transition"
    )
    transition_smoothness: float = Field(description="How smooth the transition was (0-1)")


class PersonalityMarkers(BaseModel):
    """Individual communication quirks and style markers."""

    sender_id: str = Field(description="ID of the sender")
    signature_phrases: list[str] = Field(description="Commonly used phrases or expressions")
    emoji_preferences: list[EmojiUsagePattern] = Field(
        description="Preferred emojis and usage patterns"
    )
    message_style: str = Field(
        description="Overall messaging style (burst, long-form, concise, etc.)"
    )
    humor_type: str = Field(
        description="Type of humor used (sarcastic, meme-heavy, wordplay, etc.)"
    )
    academic_tendency: float = Field(description="Tendency to use academic/formal language (0-1)")
    profanity_usage: float = Field(description="Frequency of profanity usage (0-1)")
    political_engagement: float = Field(
        description="Level of political discussion engagement (0-1)"
    )
    response_speed_preference: str = Field(
        description="Typical response timing (immediate, quick, delayed)"
    )


def generate_json_schema(model: BaseModel) -> dict[str, Any]:
    """
    Generate OpenAI-compatible JSON schema from a Pydantic model.

    Args:
        model: Pydantic model class

    Returns:
        Dict containing the JSON schema in OpenAI's expected format
    """
    schema = model.model_json_schema()

    # Remove the title field if present (OpenAI doesn't need it)
    schema.pop("title", None)

    # Ensure all objects have additionalProperties: false for strict mode
    def add_additional_properties(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object" and "properties" in obj:
                obj["additionalProperties"] = False
            for value in obj.values():
                add_additional_properties(value)
        elif isinstance(obj, list):
            for item in obj:
                add_additional_properties(item)

    add_additional_properties(schema)

    return {
        "type": "json_schema",
        "json_schema": {"name": model.__name__.lower(), "schema": schema, "strict": True},
    }


# Pre-generated schemas for common use cases
IMAGE_DESCRIPTION_SCHEMA = generate_json_schema(ImageDescription)
TWEET_CONTENT_SCHEMA = generate_json_schema(TweetContent)
BATCH_IMAGE_DESCRIPTION_SCHEMA = generate_json_schema(BatchImageDescription)
EMOJI_USAGE_PATTERN_SCHEMA = generate_json_schema(EmojiUsagePattern)
BURST_SEQUENCE_SCHEMA = generate_json_schema(BurstSequence)
PERSONALITY_MARKERS_SCHEMA = generate_json_schema(PersonalityMarkers)
