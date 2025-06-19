"""
Signal conversation-specific schemas for Astrabot.

This module defines Pydantic models specifically for processing Signal
conversation data, including conversation dynamics, messaging patterns,
and relationship modeling based on actual Signal data analysis.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field

from .schemas import (
    EmotionalTone, Sentiment, ConversationMood, MessageType, 
    TopicCategory, EmojiUsagePattern, BurstSequence, 
    TopicTransition, PersonalityMarkers
)


class ConversationDynamics(str, Enum):
    """Types of conversation dynamics observed in Signal data."""
    RAPID_FIRE = "rapid_fire"           # Quick back-and-forth exchanges
    PHILOSOPHICAL = "philosophical"     # Deep theoretical discussions
    CASUAL_BANTER = "casual_banter"    # Light, humorous conversation
    POLITICAL_DEBATE = "political_debate"  # Intense political discussion
    MEDIA_SHARING = "media_sharing"    # Focused on sharing links/content
    SUPPORTIVE = "supportive"          # Emotional support exchanges
    TANGENTIAL = "tangential"          # Topic-jumping conversation
    MONOLOGUE = "monologue"           # One person speaking at length


class RelationshipDynamic(str, Enum):
    """Types of relationship dynamics in conversations."""
    INTELLECTUAL_PEERS = "intellectual_peers"
    MENTOR_STUDENT = "mentor_student"
    CLOSE_FRIENDS = "close_friends"
    CASUAL_ACQUAINTANCES = "casual_acquaintances"
    ROMANTIC_PARTNERS = "romantic_partners"
    POLITICAL_ALLIES = "political_allies"
    DEBATE_PARTNERS = "debate_partners"


class MessageTiming(str, Enum):
    """Message timing patterns."""
    IMMEDIATE = "immediate"      # < 30 seconds
    QUICK = "quick"             # 30 seconds - 2 minutes
    MODERATE = "moderate"       # 2 - 15 minutes
    DELAYED = "delayed"         # 15 minutes - 1 hour
    LATE = "late"              # > 1 hour


class SignalMessage(BaseModel):
    """Enhanced Signal message with analysis metadata."""
    message_id: str = Field(description="Unique message ID")
    thread_id: str = Field(description="Thread/conversation ID")
    sender_id: str = Field(description="Sender's recipient ID")
    timestamp: datetime = Field(description="Message timestamp")
    body: str = Field(description="Message text content")
    
    # Analysis metadata
    message_type: MessageType = Field(description="Type of message in conversation flow")
    emotional_tone: EmotionalTone = Field(description="Emotional tone of message")
    topic_category: Optional[TopicCategory] = Field(None, description="Primary topic category")
    contains_emoji: bool = Field(default=False, description="Whether message contains emojis")
    emoji_list: List[str] = Field(default_factory=list, description="List of emojis used")
    contains_url: bool = Field(default=False, description="Whether message contains URLs")
    url_list: List[str] = Field(default_factory=list, description="List of URLs in message")
    
    # Language analysis
    word_count: int = Field(description="Number of words in message")
    character_count: int = Field(description="Number of characters in message")
    contains_profanity: bool = Field(default=False, description="Whether message contains profanity")
    academic_language: bool = Field(default=False, description="Whether message uses academic language")
    internet_slang: bool = Field(default=False, description="Whether message uses internet slang")
    
    # Conversation context
    response_to_message_id: Optional[str] = Field(None, description="ID of message this responds to")
    time_since_previous: Optional[float] = Field(None, description="Seconds since previous message")
    is_correction: bool = Field(default=False, description="Whether this corrects a previous message")
    is_continuation: bool = Field(default=False, description="Whether this continues previous thought")


class ConversationWindow(BaseModel):
    """A window of conversation messages with context and analysis."""
    window_id: str = Field(description="Unique identifier for this window")
    thread_id: str = Field(description="Thread this window belongs to")
    messages: List[SignalMessage] = Field(description="Messages in this window")
    
    # Window metadata
    start_timestamp: datetime = Field(description="Timestamp of first message")
    end_timestamp: datetime = Field(description="Timestamp of last message")
    duration_minutes: float = Field(description="Duration of conversation window in minutes")
    
    # Conversation analysis
    dominant_mood: ConversationMood = Field(description="Overall mood of conversation")
    primary_topic: TopicCategory = Field(description="Primary topic discussed")
    topic_transitions: List[TopicTransition] = Field(default_factory=list, description="Topic changes in window")
    
    # Participant analysis
    unique_speakers: List[str] = Field(description="List of sender IDs in window")
    message_distribution: Dict[str, int] = Field(description="Message count per sender")
    conversation_dynamics: ConversationDynamics = Field(description="Type of conversation dynamic")
    
    # Content analysis
    total_emojis: int = Field(default=0, description="Total emojis used in window")
    total_urls: int = Field(default=0, description="Total URLs shared in window")
    burst_sequences: List[BurstSequence] = Field(default_factory=list, description="Rapid message sequences")
    avg_response_time: Optional[float] = Field(None, description="Average response time in seconds")
    
    def to_training_format(self, your_recipient_id: str) -> Dict[str, Any]:
        """Convert conversation window to training data format."""
        # Separate your messages from others
        context_messages = []
        your_responses = []
        
        for msg in self.messages:
            if msg.sender_id == your_recipient_id:
                your_responses.append({
                    "text": msg.body,
                    "timestamp": msg.timestamp.isoformat(),
                    "emotional_tone": msg.emotional_tone,
                    "message_type": msg.message_type
                })
            else:
                context_messages.append({
                    "speaker": "Other",
                    "text": msg.body,
                    "timestamp": msg.timestamp.isoformat()
                })
        
        return {
            "conversation_id": self.thread_id,
            "window_id": self.window_id,
            "context": context_messages,
            "responses": your_responses,
            "metadata": {
                "mood": self.dominant_mood,
                "topic": self.primary_topic,
                "dynamics": self.conversation_dynamics,
                "duration_minutes": self.duration_minutes,
                "burst_sequences": len(self.burst_sequences),
                "avg_response_time": self.avg_response_time
            }
        }


class ConversationThread(BaseModel):
    """Complete conversation thread with comprehensive analysis."""
    thread_id: str = Field(description="Unique thread identifier")
    participants: List[str] = Field(description="List of participant recipient IDs")
    
    # Thread metadata
    start_timestamp: datetime = Field(description="First message timestamp")
    end_timestamp: datetime = Field(description="Last message timestamp")
    total_messages: int = Field(description="Total number of messages")
    total_duration_days: float = Field(description="Total conversation span in days")
    
    # Content analysis
    windows: List[ConversationWindow] = Field(description="Conversation windows in thread")
    topic_evolution: List[TopicTransition] = Field(description="How topics evolved over time")
    dominant_topics: List[TopicCategory] = Field(description="Most discussed topics")
    mood_patterns: List[ConversationMood] = Field(description="Mood changes over time")
    
    # Participant dynamics
    relationship_dynamic: RelationshipDynamic = Field(description="Type of relationship")
    participant_personalities: Dict[str, PersonalityMarkers] = Field(description="Personality analysis per participant")
    communication_balance: Dict[str, float] = Field(description="Speaking time distribution (0-1)")
    
    # Behavioral patterns
    typical_response_times: Dict[str, MessageTiming] = Field(description="Typical response speed per participant")
    emoji_usage_patterns: Dict[str, List[EmojiUsagePattern]] = Field(description="Emoji patterns per participant")
    url_sharing_frequency: Dict[str, int] = Field(description="URL sharing count per participant")
    
    def get_training_examples(self, your_recipient_id: str, style_focus: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract training examples from this conversation thread."""
        examples = []
        
        for window in self.windows:
            # Filter windows based on style focus if specified
            if style_focus:
                if style_focus == "philosophical" and window.conversation_dynamics != ConversationDynamics.PHILOSOPHICAL:
                    continue
                elif style_focus == "casual" and window.conversation_dynamics not in [ConversationDynamics.CASUAL_BANTER, ConversationDynamics.MEDIA_SHARING]:
                    continue
                elif style_focus == "political" and window.conversation_dynamics != ConversationDynamics.POLITICAL_DEBATE:
                    continue
            
            training_data = window.to_training_format(your_recipient_id)
            
            # Add thread-level context
            training_data["thread_metadata"] = {
                "relationship": self.relationship_dynamic,
                "thread_topics": self.dominant_topics,
                "conversation_style": window.conversation_dynamics
            }
            
            examples.append(training_data)
        
        return examples


class ConversationStyleProfile(BaseModel):
    """Profile of conversation style for a specific relationship/context."""
    participant_ids: List[str] = Field(description="Participants in this relationship")
    relationship_type: RelationshipDynamic = Field(description="Type of relationship")
    
    # Communication patterns
    typical_dynamics: List[ConversationDynamics] = Field(description="Common conversation types")
    preferred_topics: List[TopicCategory] = Field(description="Most discussed topics")
    emotional_range: List[EmotionalTone] = Field(description="Range of emotions expressed")
    
    # Style characteristics
    formality_level: float = Field(description="How formal communication is (0-1)")
    humor_frequency: float = Field(description="How often humor is used (0-1)")
    academic_language_usage: float = Field(description="Frequency of academic language (0-1)")
    emoji_usage_rate: float = Field(description="Rate of emoji usage (0-1)")
    
    # Conversation flow
    avg_message_length: float = Field(description="Average message length in characters")
    burst_messaging_tendency: float = Field(description="Tendency for burst messaging (0-1)")
    response_speed_preference: MessageTiming = Field(description="Preferred response timing")
    
    # Content sharing
    url_sharing_rate: float = Field(description="Rate of URL sharing (0-1)")
    political_discussion_rate: float = Field(description="Rate of political discussion (0-1)")
    personal_sharing_rate: float = Field(description="Rate of personal information sharing (0-1)")
    
    def generate_style_instructions(self) -> str:
        """Generate natural language instructions for replicating this communication style."""
        instructions = []
        
        # Formality
        if self.formality_level > 0.7:
            instructions.append("Use formal, academic language with complex sentence structures")
        elif self.formality_level < 0.3:
            instructions.append("Use casual, conversational language with contractions and slang")
        
        # Humor
        if self.humor_frequency > 0.5:
            instructions.append("Include humor, jokes, and playful banter frequently")
        
        # Message style
        if self.burst_messaging_tendency > 0.6:
            instructions.append("Send multiple short messages in quick succession rather than long single messages")
        elif self.avg_message_length > 200:
            instructions.append("Write longer, more detailed messages that fully express thoughts")
        
        # Emojis
        if self.emoji_usage_rate > 0.4:
            instructions.append("Use emojis regularly to express emotions and add personality")
        
        # Topics
        if TopicCategory.POLITICS in self.preferred_topics:
            instructions.append("Engage in political and theoretical discussions with depth and nuance")
        
        if TopicCategory.PERSONAL_LIFE in self.preferred_topics:
            instructions.append("Share personal experiences and ask about the other person's life")
        
        return ". ".join(instructions) + "."