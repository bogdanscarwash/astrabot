"""
Enhanced Conversation Processor for Astrabot.

This module provides comprehensive conversation processing capabilities including:
- Signal conversation analysis with personality profiling
- Burst sequence detection and processing
- Topic tracking and transition analysis
- Emoji pattern analysis and emotional tone detection
- Training data generation with preserved conversation dynamics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import re

from src.models.schemas import (
    TopicCategory, EmotionalTone, MessageType, ConversationMood,
    BurstSequence, TopicTransition, PersonalityMarkers
)
from src.models.conversation_schemas import (
    SignalMessage, ConversationWindow, ConversationThread,
    ConversationDynamics, RelationshipDynamic, MessageTiming,
    ConversationStyleProfile
)
from src.core.emoji_analyzer import EmojiAnalyzer
from src.core.topic_tracker import TopicTracker
from src.core.personality_profiler import PersonalityProfiler
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnhancedConversationProcessor:
    """Comprehensive conversation processor with advanced analysis capabilities."""
    
    def __init__(self):
        """Initialize the enhanced conversation processor."""
        self.emoji_analyzer = EmojiAnalyzer()
        self.topic_tracker = TopicTracker()
        self.personality_profiler = PersonalityProfiler()
        logger.info("EnhancedConversationProcessor initialized")
    
    def process_signal_message(self, row: pd.Series, context_messages: Optional[List[SignalMessage]] = None) -> SignalMessage:
        """Process a single Signal message with comprehensive analysis."""
        
        # Extract basic message data
        message_id = str(row.get('_id', ''))
        thread_id = str(row.get('thread_id', ''))
        sender_id = str(row.get('from_recipient_id', ''))
        timestamp = pd.to_datetime(row.get('date_sent'), unit='ms')
        body = str(row.get('body', '')) if pd.notna(row.get('body')) else ''
        
        # Analyze message content
        emoji_analysis = self.emoji_analyzer.analyze_message_emoji_patterns(body, sender_id, timestamp)
        topic_analysis = self.topic_tracker.detect_message_topics(body)
        
        # Determine message type
        message_type = self._classify_message_type(body, context_messages)
        
        # Extract URLs and emojis
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, body)
        emojis = emoji_analysis.get('emojis_list', [])
        
        # Language analysis
        word_count = len(body.split()) if body else 0
        character_count = len(body)
        
        # Profanity detection (simple)
        profanity_patterns = [r'\bfuck\w*', r'\bshit\w*', r'\bdamn\w*', r'\bhell\w*']
        contains_profanity = any(re.search(pattern, body.lower()) for pattern in profanity_patterns)
        
        # Academic language detection
        academic_indicators = [
            'analysis', 'theory', 'framework', 'paradigm', 'methodology',
            'furthermore', 'moreover', 'however', 'nevertheless'
        ]
        academic_language = sum(1 for indicator in academic_indicators if indicator in body.lower()) >= 2
        
        # Internet slang detection
        slang_indicators = ['lmao', 'lol', 'omg', 'wtf', 'tbh', 'ngl', 'fr', 'bruh']
        internet_slang = any(slang in body.lower() for slang in slang_indicators)
        
        # Conversation context analysis
        response_to_message_id = None
        time_since_previous = None
        is_correction = False
        is_continuation = False
        
        if context_messages:
            last_message = context_messages[-1] if context_messages else None
            if last_message:
                time_since_previous = (timestamp - last_message.timestamp).total_seconds()
                
                # Simple correction detection
                if (body.lower().startswith(last_message.body.lower()[:20]) and 
                    len(body) > len(last_message.body) * 0.8):
                    is_correction = True
                    response_to_message_id = last_message.message_id
                
                # Continuation detection
                elif (time_since_previous < 120 and  # Within 2 minutes
                      last_message.sender_id == sender_id and
                      not body.lower().startswith(('no', 'yes', 'but', 'however'))):
                    is_continuation = True
        
        # Determine emotional tone
        if emoji_analysis.get('has_emojis', False):
            emotional_tone = emoji_analysis.get('dominant_emotion', EmotionalTone.CASUAL)
            # Map emoji emotions to tones
            emotion_mapping = {
                'joy_laughter': EmotionalTone.HUMOROUS,
                'love_affection': EmotionalTone.AFFECTIONATE,
                'anger_frustration': EmotionalTone.ANGRY,
                'sadness_crying': EmotionalTone.SAD,
                'thinking_contemplation': EmotionalTone.CONTEMPLATIVE,
                'playful_teasing': EmotionalTone.PLAYFUL
            }
            emotional_tone = emotion_mapping.get(emotional_tone, EmotionalTone.CASUAL)
        else:
            # Determine tone from content
            if topic_analysis['primary_topic'] == TopicCategory.POLITICS:
                emotional_tone = EmotionalTone.SERIOUS
            elif topic_analysis['primary_topic'] == TopicCategory.ACADEMIC:
                emotional_tone = EmotionalTone.INTELLECTUAL
            elif contains_profanity:
                emotional_tone = EmotionalTone.CASUAL
            else:
                emotional_tone = EmotionalTone.CASUAL
        
        return SignalMessage(
            message_id=message_id,
            thread_id=thread_id,
            sender_id=sender_id,
            timestamp=timestamp,
            body=body,
            message_type=message_type,
            emotional_tone=emotional_tone,
            topic_category=topic_analysis['primary_topic'],
            contains_emoji=emoji_analysis.get('has_emojis', False),
            emoji_list=emojis,
            contains_url=len(urls) > 0,
            url_list=urls,
            word_count=word_count,
            character_count=character_count,
            contains_profanity=contains_profanity,
            academic_language=academic_language,
            internet_slang=internet_slang,
            response_to_message_id=response_to_message_id,
            time_since_previous=time_since_previous,
            is_correction=is_correction,
            is_continuation=is_continuation
        )
    
    def _classify_message_type(self, body: str, context_messages: Optional[List[SignalMessage]]) -> MessageType:
        """Classify the type of message based on content and context."""
        if not context_messages:
            return MessageType.STANDALONE
        
        # Check for corrections
        if context_messages and len(context_messages) > 0:
            last_msg = context_messages[-1]
            if (body.lower().startswith(last_msg.body.lower()[:20]) and 
                len(body) > len(last_msg.body) * 0.8):
                return MessageType.CORRECTION
        
        # Check for media sharing
        if re.search(r'https?://[^\s]+', body):
            return MessageType.MEDIA_SHARE
        
        # Check for elaboration
        if (context_messages and len(context_messages) > 0 and
            context_messages[-1].sender_id == context_messages[-1].sender_id and  # Same sender
            body.lower().startswith(('also', 'and', 'plus', 'additionally', 'furthermore'))):
            return MessageType.ELABORATION
        
        # Check for responses
        if (context_messages and len(context_messages) > 0 and
            context_messages[-1].sender_id != context_messages[-1].sender_id):  # Different sender
            return MessageType.RESPONSE
        
        # Check for continuation
        if (context_messages and len(context_messages) > 0 and
            context_messages[-1].sender_id == context_messages[-1].sender_id):  # Same sender
            return MessageType.CONTINUATION
        
        return MessageType.STANDALONE
    
    def detect_burst_sequences(self, messages: List[SignalMessage], max_gap_seconds: int = 120) -> List[BurstSequence]:
        """Detect burst messaging sequences in conversation."""
        if len(messages) < 3:
            return []
        
        burst_sequences = []
        current_burst = []
        
        for i, message in enumerate(messages):
            if i == 0:
                current_burst = [message]
                continue
            
            # Check if message continues the burst
            time_gap = (message.timestamp - messages[i-1].timestamp).total_seconds()
            same_sender = message.sender_id == messages[i-1].sender_id
            
            if time_gap <= max_gap_seconds and same_sender:
                current_burst.append(message)
            else:
                # End current burst if it's significant
                if len(current_burst) >= 3:
                    burst_seq = self._create_burst_sequence(current_burst)
                    burst_sequences.append(burst_seq)
                
                current_burst = [message]
        
        # Don't forget the last burst
        if len(current_burst) >= 3:
            burst_seq = self._create_burst_sequence(current_burst)
            burst_sequences.append(burst_seq)
        
        return burst_sequences
    
    def _create_burst_sequence(self, messages: List[SignalMessage]) -> BurstSequence:
        """Create a BurstSequence object from a list of messages."""
        message_texts = [msg.body for msg in messages]
        
        duration = (messages[-1].timestamp - messages[0].timestamp).total_seconds()
        avg_length = np.mean([len(msg.body) for msg in messages])
        
        # Check for corrections in burst
        contains_corrections = any(msg.is_correction for msg in messages)
        
        # Determine dominant topic
        topics = [msg.topic_category for msg in messages if msg.topic_category != TopicCategory.OTHER]
        topic_counter = Counter(topics)
        dominant_topic = topic_counter.most_common(1)[0][0] if topic_counter else TopicCategory.OTHER
        
        # Determine emotional tone
        tones = [msg.emotional_tone for msg in messages]
        tone_counter = Counter(tones)
        dominant_tone = tone_counter.most_common(1)[0][0] if tone_counter else EmotionalTone.CASUAL
        
        return BurstSequence(
            messages=message_texts,
            duration_seconds=duration,
            message_count=len(messages),
            avg_message_length=avg_length,
            contains_corrections=contains_corrections,
            topic_category=dominant_topic,
            emotional_tone=dominant_tone
        )
    
    def create_conversation_windows(self, messages: List[SignalMessage], 
                                  window_size: int = 10, 
                                  your_recipient_id: str = "2") -> List[ConversationWindow]:
        """Create conversation windows with comprehensive analysis."""
        windows = []
        
        if len(messages) < window_size:
            return windows
        
        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda x: x.timestamp)
        
        # Create sliding windows
        for i in range(len(sorted_messages) - window_size + 1):
            window_messages = sorted_messages[i:i + window_size]
            
            # Create window
            window = self._create_conversation_window(window_messages, your_recipient_id)
            if window:
                windows.append(window)
        
        return windows
    
    def _create_conversation_window(self, messages: List[SignalMessage], your_recipient_id: str) -> Optional[ConversationWindow]:
        """Create a single conversation window with analysis."""
        if not messages:
            return None
        
        window_id = f"window_{messages[0].thread_id}_{messages[0].timestamp.isoformat()}"
        thread_id = messages[0].thread_id
        
        # Calculate duration
        duration_minutes = (messages[-1].timestamp - messages[0].timestamp).total_seconds() / 60
        
        # Analyze participants
        unique_speakers = list(set(msg.sender_id for msg in messages))
        message_distribution = Counter(msg.sender_id for msg in messages)
        
        # Determine dominant mood
        moods = [msg.emotional_tone for msg in messages]
        mood_counter = Counter(moods)
        
        # Map emotional tones to conversation moods
        tone_to_mood = {
            EmotionalTone.HUMOROUS: ConversationMood.HUMOROUS,
            EmotionalTone.SERIOUS: ConversationMood.SERIOUS,
            EmotionalTone.CONTEMPLATIVE: ConversationMood.PHILOSOPHICAL,
            EmotionalTone.PLAYFUL: ConversationMood.PLAYFUL,
            EmotionalTone.ANGRY: ConversationMood.HEATED,
            EmotionalTone.AFFECTIONATE: ConversationMood.SUPPORTIVE
        }
        
        dominant_mood_tone = mood_counter.most_common(1)[0][0] if mood_counter else EmotionalTone.CASUAL
        dominant_mood = tone_to_mood.get(dominant_mood_tone, ConversationMood.CASUAL)
        
        # Determine primary topic
        topics = [msg.topic_category for msg in messages if msg.topic_category != TopicCategory.OTHER]
        topic_counter = Counter(topics)
        primary_topic = topic_counter.most_common(1)[0][0] if topic_counter else TopicCategory.OTHER
        
        # Analyze conversation dynamics
        conversation_dynamics = self._determine_conversation_dynamics(messages, primary_topic, dominant_mood)
        
        # Detect burst sequences
        burst_sequences = self.detect_burst_sequences(messages)
        
        # Calculate response times
        response_times = []
        for i in range(1, len(messages)):
            if messages[i].sender_id != messages[i-1].sender_id:  # Different speakers
                response_time = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
                if response_time < 3600:  # Within 1 hour
                    response_times.append(response_time)
        
        avg_response_time = np.mean(response_times) if response_times else None
        
        # Count content types
        total_emojis = sum(len(msg.emoji_list) for msg in messages)
        total_urls = sum(len(msg.url_list) for msg in messages)
        
        # Detect topic transitions
        topic_transitions = []
        prev_topic = None
        for msg in messages:
            if prev_topic and msg.topic_category != prev_topic and msg.topic_category != TopicCategory.OTHER:
                transition = TopicTransition(
                    from_topic=prev_topic,
                    to_topic=msg.topic_category,
                    transition_method="gradual",  # Simplified
                    trigger_message=msg.body[:50] + "..." if len(msg.body) > 50 else msg.body,
                    transition_smoothness=0.5  # Default
                )
                topic_transitions.append(transition)
            prev_topic = msg.topic_category
        
        return ConversationWindow(
            window_id=window_id,
            thread_id=thread_id,
            messages=messages,
            start_timestamp=messages[0].timestamp,
            end_timestamp=messages[-1].timestamp,
            duration_minutes=duration_minutes,
            dominant_mood=dominant_mood,
            primary_topic=primary_topic,
            topic_transitions=topic_transitions,
            unique_speakers=unique_speakers,
            message_distribution=dict(message_distribution),
            conversation_dynamics=conversation_dynamics,
            total_emojis=total_emojis,
            total_urls=total_urls,
            burst_sequences=burst_sequences,
            avg_response_time=avg_response_time
        )
    
    def _determine_conversation_dynamics(self, messages: List[SignalMessage], 
                                       primary_topic: TopicCategory, 
                                       dominant_mood: ConversationMood) -> ConversationDynamics:
        """Determine the type of conversation dynamics."""
        
        # Check for rapid-fire exchanges
        quick_exchanges = sum(1 for i in range(1, len(messages)) 
                            if (messages[i].timestamp - messages[i-1].timestamp).total_seconds() < 30)
        
        if quick_exchanges / len(messages) > 0.6:
            return ConversationDynamics.RAPID_FIRE
        
        # Topic-based classification
        if primary_topic in [TopicCategory.POLITICS, TopicCategory.POLITICAL_THEORY]:
            return ConversationDynamics.POLITICAL_DEBATE
        elif primary_topic == TopicCategory.ACADEMIC:
            return ConversationDynamics.PHILOSOPHICAL
        elif primary_topic == TopicCategory.HUMOR:
            return ConversationDynamics.CASUAL_BANTER
        elif primary_topic == TopicCategory.SOCIAL_MEDIA:
            return ConversationDynamics.MEDIA_SHARING
        
        # Mood-based classification
        if dominant_mood == ConversationMood.SUPPORTIVE:
            return ConversationDynamics.SUPPORTIVE
        elif dominant_mood == ConversationMood.PHILOSOPHICAL:
            return ConversationDynamics.PHILOSOPHICAL
        
        # Check for monologue (one person dominating)
        sender_counts = Counter(msg.sender_id for msg in messages)
        max_sender_ratio = max(sender_counts.values()) / len(messages)
        
        if max_sender_ratio > 0.8:
            return ConversationDynamics.MONOLOGUE
        
        # Default
        return ConversationDynamics.CASUAL_BANTER
    
    def process_full_conversation_thread(self, messages_df: pd.DataFrame, 
                                       thread_id: str, 
                                       recipients_df: pd.DataFrame,
                                       your_recipient_id: str = "2") -> ConversationThread:
        """Process a complete conversation thread with comprehensive analysis."""
        logger.info(f"Processing conversation thread {thread_id}")
        
        # Filter to thread messages
        thread_messages_df = messages_df[messages_df['thread_id'] == thread_id].copy()
        thread_messages_df = thread_messages_df.sort_values('date_sent')
        
        # Convert to SignalMessage objects
        signal_messages = []
        for _, row in thread_messages_df.iterrows():
            signal_msg = self.process_signal_message(row, signal_messages[-5:] if signal_messages else None)
            signal_messages.append(signal_msg)
        
        if len(signal_messages) < 5:
            logger.warning(f"Thread {thread_id} has insufficient messages")
            return None
        
        # Create conversation windows
        windows = self.create_conversation_windows(signal_messages, your_recipient_id=your_recipient_id)
        
        # Thread metadata
        participants = list(set(msg.sender_id for msg in signal_messages))
        start_timestamp = signal_messages[0].timestamp
        end_timestamp = signal_messages[-1].timestamp
        total_duration_days = (end_timestamp - start_timestamp).total_seconds() / (24 * 3600)
        
        # Topic evolution analysis
        topic_transitions = []
        for window in windows:
            topic_transitions.extend(window.topic_transitions)
        
        # Dominant topics
        all_topics = [msg.topic_category for msg in signal_messages if msg.topic_category != TopicCategory.OTHER]
        topic_counter = Counter(all_topics)
        dominant_topics = [topic for topic, _ in topic_counter.most_common(5)]
        
        # Mood patterns
        mood_patterns = [window.dominant_mood for window in windows]
        
        # Determine relationship dynamic
        if len(participants) == 2:
            relationship_dynamic = self.personality_profiler.analyze_relationship_dynamics(
                messages_df, participants[0], participants[1]
            )
        else:
            relationship_dynamic = RelationshipDynamic.CLOSE_FRIENDS  # Default for groups
        
        # Generate personality profiles for participants
        participant_personalities = {}
        for participant_id in participants:
            try:
                personality = self.personality_profiler.generate_personality_profile(
                    messages_df, participant_id, recipients_df
                )
                participant_personalities[participant_id] = personality
            except Exception as e:
                logger.warning(f"Could not generate personality profile for {participant_id}: {e}")
        
        # Calculate communication balance
        message_counts = Counter(msg.sender_id for msg in signal_messages)
        total_messages = len(signal_messages)
        communication_balance = {pid: count / total_messages for pid, count in message_counts.items()}
        
        # Analyze response times and emoji patterns (simplified)
        typical_response_times = {pid: MessageTiming.MODERATE for pid in participants}
        emoji_usage_patterns = {}
        url_sharing_frequency = Counter(msg.sender_id for msg in signal_messages if msg.contains_url)
        
        return ConversationThread(
            thread_id=thread_id,
            participants=participants,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            total_messages=len(signal_messages),
            total_duration_days=total_duration_days,
            windows=windows,
            topic_evolution=topic_transitions,
            dominant_topics=dominant_topics,
            mood_patterns=mood_patterns,
            relationship_dynamic=relationship_dynamic,
            participant_personalities=participant_personalities,
            communication_balance=communication_balance,
            typical_response_times=typical_response_times,
            emoji_usage_patterns=emoji_usage_patterns,
            url_sharing_frequency=dict(url_sharing_frequency)
        )
    
    def generate_training_data(self, conversation_thread: ConversationThread, 
                             your_recipient_id: str = "2",
                             style_focus: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate training data from processed conversation thread."""
        logger.info(f"Generating training data for thread {conversation_thread.thread_id}")
        
        training_examples = conversation_thread.get_training_examples(your_recipient_id, style_focus)
        
        # Enhance with conversation analysis
        for example in training_examples:
            # Add personality context
            if your_recipient_id in conversation_thread.participant_personalities:
                personality = conversation_thread.participant_personalities[your_recipient_id]
                example['personality_context'] = {
                    'message_style': personality.message_style,
                    'humor_type': personality.humor_type,
                    'formality_level': personality.academic_tendency,
                    'signature_phrases': personality.signature_phrases[:5]
                }
            
            # Add relationship context
            example['relationship_context'] = {
                'relationship_type': conversation_thread.relationship_dynamic.value,
                'communication_balance': conversation_thread.communication_balance.get(your_recipient_id, 0.5),
                'dominant_topics': [topic.value for topic in conversation_thread.dominant_topics[:3]]
            }
        
        return training_examples