"""
Personality Profiling Module for Astrabot.

This module provides comprehensive personality analysis capabilities including:
- Communication style fingerprinting
- Intellectual vs casual mode detection
- Humor pattern analysis
- Relationship dynamic modeling
- Personal communication signature detection
"""

import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.core.emoji_analyzer import EmojiAnalyzer
from src.core.topic_tracker import TopicTracker
from src.models.conversation_schemas import (
    ConversationDynamics,
    ConversationStyleProfile,
    MessageTiming,
    RelationshipDynamic,
)
from src.models.schemas import EmotionalTone, PersonalityMarkers, TopicCategory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PersonalityProfiler:
    """Advanced personality profiling based on communication patterns."""

    def __init__(self):
        """Initialize personality profiler with analysis components."""
        self.emoji_analyzer = EmojiAnalyzer()
        self.topic_tracker = TopicTracker()
        self._init_language_patterns()
        logger.info("PersonalityProfiler initialized")

    def _init_language_patterns(self) -> None:
        """Initialize language pattern detection."""

        # Academic/intellectual language indicators
        self.academic_patterns = {
            "complex_conjunctions": [
                "furthermore",
                "moreover",
                "however",
                "nevertheless",
                "consequently",
                "therefore",
                "thus",
                "hence",
                "accordingly",
                "subsequently",
            ],
            "analytical_terms": [
                "analysis",
                "synthesis",
                "framework",
                "paradigm",
                "methodology",
                "hypothesis",
                "empirical",
                "theoretical",
                "conceptual",
                "dialectical",
            ],
            "formal_phrases": [
                "it should be noted",
                "one might argue",
                "in this context",
                "from this perspective",
                "it follows that",
                "as previously mentioned",
            ],
        }

        # Casual/internet language patterns
        self.casual_patterns = {
            "contractions": [
                "don't",
                "can't",
                "won't",
                "it's",
                "you're",
                "we're",
                "they're",
                "i'm",
                "he's",
                "she's",
                "that's",
                "what's",
                "where's",
                "how's",
            ],
            "internet_slang": [
                "lmao",
                "lol",
                "omg",
                "wtf",
                "tbh",
                "ngl",
                "fr",
                "bruh",
                "imo",
                "imho",
                "smh",
                "fml",
                "yolo",
                "af",
                "lowkey",
                "highkey",
            ],
            "filler_words": [
                "like",
                "um",
                "uh",
                "you know",
                "i mean",
                "basically",
                "literally",
                "actually",
                "honestly",
                "kinda",
                "sorta",
            ],
        }

        # Profanity patterns (for communication style analysis)
        self.profanity_patterns = [
            r"\bfuck\w*",
            r"\bshit\w*",
            r"\bdamn\w*",
            r"\bhell\w*",
            r"\bass\w*",
            r"\bbitch\w*",
            r"\bcrap\w*",
            r"\bpiss\w*",
        ]

        # Humor indicators
        self.humor_indicators = {
            "laughter": ["haha", "hehe", "lololol", "ahahaha", "bahahaha"],
            "hyperbole": ["literally dying", "dead", "killed me", "can't even", "i can't"],
            "self_deprecation": ["i'm terrible", "i suck", "i'm the worst", "my bad"],
            "wordplay": [],  # Will be detected via patterns
            "sarcasm": ["yeah right", "sure thing", "oh really", "how shocking"],
        }

    def analyze_language_patterns(self, messages: list[str]) -> dict[str, Any]:
        """Analyze language usage patterns in messages."""
        if not messages:
            return {}

        total_words = 0
        academic_score = 0
        casual_score = 0
        profanity_count = 0

        # Contraction usage
        contraction_usage = defaultdict(int)

        # Pattern analysis
        for message in messages:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()
            words = message.split()
            total_words += len(words)

            # Academic language scoring
            for category, patterns in self.academic_patterns.items():
                for pattern in patterns:
                    if pattern in message_lower:
                        academic_score += 2 if category == "analytical_terms" else 1

            # Casual language scoring
            for category, patterns in self.casual_patterns.items():
                for pattern in patterns:
                    if pattern in message_lower:
                        casual_score += 2 if category == "internet_slang" else 1
                        if category == "contractions":
                            contraction_usage[pattern] += 1

            # Profanity detection
            for pattern in self.profanity_patterns:
                profanity_count += len(re.findall(pattern, message_lower))

        # Calculate normalized scores
        formality_score = academic_score / max(total_words / 50, 1)  # Normalize by message density
        casualness_score = casual_score / max(total_words / 50, 1)

        # Overall formality (0 = very casual, 1 = very formal)
        if formality_score + casualness_score > 0:
            formality_ratio = formality_score / (formality_score + casualness_score)
        else:
            formality_ratio = 0.5

        return {
            "formality_level": formality_ratio,
            "academic_score": academic_score,
            "casual_score": casual_score,
            "profanity_frequency": profanity_count / len(messages) if messages else 0,
            "total_words": total_words,
            "avg_words_per_message": total_words / len(messages) if messages else 0,
            "contraction_usage": dict(contraction_usage),
        }

    def detect_humor_patterns(self, messages: list[str]) -> dict[str, Any]:
        """Detect humor patterns and style in messages."""
        humor_scores = defaultdict(int)
        total_messages = len(messages)

        if not messages:
            return {}

        for message in messages:
            if not isinstance(message, str):
                continue

            message_lower = message.lower()

            # Detect different humor types
            for humor_type, indicators in self.humor_indicators.items():
                for indicator in indicators:
                    if indicator in message_lower:
                        humor_scores[humor_type] += 1

            # Wordplay detection (simple patterns)
            if (
                len(set(re.findall(r"\b\w+\b", message_lower))) < len(message_lower.split()) / 2
                and len(message_lower.split()) > 3
            ):
                humor_scores["wordplay"] += 1

            # Excessive punctuation (excitement/humor)
            if re.search(r"[!]{2,}", message) or re.search(r"[?]{2,}", message):
                humor_scores["excitement"] += 1

            # ALL CAPS (shouting/emphasis/humor)
            caps_words = re.findall(r"\b[A-Z]{3,}\b", message)
            if caps_words:
                humor_scores["caps_humor"] += len(caps_words)

        # Calculate humor frequency
        humor_frequency = sum(humor_scores.values()) / total_messages if total_messages > 0 else 0

        return {
            "humor_frequency": humor_frequency,
            "humor_types": dict(humor_scores),
            "dominant_humor_style": (
                max(humor_scores.items(), key=lambda x: x[1])[0] if humor_scores else None
            ),
        }

    def analyze_messaging_style(self, messages_df: pd.DataFrame, sender_id: str) -> dict[str, Any]:
        """Analyze messaging style patterns for a specific sender."""
        sender_messages = messages_df[messages_df["from_recipient_id"] == sender_id].copy()

        if len(sender_messages) == 0:
            return {}

        # Convert timestamps and sort
        sender_messages["datetime"] = pd.to_datetime(sender_messages["date_sent"], unit="ms")
        sender_messages = sender_messages.sort_values("datetime")

        # Message length analysis
        message_texts = []
        message_lengths = []

        for _, row in sender_messages.iterrows():
            body = row.get("body", "")
            if isinstance(body, str) and body.strip():
                message_texts.append(body)
                message_lengths.append(len(body))

        if not message_texts:
            return {}

        # Burst messaging analysis
        burst_sequences = []
        current_burst = []

        for i in range(len(sender_messages)):
            if i == 0:
                current_burst = [sender_messages.iloc[i]]
            else:
                time_diff = (
                    sender_messages.iloc[i]["datetime"] - sender_messages.iloc[i - 1]["datetime"]
                ).total_seconds()

                if time_diff <= 120:  # Within 2 minutes
                    current_burst.append(sender_messages.iloc[i])
                else:
                    if len(current_burst) >= 3:  # Burst of 3+ messages
                        burst_sequences.append(current_burst)
                    current_burst = [sender_messages.iloc[i]]

        # Don't forget the last burst
        if len(current_burst) >= 3:
            burst_sequences.append(current_burst)

        # Correction pattern detection
        corrections = 0
        for i in range(1, len(message_texts)):
            current = message_texts[i].lower()
            previous = message_texts[i - 1].lower()

            # Simple correction detection
            if (
                len(current) > 10
                and len(previous) > 10
                and current.startswith(previous[: int(len(previous) * 0.7)])
            ):
                corrections += 1

        # Response timing analysis
        # This would require analyzing conversation partners, simplified for now

        return {
            "total_messages": len(message_texts),
            "avg_message_length": np.mean(message_lengths),
            "message_length_std": np.std(message_lengths),
            "message_length_distribution": {
                "short": len([l for l in message_lengths if l <= 20]),
                "medium": len([l for l in message_lengths if 20 < l <= 100]),
                "long": len([l for l in message_lengths if l > 100]),
            },
            "burst_messaging": {
                "burst_sequences": len(burst_sequences),
                "avg_burst_size": (
                    np.mean([len(burst) for burst in burst_sequences]) if burst_sequences else 0
                ),
                "burst_tendency": len(burst_sequences) / len(message_texts) if message_texts else 0,
            },
            "correction_frequency": corrections / len(message_texts) if message_texts else 0,
            "message_texts": message_texts,  # For further analysis
        }

    def analyze_relationship_dynamics(
        self, messages_df: pd.DataFrame, sender_id: str, partner_id: str
    ) -> RelationshipDynamic:
        """Analyze relationship dynamic between two conversation partners."""

        # Get conversation between these two people
        dyad_messages = messages_df[
            (
                (messages_df["from_recipient_id"] == sender_id)
                | (messages_df["from_recipient_id"] == partner_id)
            )
        ].copy()

        if len(dyad_messages) < 10:  # Need sufficient data
            return RelationshipDynamic.CASUAL_ACQUAINTANCES

        # Analyze conversation characteristics
        dyad_messages["datetime"] = pd.to_datetime(dyad_messages["date_sent"], unit="ms")
        dyad_messages = dyad_messages.sort_values("datetime")

        # Topic analysis
        political_msgs = 0
        academic_msgs = 0
        personal_msgs = 0
        humor_msgs = 0

        for _, row in dyad_messages.iterrows():
            body = row.get("body", "")
            if not isinstance(body, str):
                continue

            topic_analysis = self.topic_tracker.detect_message_topics(body)
            primary_topic = topic_analysis["primary_topic"]

            if primary_topic in [TopicCategory.POLITICS, TopicCategory.POLITICAL_THEORY]:
                political_msgs += 1
            elif primary_topic == TopicCategory.ACADEMIC:
                academic_msgs += 1
            elif primary_topic == TopicCategory.PERSONAL_LIFE:
                personal_msgs += 1
            elif primary_topic == TopicCategory.HUMOR:
                humor_msgs += 1

        total_msgs = len(dyad_messages)

        # Calculate ratios
        political_ratio = political_msgs / total_msgs
        academic_ratio = academic_msgs / total_msgs
        personal_ratio = personal_msgs / total_msgs
        humor_ratio = humor_msgs / total_msgs

        # Message balance (how equally both parties contribute)
        sender_msgs = len(dyad_messages[dyad_messages["from_recipient_id"] == sender_id])
        balance = min(sender_msgs, total_msgs - sender_msgs) / max(
            sender_msgs, total_msgs - sender_msgs
        )

        # Determine relationship type based on patterns
        if political_ratio > 0.4 and academic_ratio > 0.2:
            if balance > 0.7:
                return RelationshipDynamic.INTELLECTUAL_PEERS
            else:
                return RelationshipDynamic.MENTOR_STUDENT
        elif political_ratio > 0.3:
            return RelationshipDynamic.POLITICAL_ALLIES
        elif personal_ratio > 0.3 and humor_ratio > 0.2:
            return RelationshipDynamic.CLOSE_FRIENDS
        elif humor_ratio > 0.4:
            return RelationshipDynamic.CLOSE_FRIENDS
        else:
            return RelationshipDynamic.CASUAL_ACQUAINTANCES

    def generate_personality_profile(
        self, messages_df: pd.DataFrame, sender_id: str, recipients_df: pd.DataFrame
    ) -> PersonalityMarkers:
        """Generate comprehensive personality profile for a sender."""
        logger.info(f"Generating personality profile for sender {sender_id}")

        # Get sender info
        sender_info = recipients_df[recipients_df["_id"] == sender_id]
        sender_name = (
            sender_info.iloc[0]["profile_given_name"]
            if len(sender_info) > 0
            else f"User_{sender_id}"
        )

        # Get messaging style analysis
        messaging_style = self.analyze_messaging_style(messages_df, sender_id)

        if not messaging_style or not messaging_style.get("message_texts"):
            logger.warning(f"Insufficient data for sender {sender_id}")
            return PersonalityMarkers(
                sender_id=str(sender_id),
                signature_phrases=[],
                emoji_preferences=[],
                message_style="insufficient_data",
                humor_type="unknown",
                academic_tendency=0.0,
                profanity_usage=0.0,
                political_engagement=0.0,
                response_speed_preference="unknown",
            )

        message_texts = messaging_style["message_texts"]

        # Language pattern analysis
        language_patterns = self.analyze_language_patterns(message_texts)

        # Humor analysis
        humor_analysis = self.detect_humor_patterns(message_texts)

        # Emoji analysis
        emoji_signatures = self.emoji_analyzer.detect_emoji_signatures(
            messages_df[messages_df["from_recipient_id"] == sender_id]
        )
        emoji_prefs = emoji_signatures.get(str(sender_id), [])

        # Topic engagement analysis
        topic_analysis = self.topic_tracker.analyze_conversation_topics(
            messages_df[messages_df["from_recipient_id"] == sender_id]
        )

        engagement_scores = topic_analysis.get("topic_engagement_scores", {}).get(
            str(sender_id), {}
        )
        political_engagement = engagement_scores.get("political_engagement", 0.0)

        # Determine message style
        burst_tendency = messaging_style["burst_messaging"]["burst_tendency"]
        avg_length = messaging_style["avg_message_length"]

        if burst_tendency > 0.3:
            if avg_length < 50:
                message_style = "rapid_burst"
            else:
                message_style = "verbose_burst"
        elif avg_length > 200:
            message_style = "long_form"
        elif avg_length < 30:
            message_style = "concise"
        else:
            message_style = "balanced"

        # Extract signature phrases (simple frequency analysis)
        all_text = " ".join(message_texts).lower()
        words = re.findall(r"\b\w+\b", all_text)
        word_freq = Counter(words)

        # Filter out common words
        stopwords = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "a",
            "an",
        }
        signature_phrases = [
            word
            for word, count in word_freq.most_common(20)
            if word not in stopwords and len(word) > 3 and count >= 3
        ]

        # Response speed (simplified - would need conversation partner analysis)
        response_speed = MessageTiming.MODERATE  # Default

        return PersonalityMarkers(
            sender_id=str(sender_id),
            signature_phrases=signature_phrases[:10],
            emoji_preferences=emoji_prefs[:10],
            message_style=message_style,
            humor_type=humor_analysis.get("dominant_humor_style", "unknown"),
            academic_tendency=language_patterns.get("formality_level", 0.0),
            profanity_usage=language_patterns.get("profanity_frequency", 0.0),
            political_engagement=political_engagement,
            response_speed_preference=response_speed.value,
        )

    def generate_conversation_style_profile(
        self, messages_df: pd.DataFrame, participant_ids: list[str], recipients_df: pd.DataFrame
    ) -> ConversationStyleProfile:
        """Generate conversation style profile for a group of participants."""
        logger.info(
            f"Generating conversation style profile for {len(participant_ids)} participants"
        )

        # Filter messages to this group
        group_messages = messages_df[messages_df["from_recipient_id"].isin(participant_ids)].copy()

        if len(group_messages) < 20:
            logger.warning("Insufficient messages for style profile")
            return None

        # Analyze relationship dynamic (simplified for group)
        if len(participant_ids) == 2:
            relationship = self.analyze_relationship_dynamics(
                messages_df, participant_ids[0], participant_ids[1]
            )
        else:
            relationship = RelationshipDynamic.CLOSE_FRIENDS  # Default for groups

        # Topic analysis
        topic_analysis = self.topic_tracker.analyze_conversation_topics(group_messages)
        topic_dist = topic_analysis.get("topic_distribution", {})

        # Determine preferred topics
        sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)
        preferred_topics = [TopicCategory(topic) for topic, _ in sorted_topics[:5]]

        # Conversation dynamics analysis
        political_ratio = (
            topic_dist.get(TopicCategory.POLITICS, 0)
            + topic_dist.get(TopicCategory.POLITICAL_THEORY, 0)
        ) / max(sum(topic_dist.values()), 1)
        humor_ratio = topic_dist.get(TopicCategory.HUMOR, 0) / max(sum(topic_dist.values()), 1)
        academic_ratio = topic_dist.get(TopicCategory.ACADEMIC, 0) / max(
            sum(topic_dist.values()), 1
        )

        # Determine typical dynamics
        typical_dynamics = []
        if political_ratio > 0.3:
            typical_dynamics.append(ConversationDynamics.POLITICAL_DEBATE)
        if academic_ratio > 0.2:
            typical_dynamics.append(ConversationDynamics.PHILOSOPHICAL)
        if humor_ratio > 0.2:
            typical_dynamics.append(ConversationDynamics.CASUAL_BANTER)

        if not typical_dynamics:
            typical_dynamics = [ConversationDynamics.CASUAL_BANTER]

        # Message characteristics
        message_lengths = []
        emoji_count = 0
        url_count = 0

        for _, row in group_messages.iterrows():
            body = row.get("body", "")
            if isinstance(body, str):
                message_lengths.append(len(body))
                emoji_count += len(self.emoji_analyzer.extract_emojis_from_text(body))
                url_count += len(re.findall(r"https?://[^\s]+", body))

        avg_message_length = np.mean(message_lengths) if message_lengths else 0

        # Calculate style metrics
        formality_level = min(academic_ratio * 2, 1.0)  # Academic content indicates formality
        humor_frequency = humor_ratio
        emoji_usage_rate = emoji_count / len(group_messages) if len(group_messages) > 0 else 0
        url_sharing_rate = url_count / len(group_messages) if len(group_messages) > 0 else 0

        return ConversationStyleProfile(
            participant_ids=[str(pid) for pid in participant_ids],
            relationship_type=relationship,
            typical_dynamics=typical_dynamics,
            preferred_topics=preferred_topics,
            emotional_range=[
                EmotionalTone.HUMOROUS,
                EmotionalTone.SERIOUS,
                EmotionalTone.CONTEMPLATIVE,
            ],  # Default range
            formality_level=formality_level,
            humor_frequency=humor_frequency,
            academic_language_usage=academic_ratio,
            emoji_usage_rate=min(emoji_usage_rate, 1.0),
            avg_message_length=avg_message_length,
            burst_messaging_tendency=0.3,  # Simplified
            response_speed_preference=MessageTiming.MODERATE,
            url_sharing_rate=url_sharing_rate,
            political_discussion_rate=political_ratio,
            personal_sharing_rate=topic_dist.get(TopicCategory.PERSONAL_LIFE, 0)
            / max(sum(topic_dist.values()), 1),
        )

    def analyze_communication_adaptation(
        self, messages_df: pd.DataFrame, sender_id: str
    ) -> dict[str, ConversationStyleProfile]:
        """Analyze how a sender adapts their communication style to different conversation partners."""
        logger.info(f"Analyzing communication adaptation for sender {sender_id}")

        sender_messages = messages_df[messages_df["from_recipient_id"] == sender_id]
        adaptation_profiles = {}

        # Group by thread to find conversation partners
        for thread_id in sender_messages["thread_id"].unique():
            thread_messages = messages_df[messages_df["thread_id"] == thread_id]

            # Find other participants in this thread
            other_participants = list(
                thread_messages[thread_messages["from_recipient_id"] != sender_id][
                    "from_recipient_id"
                ].unique()
            )

            if not other_participants or len(thread_messages) < 10:
                continue

            # Create style profile for this conversation
            participants = [sender_id] + other_participants
            try:
                profile = self.generate_conversation_style_profile(
                    thread_messages, participants, pd.DataFrame()  # Empty recipients_df for now
                )

                if profile:
                    key = f"thread_{thread_id}"
                    adaptation_profiles[key] = profile
            except Exception as e:
                logger.warning(f"Error generating profile for thread {thread_id}: {e}")
                continue

        logger.info(f"Generated {len(adaptation_profiles)} adaptation profiles")
        return adaptation_profiles
