"""
Unit tests for personality profiler module
"""

from datetime import timedelta

import pandas as pd
import pytest

from src.core.personality_profiler import PersonalityProfiler
from src.models.conversation_schemas import (
    ConversationStyleProfile,
    MessageTiming,
    RelationshipDynamic,
)
from src.models.schemas import PersonalityMarkers


@pytest.mark.unit
class TestPersonalityProfiler:
    """Test personality profiling functionality"""

    @pytest.fixture
    def profiler(self):
        """Create PersonalityProfiler instance"""
        return PersonalityProfiler()

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for analysis"""
        return [
            "Hey! How's it going? Just wanted to check in 😊",
            "Furthermore, I believe the theoretical framework needs adjustment",
            "lmao that's crazy fr fr 😂",
            "The empirical analysis suggests a paradigm shift is necessary",
            "btw did u see that? lowkey impressive ngl",
            "Consequently, we must reconsider our methodology",
            "haha yeah totally! that's what I'm saying!!",
            "It should be noted that this hypothesis requires further testing",
        ]

    @pytest.fixture
    def sample_messages_df(self):
        """Create sample messages DataFrame for comprehensive analysis"""
        base_time = pd.Timestamp("2024-01-01 12:00:00")

        # Create diverse conversation data
        messages = []

        # Academic conversation
        for i in range(5):
            messages.append(
                {
                    "_id": i,
                    "thread_id": 1,
                    "from_recipient_id": 2,
                    "to_recipient_id": 3,
                    "body": [
                        "Furthermore, the theoretical framework suggests...",
                        "Indeed, the empirical evidence supports this hypothesis",
                        "However, we must consider the methodological limitations",
                        "Consequently, further analysis is required",
                        "In conclusion, the paradigm shift is inevitable",
                    ][i],
                    "date_sent": int((base_time + timedelta(minutes=i * 5)).timestamp() * 1000),
                }
            )

        # Casual conversation
        for i in range(5):
            messages.append(
                {
                    "_id": i + 5,
                    "thread_id": 2,
                    "from_recipient_id": 2,
                    "to_recipient_id": 4,
                    "body": [
                        "lmao did you see that? 😂",
                        "fr fr that's wild",
                        "btw what r u doing later?",
                        "omg yes!!! let's goooo 🎉",
                        "ngl that's lowkey amazing",
                    ][i],
                    "date_sent": int((base_time + timedelta(minutes=i * 2)).timestamp() * 1000),
                }
            )

        # Political discussion
        for i in range(3):
            messages.append(
                {
                    "_id": i + 10,
                    "thread_id": 3,
                    "from_recipient_id": 2,
                    "to_recipient_id": 5,
                    "body": [
                        "The current political climate is concerning",
                        "This policy will have far-reaching implications",
                        "We need systemic change to address these issues",
                    ][i],
                    "date_sent": int((base_time + timedelta(minutes=i * 10)).timestamp() * 1000),
                }
            )

        return pd.DataFrame(messages)

    @pytest.fixture
    def sample_recipients_df(self):
        """Create sample recipients DataFrame"""
        return pd.DataFrame(
            {
                "_id": [2, 3, 4, 5],
                "profile_given_name": ["TestUser", "Academic", "Casual", "Political"],
                "blocked": [0, 0, 0, 0],
            }
        )

    def test_language_pattern_analysis(self, profiler, sample_messages):
        """Test language pattern detection"""
        analysis = profiler.analyze_language_patterns(sample_messages)

        assert isinstance(analysis, dict)
        assert "formality_level" in analysis
        assert 0 <= analysis["formality_level"] <= 1
        assert "academic_score" in analysis
        assert "casual_score" in analysis
        assert "profanity_frequency" in analysis
        assert "avg_words_per_message" in analysis
        assert "contraction_usage" in analysis

        # With mixed messages, should have both academic and casual scores
        assert analysis["academic_score"] > 0
        assert analysis["casual_score"] > 0

    def test_academic_language_detection(self, profiler):
        """Test detection of academic/formal language"""
        academic_messages = [
            "Furthermore, the analysis suggests a correlation",
            "Consequently, we must reconsider our hypothesis",
            "The theoretical framework provides empirical evidence",
            "Nevertheless, the methodology remains sound",
        ]

        analysis = profiler.analyze_language_patterns(academic_messages)

        assert analysis["academic_score"] > analysis["casual_score"]
        assert analysis["formality_level"] > 0.6  # Should be formal

    def test_casual_language_detection(self, profiler):
        """Test detection of casual/internet language"""
        casual_messages = [
            "lmao that's crazy fr",
            "btw did u see that? it's lowkey amazing",
            "omg I can't even rn",
            "ngl that's actually pretty cool tho",
        ]

        analysis = profiler.analyze_language_patterns(casual_messages)

        assert analysis["casual_score"] > analysis["academic_score"]
        assert analysis["formality_level"] < 0.4  # Should be casual

    def test_humor_pattern_detection(self, profiler):
        """Test humor detection patterns"""
        humorous_messages = [
            "hahaha that's hilarious!",
            "lololol I'm literally dying 😂",
            "THAT'S AMAZING!!!",
            "i'm terrible at this lol",
            "yeah right, sure thing buddy",
        ]

        humor_analysis = profiler.detect_humor_patterns(humorous_messages)

        assert isinstance(humor_analysis, dict)
        assert humor_analysis["humor_frequency"] > 0
        assert "humor_types" in humor_analysis
        assert humor_analysis["dominant_humor_style"] is not None

        # Should detect various humor types
        humor_types = humor_analysis["humor_types"]
        assert humor_types.get("laughter", 0) > 0
        assert humor_types.get("excitement", 0) > 0

    def test_messaging_style_analysis(self, profiler, sample_messages_df):
        """Test messaging style pattern analysis"""
        style_analysis = profiler.analyze_messaging_style(sample_messages_df, "2")

        assert isinstance(style_analysis, dict)
        assert "total_messages" in style_analysis
        assert "avg_message_length" in style_analysis
        assert "message_length_distribution" in style_analysis
        assert "burst_messaging" in style_analysis
        assert "correction_frequency" in style_analysis

        # Check message length distribution
        dist = style_analysis["message_length_distribution"]
        assert "short" in dist
        assert "medium" in dist
        assert "long" in dist

        # Check burst analysis
        burst = style_analysis["burst_messaging"]
        assert "burst_sequences" in burst
        assert "avg_burst_size" in burst
        assert "burst_tendency" in burst

    def test_relationship_dynamics_analysis(self, profiler, sample_messages_df):
        """Test relationship dynamic detection"""
        # Test academic relationship
        academic_dynamic = profiler.analyze_relationship_dynamics(sample_messages_df, "2", "3")
        assert academic_dynamic in [
            RelationshipDynamic.INTELLECTUAL_PEERS,
            RelationshipDynamic.MENTOR_STUDENT,
            RelationshipDynamic.CASUAL_ACQUAINTANCES,
        ]

        # Test casual relationship
        casual_dynamic = profiler.analyze_relationship_dynamics(sample_messages_df, "2", "4")
        assert casual_dynamic in [
            RelationshipDynamic.CLOSE_FRIENDS,
            RelationshipDynamic.CASUAL_ACQUAINTANCES,
        ]

    def test_personality_profile_generation(
        self, profiler, sample_messages_df, sample_recipients_df
    ):
        """Test comprehensive personality profile generation"""
        profile = profiler.generate_personality_profile(
            sample_messages_df, "2", sample_recipients_df
        )

        assert isinstance(profile, PersonalityMarkers)
        assert profile.sender_id == "2"
        assert isinstance(profile.signature_phrases, list)
        assert isinstance(profile.emoji_preferences, list)
        assert profile.message_style in [
            "rapid_burst",
            "verbose_burst",
            "long_form",
            "concise",
            "balanced",
        ]
        assert 0 <= profile.academic_tendency <= 1
        assert 0 <= profile.profanity_usage <= 1
        assert 0 <= profile.political_engagement <= 1
        assert profile.response_speed_preference in [
            MessageTiming.IMMEDIATE.value,
            MessageTiming.QUICK.value,
            MessageTiming.MODERATE.value,
            MessageTiming.DELAYED.value,
        ]

    def test_empty_data_handling(self, profiler):
        """Test handling of empty or minimal data"""
        empty_messages = []

        # Language analysis with empty data
        lang_analysis = profiler.analyze_language_patterns(empty_messages)
        assert lang_analysis == {}

        # Humor analysis with empty data
        humor_analysis = profiler.detect_humor_patterns(empty_messages)
        assert humor_analysis == {}

        # Empty DataFrame
        empty_df = pd.DataFrame()
        empty_recipients = pd.DataFrame()

        profile = profiler.generate_personality_profile(empty_df, "999", empty_recipients)
        assert profile.sender_id == "999"
        assert profile.message_style == "insufficient_data"

    def test_profanity_detection(self, profiler):
        """Test profanity pattern detection"""
        messages_with_profanity = [
            "This is fucking amazing!",
            "Holy shit, that's incredible",
            "Damn, that's a good point",
            "What the hell is going on?",
        ]

        analysis = profiler.analyze_language_patterns(messages_with_profanity)
        assert analysis["profanity_frequency"] > 0

    def test_conversation_style_profile_generation(
        self, profiler, sample_messages_df, sample_recipients_df
    ):
        """Test conversation style profile for groups"""
        participants = ["2", "3"]

        style_profile = profiler.generate_conversation_style_profile(
            sample_messages_df, participants, sample_recipients_df
        )

        if style_profile:  # May return None for insufficient data
            assert isinstance(style_profile, ConversationStyleProfile)
            assert style_profile.participant_ids == participants
            assert isinstance(style_profile.relationship_type, RelationshipDynamic)
            assert isinstance(style_profile.typical_dynamics, list)
            assert isinstance(style_profile.preferred_topics, list)
            assert 0 <= style_profile.formality_level <= 1
            assert 0 <= style_profile.humor_frequency <= 1

    def test_communication_adaptation_analysis(self, profiler, sample_messages_df):
        """Test analysis of communication style adaptation"""
        adaptations = profiler.analyze_communication_adaptation(sample_messages_df, "2")

        assert isinstance(adaptations, dict)

        # Should have different profiles for different conversation partners
        for key, profile in adaptations.items():
            assert key.startswith("thread_")
            if profile:
                assert isinstance(profile, ConversationStyleProfile)

    def test_signature_phrase_detection(self, profiler):
        """Test detection of signature phrases"""
        # Messages with repeated phrases
        messages_with_signatures = [
            "Actually, that's a good point",
            "Actually, I was thinking the same",
            "You know what? That makes sense",
            "You know what? I agree",
            "Basically, it comes down to this",
            "Basically, we need to consider",
        ]

        # Analyze for user who uses these messages
        df = pd.DataFrame(
            {
                "_id": range(len(messages_with_signatures)),
                "from_recipient_id": ["2"] * len(messages_with_signatures),
                "body": messages_with_signatures,
                "date_sent": [
                    1640995200000 + i * 60000 for i in range(len(messages_with_signatures))
                ],
            }
        )

        recipients_df = pd.DataFrame({"_id": ["2"], "profile_given_name": ["TestUser"]})

        profile = profiler.generate_personality_profile(df, "2", recipients_df)

        # Should detect repeated words/phrases
        assert len(profile.signature_phrases) > 0
        # Common words like 'actually', 'basically' should be detected
        common_words = ["actually", "basically"]
        assert any(word in profile.signature_phrases for word in common_words)

    def test_emoji_preference_integration(self, profiler, sample_messages_df, sample_recipients_df):
        """Test integration with emoji analyzer for preferences"""
        # Add some messages with emojis
        emoji_messages = pd.DataFrame(
            {
                "_id": [100, 101, 102],
                "from_recipient_id": ["2", "2", "2"],
                "body": ["That's great! 😊 😊", "Love it! ❤️ ❤️ ❤️", "So happy! 😊 ❤️"],
                "date_sent": [1640995200000 + i * 60000 for i in range(3)],
            }
        )

        combined_df = pd.concat([sample_messages_df, emoji_messages], ignore_index=True)

        profile = profiler.generate_personality_profile(combined_df, "2", sample_recipients_df)

        # Should have emoji preferences
        assert isinstance(profile.emoji_preferences, list)
        # Note: May be empty if emoji analyzer doesn't find significant patterns

    def test_topic_engagement_scoring(self, profiler, sample_messages_df):
        """Test topic engagement analysis"""
        # The profiler should work with topic tracker to identify engagement
        profile = profiler.generate_personality_profile(
            sample_messages_df, "2", pd.DataFrame({"_id": ["2"], "profile_given_name": ["Test"]})
        )

        # Political engagement should be detected from political messages
        assert isinstance(profile.political_engagement, float)
        assert 0 <= profile.political_engagement <= 1

    def test_message_style_classification(self, profiler):
        """Test different message style classifications"""
        # Rapid burst style
        burst_df = pd.DataFrame(
            {
                "_id": range(10),
                "from_recipient_id": ["2"] * 10,
                "body": ["Short msg"] * 10,
                "date_sent": [1640995200000 + i * 30000 for i in range(10)],  # 30 seconds apart
            }
        )

        style_analysis = profiler.analyze_messaging_style(burst_df, "2")
        assert style_analysis["burst_messaging"]["burst_sequences"] > 0

        # Long form style
        long_df = pd.DataFrame(
            {
                "_id": range(3),
                "from_recipient_id": ["2"] * 3,
                "body": [
                    "This is a very long message that contains a lot of detailed information and goes on for quite a while to demonstrate the long-form writing style that some people prefer when they have a lot to say"
                ]
                * 3,
                "date_sent": [1640995200000 + i * 600000 for i in range(3)],  # 10 minutes apart
            }
        )

        long_style_analysis = profiler.analyze_messaging_style(long_df, "2")
        assert long_style_analysis["avg_message_length"] > 150
