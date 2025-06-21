"""
Unit tests for conversation analyzer module
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.core.conversation_analyzer import ConversationAnalyzer


@pytest.mark.unit
class TestConversationAnalyzer:
    """Test conversation analysis functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create conversation analyzer instance"""
        return ConversationAnalyzer()

    @pytest.fixture
    def sample_conversation_df(self):
        """Create sample conversation data"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        messages = []

        # Create a conversation with back-and-forth
        for i in range(10):
            sender = 2 if i % 2 == 0 else 3
            messages.append(
                {
                    "_id": i + 1,
                    "thread_id": 1,
                    "from_recipient_id": sender,
                    "to_recipient_id": 3 if sender == 2 else 2,
                    "body": f"Message {i} from sender {sender}",
                    "date_sent": int((base_time + timedelta(minutes=i * 2)).timestamp() * 1000),
                    "date_received": int(
                        (base_time + timedelta(minutes=i * 2, seconds=1)).timestamp() * 1000
                    ),
                }
            )

        return pd.DataFrame(messages)

    def test_create_conversation_windows(self, analyzer, sample_conversation_df):
        """Test conversation window creation"""
        windows = analyzer.create_conversation_windows(sample_conversation_df, window_size=5)

        assert isinstance(windows, list)
        assert len(windows) > 0

        # Check window structure
        first_window = windows[0]
        assert "thread_id" in first_window
        assert "context" in first_window
        assert "response" in first_window
        assert "metadata" in first_window

        # Check metadata
        metadata = first_window["metadata"]
        assert "momentum" in metadata
        assert "avg_time_gap" in metadata
        assert "response_delay" in metadata

    def test_segment_natural_dialogues(self, analyzer, sample_conversation_df):
        """Test dialogue segmentation"""
        episodes = analyzer.segment_natural_dialogues(sample_conversation_df, time_gap_minutes=30)

        assert isinstance(episodes, list)
        assert len(episodes) > 0

        # Check episode structure
        first_episode = episodes[0]
        assert "thread_id" in first_episode
        assert "messages" in first_episode
        assert "metadata" in first_episode

        # Check metadata - actual fields from implementation
        metadata = first_episode["metadata"]
        assert "episode_length" in metadata
        assert "duration_minutes" in metadata
        assert "your_message_count" in metadata
        assert "turn_pattern" in metadata
        assert "initiated_by" in metadata
        assert "ended_by" in metadata

    def test_analyze_personal_texting_style(self, analyzer, sample_conversation_df):
        """Test personal texting style analysis"""
        style = analyzer.analyze_personal_texting_style(sample_conversation_df, your_recipient_id=2)

        assert isinstance(style, dict)
        assert "message_statistics" in style
        assert "communication_patterns" in style
        assert "style_classification" in style

        # Check nested structure
        assert "avg_message_length" in style["message_statistics"]
        assert "message_length_distribution" in style["message_statistics"]
        assert "burst_patterns" in style["communication_patterns"]
        assert "preferred_length" in style["style_classification"]
        assert "total_messages" in style["message_statistics"]

    def test_analyze_message_bursts(self, analyzer, sample_conversation_df):
        """Test message burst analysis"""
        # Create burst messages
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        burst_messages = []

        # Create a burst of 3 messages within 1 minute
        for i in range(3):
            burst_messages.append(
                {
                    "_id": i + 100,
                    "thread_id": 2,
                    "from_recipient_id": 2,
                    "to_recipient_id": 3,
                    "body": f"Burst message {i}",
                    "date_sent": int((base_time + timedelta(seconds=i * 20)).timestamp() * 1000),
                }
            )

        burst_df = pd.DataFrame(burst_messages)
        burst_analysis = analyzer.analyze_message_bursts(burst_df)

        # Check actual return structure
        assert "total_bursts" in burst_analysis
        assert "avg_burst_size" in burst_analysis
        assert "burst_frequency" in burst_analysis
        assert "max_burst_size" in burst_analysis
        assert "avg_burst_duration_seconds" in burst_analysis
        assert "messages_in_bursts" in burst_analysis
        assert "burst_ratio" in burst_analysis

        assert burst_analysis["total_bursts"] >= 1

    def test_model_conversation_roles(self, analyzer, sample_conversation_df):
        """Test conversation role modeling"""
        # First create episodes
        episodes = analyzer.segment_natural_dialogues(sample_conversation_df)

        # Then model roles - returns list of role patterns, not dict
        role_patterns = analyzer.model_conversation_roles(episodes, your_recipient_id=2)

        assert isinstance(role_patterns, list)

        # Check role pattern structure if any patterns found
        if len(role_patterns) > 0:
            first_pattern = role_patterns[0]
            assert "episode_id" in first_pattern
            assert "context" in first_pattern
            assert "response" in first_pattern
            assert "role" in first_pattern
            assert "response_type" in first_pattern
            assert "metadata" in first_pattern

            # Verify role is valid
            assert first_pattern["role"] in [
                "conversation_driver",
                "responsive_participant",
                "active_engager",
                "balanced_conversationalist",
            ]

    def test_empty_conversation_handling(self, analyzer):
        """Test handling of empty conversation data"""
        # Create empty DataFrame with required columns
        empty_df = pd.DataFrame(
            columns=[
                "thread_id",
                "from_recipient_id",
                "to_recipient_id",
                "body",
                "date_sent",
                "date_received",
            ]
        )

        # Create windows from empty DataFrame
        windows = analyzer.create_conversation_windows(empty_df)
        assert windows == []

        # Segment empty DataFrame
        episodes = analyzer.segment_natural_dialogues(empty_df)
        assert episodes == []

        # Analyze style of empty DataFrame
        style = analyzer.analyze_personal_texting_style(empty_df)
        assert style == {}  # Returns empty dict for no messages

    def test_single_message_conversation(self, analyzer):
        """Test handling of single message"""
        single_message_df = pd.DataFrame(
            [
                {
                    "_id": 1,
                    "thread_id": 1,
                    "from_recipient_id": 2,
                    "to_recipient_id": 3,
                    "body": "Single message",
                    "date_sent": int(datetime.now().timestamp() * 1000),
                }
            ]
        )

        # Windows from single message
        windows = analyzer.create_conversation_windows(single_message_df, window_size=3)
        assert len(windows) == 0  # Can't create windows with only one message

        # Episodes from single message - requires at least 2 messages
        episodes = analyzer.segment_natural_dialogues(single_message_df)
        assert len(episodes) == 0  # Single message doesn't create episode

    @pytest.mark.parametrize("window_size", [3, 5, 7, 10])
    def test_different_window_sizes(self, analyzer, sample_conversation_df, window_size):
        """Test window creation with different sizes"""
        windows = analyzer.create_conversation_windows(
            sample_conversation_df, window_size=window_size
        )

        assert isinstance(windows, list)

        # Check that context size matches window size constraints
        if len(windows) > 0:
            max_context = max(len(w["context"]) for w in windows)
            # Context can be at most window_size messages (not window_size - 1)
            assert max_context <= window_size

    def test_analyze_conversational_patterns(self, analyzer, sample_conversation_df):
        """Test conversational pattern analysis"""
        # Create some training data first
        windows = analyzer.create_conversation_windows(sample_conversation_df)

        # Convert to training format with proper structure
        training_data = []
        for window in windows[:5]:  # Use first 5 windows
            training_data.append(
                {
                    "metadata": {
                        "type": "conversation_window",
                        "response_delay": window["metadata"]["response_delay"],
                    },
                    "output": window["response"]["text"],
                    "context": window["context"],
                }
            )

        patterns = analyzer.analyze_conversational_patterns(training_data)

        assert isinstance(patterns, dict)
        assert "dataset_size" in patterns
        assert "type_distribution" in patterns
        assert "response_metrics" in patterns
        assert "message_metrics" in patterns
        assert "role_distribution" in patterns
        assert "burst_analysis" in patterns
        assert "context_analysis" in patterns

    def test_generate_analysis_summary(self, analyzer, sample_conversation_df):
        """Test comprehensive analysis summary generation"""
        summary = analyzer.generate_analysis_summary(sample_conversation_df, your_recipient_id=2)

        assert isinstance(summary, dict)
        assert "dataset_overview" in summary
        assert "personal_style" in summary
        assert "conversation_dynamics" in summary
        assert "temporal_patterns" in summary

        # Check nested structure
        assert "total_messages" in summary["dataset_overview"]
        assert "total_windows" in summary["conversation_dynamics"]
        assert "total_episodes" in summary["conversation_dynamics"]
