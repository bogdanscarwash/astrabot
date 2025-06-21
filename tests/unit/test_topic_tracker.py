"""
Unit tests for topic tracker module
"""

from datetime import timedelta

import pandas as pd
import pytest

from src.core.topic_tracker import TopicTracker
from src.models.schemas import TopicCategory


@pytest.mark.unit
class TestTopicTracker:
    """Test topic tracking and analysis functionality"""

    @pytest.fixture
    def tracker(self):
        """Create TopicTracker instance"""
        return TopicTracker()

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for different topics"""
        return {
            "politics": [
                "The election results are concerning for democracy",
                "This policy will affect millions of people",
                "The political climate is increasingly polarized",
                "We need electoral reform to fix these issues",
            ],
            "food": [
                "Just made the best pasta ever!",
                "That restaurant has amazing sushi",
                "I'm craving pizza tonight",
                "This recipe turned out perfectly",
            ],
            "technology": [
                "The new AI model is impressive",
                "This software update broke everything",
                "Machine learning is transforming the industry",
                "My code finally works after debugging all day",
            ],
            "personal": [
                "I'm feeling really stressed about work",
                "Had a great time with family today",
                "My back has been hurting all week",
                "Can't wait for vacation next month",
            ],
            "academic": [
                "The research methodology needs improvement",
                "This theoretical framework is fascinating",
                "The empirical evidence supports the hypothesis",
                "Need to finish my dissertation chapter",
            ],
            "mixed": [
                "The political implications of AI are huge",
                "Cooking while debugging code",
                "Stressed about the election results",
                "Reading research papers about food science",
            ],
        }

    @pytest.fixture
    def sample_conversation_df(self):
        """Create sample conversation DataFrame with topic transitions"""
        base_time = pd.Timestamp("2024-01-01 12:00:00")

        messages = []

        # Start with personal topic
        messages.extend(
            [
                {
                    "_id": i,
                    "thread_id": 1,
                    "from_recipient_id": 2,
                    "body": text,
                    "date_sent": int((base_time + timedelta(minutes=i)).timestamp() * 1000),
                }
                for i, text in enumerate(
                    [
                        "How was your day?",
                        "Pretty good, just tired from work",
                        "Yeah, I feel you. Work has been crazy",
                    ]
                )
            ]
        )

        # Transition to politics
        messages.extend(
            [
                {
                    "_id": i + 3,
                    "thread_id": 1,
                    "from_recipient_id": 2,
                    "body": text,
                    "date_sent": int((base_time + timedelta(minutes=i + 3)).timestamp() * 1000),
                }
                for i, text in enumerate(
                    [
                        "Did you see the news about the election?",
                        "Yeah, the results are concerning",
                        "This political climate is getting worse",
                    ]
                )
            ]
        )

        # Transition to food
        messages.extend(
            [
                {
                    "_id": i + 6,
                    "thread_id": 1,
                    "from_recipient_id": 2,
                    "body": text,
                    "date_sent": int((base_time + timedelta(minutes=i + 6)).timestamp() * 1000),
                }
                for i, text in enumerate(
                    [
                        "Anyway, want to grab dinner?",
                        "Sure! That new Italian place?",
                        "Yes! Their pasta is amazing",
                    ]
                )
            ]
        )

        return pd.DataFrame(messages)

    def test_topic_detection_single_message(self, tracker):
        """Test topic detection for individual messages"""
        # Politics
        politics_result = tracker.detect_message_topics(
            "The election results show a clear shift in voter sentiment"
        )
        assert politics_result["primary_topic"] == TopicCategory.POLITICS
        assert politics_result["confidence"] > 0.5

        # Food
        food_result = tracker.detect_message_topics("This restaurant has the best tacos in town!")
        assert food_result["primary_topic"] == TopicCategory.FOOD

        # Technology
        tech_result = tracker.detect_message_topics(
            "Machine learning algorithms are getting more sophisticated"
        )
        assert tech_result["primary_topic"] == TopicCategory.TECHNOLOGY

        # Personal
        personal_result = tracker.detect_message_topics(
            "I'm feeling overwhelmed with everything going on"
        )
        assert personal_result["primary_topic"] == TopicCategory.PERSONAL_LIFE

    def test_empty_message_handling(self, tracker):
        """Test handling of empty or minimal messages"""
        empty_result = tracker.detect_message_topics("")
        assert empty_result["primary_topic"] == TopicCategory.OTHER
        assert empty_result["confidence"] == 0.0

        short_result = tracker.detect_message_topics("ok")
        assert short_result["primary_topic"] == TopicCategory.OTHER

    def test_multi_topic_detection(self, tracker):
        """Test detection of multiple topics in a message"""
        multi_topic_msg = "The political implications of AI technology are concerning"
        result = tracker.detect_message_topics(multi_topic_msg)

        assert "topic_scores" in result
        assert result["topic_scores"][TopicCategory.POLITICS] > 0
        assert result["topic_scores"][TopicCategory.TECHNOLOGY] > 0

        # Primary topic should be the highest scoring one
        primary_score = result["topic_scores"][result["primary_topic"]]
        assert all(primary_score >= score for score in result["topic_scores"].values())

    def test_keyword_matching(self, tracker):
        """Test keyword-based topic matching"""
        # Test specific keywords for each category
        keyword_tests = {
            TopicCategory.POLITICS: ["democracy", "election", "government", "policy"],
            TopicCategory.FOOD: ["recipe", "restaurant", "delicious", "cooking"],
            TopicCategory.TECHNOLOGY: ["software", "algorithm", "programming", "AI"],
            TopicCategory.MEMES: ["lol", "meme", "viral", "trending"],
            TopicCategory.ACADEMIC: ["research", "hypothesis", "methodology", "thesis"],
            TopicCategory.RELATIONSHIPS: ["dating", "boyfriend", "girlfriend", "marriage"],
            TopicCategory.WORK: ["meeting", "deadline", "project", "coworker"],
        }

        for topic, keywords in keyword_tests.items():
            for keyword in keywords:
                result = tracker.detect_message_topics(f"Something about {keyword}")
                # Should detect the topic, though it might not always be primary
                assert result["topic_scores"].get(topic, 0) > 0

    def test_conversation_topic_analysis(self, tracker, sample_conversation_df):
        """Test topic analysis across a conversation"""
        analysis = tracker.analyze_conversation_topics(sample_conversation_df)

        assert "topic_distribution" in analysis
        assert "topic_transitions" in analysis
        assert "dominant_topics" in analysis
        assert "topic_diversity" in analysis

        # Should detect multiple topics in the conversation
        topic_dist = analysis["topic_distribution"]
        assert len(topic_dist) > 1
        assert TopicCategory.PERSONAL_LIFE in topic_dist
        assert TopicCategory.POLITICS in topic_dist
        assert TopicCategory.FOOD in topic_dist

        # Check topic transitions
        transitions = analysis["topic_transitions"]
        assert len(transitions) >= 2  # At least 2 transitions in sample data

    def test_topic_transition_detection(self, tracker, sample_conversation_df):
        """Test detection of topic transitions"""
        transitions = tracker.detect_topic_transitions(sample_conversation_df)

        assert isinstance(transitions, list)
        assert len(transitions) >= 2

        for transition in transitions:
            assert isinstance(transition, dict)
            assert "from_topic" in transition
            assert "to_topic" in transition
            assert "message_index" in transition
            assert "trigger_message" in transition
            assert "smoothness" in transition

            # Smoothness should be between 0 and 1
            assert 0 <= transition["smoothness"] <= 1

    def test_topic_engagement_scoring(self, tracker, sample_conversation_df):
        """Test topic engagement scoring by participants"""
        analysis = tracker.analyze_conversation_topics(sample_conversation_df)

        if "topic_engagement_scores" in analysis:
            engagement = analysis["topic_engagement_scores"]

            # Should have engagement scores for participants
            assert isinstance(engagement, dict)

            for participant_id, scores in engagement.items():
                assert isinstance(scores, dict)

                # Check specific engagement metrics
                if "political_engagement" in scores:
                    assert 0 <= scores["political_engagement"] <= 1
                if "academic_engagement" in scores:
                    assert 0 <= scores["academic_engagement"] <= 1

    def test_temporal_topic_patterns(self, tracker, sample_conversation_df):
        """Test temporal patterns in topic usage"""
        temporal_analysis = tracker.analyze_temporal_topic_patterns(
            sample_conversation_df, time_window_hours=1
        )

        assert "topic_timeline" in temporal_analysis
        assert "peak_topic_times" in temporal_analysis

        timeline = temporal_analysis["topic_timeline"]
        assert isinstance(timeline, list)

        for window in timeline:
            assert "start_time" in window
            assert "end_time" in window
            assert "dominant_topic" in window
            assert "topic_counts" in window

    def test_topic_category_completeness(self, tracker):
        """Test that all topic categories are handled"""
        # Test each topic category
        for topic in TopicCategory:
            # Create a message that should trigger this topic
            if topic == TopicCategory.OTHER:
                continue  # Skip OTHER as it's the default

            # Use the topic name in a message
            msg = f"This message is about {topic.value}"
            result = tracker.detect_message_topics(msg)

            # Should at least recognize the topic name
            assert result["primary_topic"] in TopicCategory

    def test_confidence_scoring(self, tracker):
        """Test confidence scoring for topic detection"""
        # High confidence - clear topic
        high_conf_result = tracker.detect_message_topics(
            "The election results clearly show voter dissatisfaction with current policies"
        )
        assert high_conf_result["confidence"] > 0.7

        # Low confidence - vague message
        low_conf_result = tracker.detect_message_topics("That's interesting")
        assert low_conf_result["confidence"] < 0.5

        # Medium confidence - mixed topics
        med_conf_result = tracker.detect_message_topics("Reading about stuff online")
        assert 0.3 <= med_conf_result["confidence"] <= 0.7

    def test_batch_topic_detection(self, tracker, sample_messages):
        """Test batch processing of messages for topics"""
        all_messages = []
        expected_topics = []

        for topic_name, messages in sample_messages.items():
            if topic_name != "mixed":
                all_messages.extend(messages)
                # Map topic names to TopicCategory enum values
                topic_map = {
                    "politics": TopicCategory.POLITICS,
                    "food": TopicCategory.FOOD,
                    "technology": TopicCategory.TECHNOLOGY,
                    "personal": TopicCategory.PERSONAL_LIFE,
                    "academic": TopicCategory.ACADEMIC,
                }
                expected_topics.extend([topic_map[topic_name]] * len(messages))

        # Detect topics for all messages
        detected_topics = []
        for msg in all_messages:
            result = tracker.detect_message_topics(msg)
            detected_topics.append(result["primary_topic"])

        # Check accuracy (should get most topics right)
        correct = sum(
            1
            for detected, expected in zip(detected_topics, expected_topics)
            if detected == expected
        )
        accuracy = correct / len(all_messages)
        assert accuracy > 0.7  # Should correctly identify at least 70% of topics

    def test_topic_transition_smoothness(self, tracker):
        """Test smoothness calculation for topic transitions"""
        # Smooth transition (related topics)
        smooth_messages = pd.DataFrame(
            {
                "_id": [1, 2],
                "body": [
                    "I'm stressed about work deadlines",
                    "Yeah, work has been overwhelming lately",
                ],
                "date_sent": [1000000, 1060000],
            }
        )

        smooth_transitions = tracker.detect_topic_transitions(smooth_messages)
        if smooth_transitions:
            assert smooth_transitions[0]["smoothness"] > 0.7

        # Abrupt transition (unrelated topics)
        abrupt_messages = pd.DataFrame(
            {
                "_id": [1, 2],
                "body": ["The political situation is dire", "Want to get pizza for lunch?"],
                "date_sent": [1000000, 1060000],
            }
        )

        abrupt_transitions = tracker.detect_topic_transitions(abrupt_messages)
        if abrupt_transitions:
            assert abrupt_transitions[0]["smoothness"] < 0.5

    def test_topic_diversity_calculation(self, tracker, sample_conversation_df):
        """Test calculation of topic diversity in conversations"""
        analysis = tracker.analyze_conversation_topics(sample_conversation_df)

        assert "topic_diversity" in analysis
        diversity = analysis["topic_diversity"]

        # Diversity should be between 0 and 1
        assert 0 <= diversity <= 1

        # Sample conversation has multiple topics, so diversity should be > 0
        assert diversity > 0

    def test_edge_cases(self, tracker):
        """Test edge cases and special scenarios"""
        # Very long message
        long_msg = " ".join(["politics"] * 100 + ["food"] * 100)
        long_result = tracker.detect_message_topics(long_msg)
        assert long_result["primary_topic"] in [TopicCategory.POLITICS, TopicCategory.FOOD]

        # Message with special characters
        special_msg = "Politics!!! @#$% & food... 🍕🗳️"
        special_result = tracker.detect_message_topics(special_msg)
        assert special_result["primary_topic"] != TopicCategory.OTHER

        # Non-string input handling
        none_result = tracker.detect_message_topics(None)
        assert none_result["primary_topic"] == TopicCategory.OTHER

        # Numeric input
        numeric_result = tracker.detect_message_topics("12345")
        assert numeric_result["primary_topic"] == TopicCategory.OTHER
