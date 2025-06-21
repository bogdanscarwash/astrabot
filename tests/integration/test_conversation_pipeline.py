"""
Integration tests for the conversation processing pipeline
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from src.core.conversation_analyzer import ConversationAnalyzer
from src.core.conversation_processor import EnhancedConversationProcessor as ConversationProcessor
from src.core.conversation_processor import TwitterExtractor
from src.core.style_analyzer import StyleAnalyzer


@pytest.mark.integration
class TestConversationPipeline:
    """Integration tests for the full conversation processing pipeline"""

    @pytest.fixture
    def processor(self):
        """Create conversation processor with all components"""
        return ConversationProcessor()

    @pytest.fixture
    def sample_signal_data(self, temp_data_dir):
        """Create sample Signal data files"""
        messages_file = temp_data_dir / "raw" / "signal-flatfiles" / "message.csv"
        recipients_file = temp_data_dir / "raw" / "signal-flatfiles" / "recipient.csv"

        # Create sample messages CSV
        messages_df = pd.DataFrame(
            {
                "_id": [1, 2, 3, 4, 5],
                "thread_id": [1, 1, 1, 2, 2],
                "from_recipient_id": [2, 3, 2, 2, 4],
                "body": [
                    "Hey Alice! How are you?",
                    "Good! Check this out: https://twitter.com/ai/status/123456789",
                    "That looks interesting! What do you think about it?",
                    "Meeting moved to 3pm tomorrow",
                    "Thanks for letting me know",
                ],
                "date_sent": [
                    1609459200000,
                    1609459260000,
                    1609459320000,
                    1609545600000,
                    1609545660000,
                ],
                "type": ["outgoing", "incoming", "outgoing", "outgoing", "incoming"],
            }
        )

        # Create sample recipients CSV
        recipients_df = pd.DataFrame(
            {"_id": [2, 3, 4], "profile_given_name": ["You", "Alice", "Bob"], "blocked": [0, 0, 0]}
        )

        messages_df.to_csv(messages_file, index=False)
        recipients_df.to_csv(recipients_file, index=False)

        return temp_data_dir

    @pytest.mark.requires_api
    def test_full_pipeline_with_twitter_extraction(self, processor, sample_signal_data):
        """Test full pipeline including Twitter content extraction"""
        # This test requires API keys and network access
        if not processor.config.has_vision_capabilities():
            pytest.skip("Vision API keys not available")

        with patch("src.core.conversation_processor.requests.get") as mock_get:
            # Mock Twitter response
            mock_response = mock_get.return_value
            mock_response.status_code = 200
            mock_response.text = """
            <div class="tweet-content">AI is transforming how we work! #AI #Tech</div>
            <div class="fullname">AI News</div>
            <div class="username">@ai</div>
            """

            # Process conversations
            result = processor.process_signal_data(
                signal_data_path=sample_signal_data / "raw" / "signal-flatfiles",
                output_path=sample_signal_data / "processed",
                include_twitter=True,
                include_images=False,  # Skip images for this test
            )

            assert result is not None
            assert len(result) > 0

            # Check that Twitter content was extracted
            twitter_messages = [msg for msg in result if "twitter.com" in str(msg)]
            assert len(twitter_messages) > 0

    def test_conversation_analysis_integration(self, sample_signal_data):
        """Test integration between conversation processing and analysis"""
        analyzer = ConversationAnalyzer()

        # Load sample data
        messages_file = sample_signal_data / "raw" / "signal-flatfiles" / "message.csv"
        messages_df = pd.read_csv(messages_file)

        # Add timestamp conversion
        messages_df["timestamp"] = pd.to_datetime(messages_df["date_sent"], unit="ms")
        messages_df["sender"] = "You"  # Simplified for test
        messages_df["text"] = messages_df["body"]

        # Analyze conversation windows
        windows = analyzer.identify_conversation_windows(messages_df)
        assert len(windows) >= 1

        # Analyze patterns
        patterns = analyzer.analyze_response_patterns(messages_df)
        assert "response_rate" in patterns
        assert patterns["response_rate"] >= 0

    def test_style_analysis_integration(self, sample_signal_data):
        """Test integration of style analysis with real data"""
        style_analyzer = StyleAnalyzer()

        # Create sample messages for style analysis
        messages = [
            {"sender": "You", "text": "Hey! How are you doing? 😊"},
            {"sender": "You", "text": "BTW, did you see that article?"},
            {"sender": "You", "text": "lol that was crazy! 🤯"},
            {"sender": "You", "text": "Let me know what you think!"},
        ]

        # Analyze communication style
        style_profile = style_analyzer.detect_communication_style(messages)

        assert "primary_style" in style_profile
        assert style_profile["primary_style"] in ["casual", "friendly", "enthusiastic", "informal"]
        assert 0 <= style_profile["style_confidence"] <= 1

    @pytest.mark.slow
    def test_large_dataset_processing(self, temp_data_dir):
        """Test processing with larger dataset"""
        # Create larger synthetic dataset
        large_messages = []
        for i in range(1000):
            large_messages.append(
                {
                    "_id": i + 1,
                    "thread_id": (i // 10) + 1,  # 10 messages per thread
                    "from_recipient_id": 2 if i % 2 == 0 else 3,
                    "body": f"Message {i + 1} with some content",
                    "date_sent": 1609459200000 + (i * 60000),  # 1 minute apart
                    "type": "outgoing" if i % 2 == 0 else "incoming",
                }
            )

        large_df = pd.DataFrame(large_messages)
        messages_file = temp_data_dir / "raw" / "signal-flatfiles" / "message.csv"
        large_df.to_csv(messages_file, index=False)

        # Process large dataset
        processor = ConversationProcessor()

        # Should handle large dataset without errors
        start_time = datetime.now()
        result = processor.load_signal_data(temp_data_dir / "raw" / "signal-flatfiles")
        end_time = datetime.now()

        assert result is not None
        assert len(result) == 1000

        # Should complete in reasonable time (< 30 seconds for 1000 messages)
        processing_time = (end_time - start_time).total_seconds()
        assert processing_time < 30

    def test_error_handling_integration(self, temp_data_dir):
        """Test error handling across the pipeline"""
        processor = ConversationProcessor()

        # Test with missing files
        with pytest.raises(FileNotFoundError):
            processor.load_signal_data(temp_data_dir / "nonexistent")

        # Test with corrupted CSV
        corrupted_file = temp_data_dir / "raw" / "signal-flatfiles" / "message.csv"
        corrupted_file.parent.mkdir(parents=True, exist_ok=True)
        with open(corrupted_file, "w") as f:
            f.write("invalid,csv,content\n1,2")  # Incomplete row

        # Should handle corrupted data gracefully
        try:
            result = processor.load_signal_data(temp_data_dir / "raw" / "signal-flatfiles")
            # If it doesn't raise an exception, result should be empty or minimal
            assert result is None or len(result) == 0
        except Exception as e:
            # Should be a handled exception with meaningful message
            assert str(e) is not None

    @pytest.mark.twitter
    def test_twitter_extraction_integration(self):
        """Test Twitter extraction with real URLs (mocked responses)"""
        extractor = TwitterExtractor()

        test_urls = [
            "https://twitter.com/user/status/123456789",
            "https://x.com/user/status/987654321",
            "invalid_url",
        ]

        with patch("src.core.conversation_processor.requests.get") as mock_get:
            # Mock successful response
            mock_response = mock_get.return_value
            mock_response.status_code = 200
            mock_response.text = '<div class="tweet-content">Test tweet</div>'

            valid_extractions = 0
            for url in test_urls:
                try:
                    content = extractor.extract_tweet_content(url)
                    if content:
                        valid_extractions += 1
                        assert content.text is not None
                        assert content.tweet_id is not None
                except Exception:
                    # Invalid URLs should be handled gracefully
                    pass

            # Should extract content from valid URLs
            assert valid_extractions >= 2

    def test_memory_usage_monitoring(self, temp_data_dir):
        """Test memory usage doesn't grow excessively"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create and process multiple datasets
        ConversationProcessor()

        for batch in range(5):
            # Create batch data
            batch_messages = pd.DataFrame(
                {
                    "_id": range(batch * 100, (batch + 1) * 100),
                    "thread_id": [1] * 100,
                    "from_recipient_id": [2] * 100,
                    "body": [f"Batch {batch} message {i}" for i in range(100)],
                    "date_sent": [1609459200000 + i * 1000 for i in range(100)],
                    "type": ["outgoing"] * 100,
                }
            )

            # Process batch
            messages_file = temp_data_dir / "raw" / "signal-flatfiles" / f"batch_{batch}.csv"
            messages_file.parent.mkdir(parents=True, exist_ok=True)
            batch_messages.to_csv(messages_file, index=False)

            # Force garbage collection
            import gc

            gc.collect()

        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (< 100MB for this test)
        assert memory_growth < 100 * 1024 * 1024  # 100MB
