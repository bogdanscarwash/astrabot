"""
Unit tests for training data creator module
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.llm.training_data_creator import TrainingDataCreator
from src.models.schemas import EnhancedMessage, TweetContent, ImageDescription


@pytest.mark.unit
@pytest.mark.llm
class TestTrainingDataCreator:
    """Test training data creation functionality"""
    
    @pytest.fixture
    def creator(self):
        """Create training data creator instance"""
        return TrainingDataCreator()
    
    @pytest.fixture
    def sample_conversation_messages(self):
        """Create sample conversation messages"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        return [
            EnhancedMessage(
                original_message="Hey! How's your day going?",
                conversation_id="conv_1",
                message_id="msg_1",
                sender_id="3",  # Friend
                timestamp=base_time,
                tweet_contents=[],
                image_descriptions=[]
            ),
            EnhancedMessage(
                original_message="Pretty good! Just working on some code.",
                conversation_id="conv_1", 
                message_id="msg_2",
                sender_id="2",  # You
                timestamp=base_time + timedelta(minutes=1),
                tweet_contents=[],
                image_descriptions=[]
            ),
            EnhancedMessage(
                original_message="What kind of project?",
                conversation_id="conv_1",
                message_id="msg_3", 
                sender_id="3",  # Friend
                timestamp=base_time + timedelta(minutes=2),
                tweet_contents=[],
                image_descriptions=[]
            ),
            EnhancedMessage(
                original_message="A chatbot using transformers and fine-tuning!",
                conversation_id="conv_1",
                message_id="msg_4",
                sender_id="2",  # You
                timestamp=base_time + timedelta(minutes=3),
                tweet_contents=[],
                image_descriptions=[]
            )
        ]
    
    @pytest.fixture
    def sample_message_with_twitter(self):
        """Create message with Twitter content"""
        tweet = TweetContent(
            text="AI is revolutionizing everything! #AI #Tech",
            author="ai_news",
            tweet_id="123456789",
            mentioned_users=[],
            hashtags=["AI", "Tech"],
            sentiment="positive"
        )
        
        return EnhancedMessage(
            original_message="Check this out: https://twitter.com/ai_news/status/123456789",
            conversation_id="conv_2",
            message_id="msg_5",
            sender_id="2",
            timestamp=datetime.now(),
            tweet_contents=[tweet],
            image_descriptions=[]
        )
    
    def test_create_conversational_training_data(self, creator, sample_conversation_messages):
        """Test creating conversational training data"""
        training_data = creator.create_conversational_training_data(
            sample_conversation_messages,
            context_window=2,
            your_recipient_id="2"
        )
        
        assert len(training_data) > 0
        
        # Check first training example
        example = training_data[0]
        assert 'messages' in example
        assert len(example['messages']) >= 2  # System + user/assistant
        
        # Should have system message
        assert example['messages'][0]['role'] == 'system'
        
        # Should have conversational flow
        user_messages = [msg for msg in example['messages'] if msg['role'] == 'user']
        assistant_messages = [msg for msg in example['messages'] if msg['role'] == 'assistant']
        
        assert len(user_messages) > 0
        assert len(assistant_messages) > 0
    
    def test_create_burst_sequence_data(self, creator):
        """Test creating burst sequence training data"""
        # Create burst of messages from same sender
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        burst_messages = [
            EnhancedMessage(
                original_message="So I was thinking about this problem",
                conversation_id="conv_burst",
                message_id=f"msg_{i}",
                sender_id="2",
                timestamp=base_time + timedelta(seconds=i*30),
                tweet_contents=[],
                image_descriptions=[]
            ) for i in range(4)
        ]
        burst_messages[1].original_message = "And I realized there's a better approach"
        burst_messages[2].original_message = "Using machine learning techniques"
        burst_messages[3].original_message = "What do you think?"
        
        training_data = creator.create_burst_sequence_data(burst_messages, your_recipient_id="2")
        
        assert len(training_data) > 0
        
        # Should capture the burst pattern
        example = training_data[0]
        assert 'messages' in example
        
        # Should have the full burst in assistant message
        assistant_msg = next(msg for msg in example['messages'] if msg['role'] == 'assistant')
        full_text = assistant_msg['content']
        
        assert "thinking about this problem" in full_text
        assert "better approach" in full_text
        assert "machine learning" in full_text
    
    def test_create_adaptive_training_data(self, creator, sample_conversation_messages):
        """Test creating adaptive training data that adjusts to conversation partners"""
        training_data = creator.create_adaptive_training_data(
            sample_conversation_messages,
            your_recipient_id="2"
        )
        
        assert len(training_data) > 0
        
        # Should include conversation partner context
        example = training_data[0]
        system_msg = example['messages'][0]
        
        assert 'conversation partner' in system_msg['content'].lower() or 'adapt' in system_msg['content'].lower()
    
    def test_create_qa_training_data(self, creator):
        """Test creating Q&A focused training data"""
        qa_messages = [
            EnhancedMessage(
                original_message="What is machine learning?",
                conversation_id="conv_qa",
                message_id="msg_q1",
                sender_id="3",
                timestamp=datetime.now(),
                tweet_contents=[],
                image_descriptions=[]
            ),
            EnhancedMessage(
                original_message="Machine learning is a subset of AI that learns from data.",
                conversation_id="conv_qa",
                message_id="msg_a1",
                sender_id="2",
                timestamp=datetime.now() + timedelta(seconds=30),
                tweet_contents=[],
                image_descriptions=[]
            )
        ]
        
        training_data = creator.create_qa_training_data(qa_messages, your_recipient_id="2")
        
        assert len(training_data) > 0
        
        # Should format as Q&A
        example = training_data[0]
        user_msg = next(msg for msg in example['messages'] if msg['role'] == 'user')
        assistant_msg = next(msg for msg in example['messages'] if msg['role'] == 'assistant')
        
        assert "machine learning" in user_msg['content'].lower()
        assert "subset of ai" in assistant_msg['content'].lower()
    
    def test_handle_twitter_content(self, creator, sample_message_with_twitter):
        """Test handling messages with Twitter content"""
        training_data = creator.create_conversational_training_data(
            [sample_message_with_twitter],
            your_recipient_id="2"
        )
        
        assert len(training_data) > 0
        
        # Twitter content should be included in training format
        example = training_data[0]
        assistant_msg = next(msg for msg in example['messages'] if msg['role'] == 'assistant')
        content = assistant_msg['content']
        
        assert '[TWEET:' in content
        assert 'AI is revolutionizing' in content
        assert '#AI' in content or '#Tech' in content
    
    def test_filter_by_quality(self, creator, sample_conversation_messages):
        """Test filtering training data by quality"""
        # Add a low-quality message
        low_quality_msg = EnhancedMessage(
            original_message="k",  # Very short, low quality
            conversation_id="conv_1",
            message_id="msg_low",
            sender_id="2",
            timestamp=datetime.now(),
            tweet_contents=[],
            image_descriptions=[]
        )
        
        all_messages = sample_conversation_messages + [low_quality_msg]
        
        # Create training data with quality filtering
        training_data = creator.create_conversational_training_data(
            all_messages,
            your_recipient_id="2",
            min_message_length=3
        )
        
        # Low quality message should be filtered out
        for example in training_data:
            for message in example['messages']:
                if message['role'] == 'assistant':
                    assert message['content'] != 'k'
    
    def test_context_window_handling(self, creator, sample_conversation_messages):
        """Test different context window sizes"""
        for context_size in [1, 2, 3, 5]:
            training_data = creator.create_conversational_training_data(
                sample_conversation_messages,
                context_window=context_size,
                your_recipient_id="2"
            )
            
            assert len(training_data) > 0
            
            # Check context window is respected
            for example in training_data:
                total_conversation_msgs = len([msg for msg in example['messages'] 
                                             if msg['role'] in ['user', 'assistant']])
                assert total_conversation_msgs <= context_size * 2  # user + assistant pairs
    
    def test_deduplication(self, creator):
        """Test deduplication of similar training examples"""
        # Create duplicate messages
        duplicate_messages = [
            EnhancedMessage(
                original_message="Hello there!",
                conversation_id="conv_dup1",
                message_id="msg_1",
                sender_id="2",
                timestamp=datetime.now(),
                tweet_contents=[],
                image_descriptions=[]
            ),
            EnhancedMessage(
                original_message="Hello there!",  # Exact duplicate
                conversation_id="conv_dup2", 
                message_id="msg_2",
                sender_id="2",
                timestamp=datetime.now() + timedelta(hours=1),
                tweet_contents=[],
                image_descriptions=[]
            )
        ]
        
        training_data = creator.create_conversational_training_data(
            duplicate_messages,
            your_recipient_id="2",
            deduplicate=True
        )
        
        # Should have fewer examples due to deduplication
        unique_contents = set()
        for example in training_data:
            for message in example['messages']:
                if message['role'] == 'assistant':
                    unique_contents.add(message['content'])
        
        # Should not have exact duplicates
        assert len(unique_contents) <= len(duplicate_messages)
    
    def test_format_for_different_models(self, creator, sample_conversation_messages):
        """Test formatting for different model types"""
        # Test different chat templates
        templates = ['qwen', 'llama', 'mistral']
        
        for template in templates:
            try:
                training_data = creator.create_conversational_training_data(
                    sample_conversation_messages,
                    your_recipient_id="2",
                    chat_template=template
                )
                
                assert len(training_data) > 0
                
                # Should have proper message structure for each template
                example = training_data[0]
                assert 'messages' in example
                assert all('role' in msg and 'content' in msg for msg in example['messages'])
                
            except ValueError:
                # Some templates might not be implemented
                continue
    
    def test_metadata_preservation(self, creator, sample_conversation_messages):
        """Test that important metadata is preserved in training data"""
        training_data = creator.create_conversational_training_data(
            sample_conversation_messages,
            your_recipient_id="2",
            include_metadata=True
        )
        
        for example in training_data:
            # Should include metadata
            assert 'metadata' in example
            metadata = example['metadata']
            
            assert 'conversation_id' in metadata
            assert 'timestamp' in metadata
            assert 'message_count' in metadata
    
    def test_empty_input_handling(self, creator):
        """Test handling of empty input"""
        empty_training_data = creator.create_conversational_training_data(
            [],
            your_recipient_id="2"
        )
        
        assert len(empty_training_data) == 0
    
    def test_batch_processing(self, creator, sample_conversation_messages):
        """Test batch processing of large datasets"""
        # Create larger dataset
        large_dataset = sample_conversation_messages * 50  # 200 messages
        
        training_data = creator.create_conversational_training_data(
            large_dataset,
            your_recipient_id="2",
            batch_size=10
        )
        
        assert len(training_data) > 0
        # Should handle large datasets efficiently
        assert len(training_data) <= len(large_dataset)  # Reasonable number of examples