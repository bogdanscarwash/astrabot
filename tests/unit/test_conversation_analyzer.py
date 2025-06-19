"""
Unit tests for conversation analyzer module
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.conversation_analyzer import ConversationAnalyzer


@pytest.mark.unit
class TestConversationAnalyzer:
    """Test conversation analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return ConversationAnalyzer()
    
    @pytest.fixture
    def sample_conversation_df(self):
        """Create sample conversation data"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        return pd.DataFrame({
            'sender': ['You', 'Alice', 'You', 'You', 'Alice', 'You'],
            'text': [
                'Hey!',
                'Hi there!',
                'How are you doing?',
                'Been working on some code',
                'Nice! What kind of project?',
                'A chatbot using LLMs'
            ],
            'timestamp': [
                base_time,
                base_time + timedelta(minutes=1),
                base_time + timedelta(minutes=2),
                base_time + timedelta(minutes=2, seconds=30),
                base_time + timedelta(minutes=5),
                base_time + timedelta(minutes=6)
            ]
        })
    
    def test_identify_conversation_windows(self, analyzer, sample_conversation_df):
        """Test conversation window identification"""
        windows = analyzer.identify_conversation_windows(
            sample_conversation_df, 
            window_minutes=10
        )
        
        assert len(windows) == 1  # All messages within 10 minute window
        assert len(windows[0]['messages']) == 6
        assert windows[0]['participants'] == {'You', 'Alice'}
    
    def test_multiple_conversation_windows(self, analyzer):
        """Test identifying multiple conversation windows"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        df = pd.DataFrame({
            'sender': ['You', 'Alice', 'You', 'Bob'],
            'text': ['Message 1', 'Reply 1', 'Message 2', 'Much later'],
            'timestamp': [
                base_time,
                base_time + timedelta(minutes=1),
                base_time + timedelta(minutes=2),
                base_time + timedelta(hours=2)  # 2 hours later
            ]
        })
        
        windows = analyzer.identify_conversation_windows(df, window_minutes=30)
        
        assert len(windows) == 2
        assert len(windows[0]['messages']) == 3
        assert len(windows[1]['messages']) == 1
    
    def test_analyze_response_patterns(self, analyzer, sample_conversation_df):
        """Test response pattern analysis"""
        patterns = analyzer.analyze_response_patterns(sample_conversation_df)
        
        assert 'avg_response_time' in patterns
        assert 'response_rate' in patterns
        assert 'initiation_rate' in patterns
        assert patterns['response_rate'] > 0
    
    def test_extract_conversation_topics(self, analyzer, sample_conversation_df):
        """Test topic extraction from conversations"""
        topics = analyzer.extract_conversation_topics(sample_conversation_df)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        # Should identify topics like 'code', 'chatbot', 'LLMs'
    
    def test_calculate_engagement_metrics(self, analyzer, sample_conversation_df):
        """Test engagement metrics calculation"""
        metrics = analyzer.calculate_engagement_metrics(sample_conversation_df)
        
        assert 'message_frequency' in metrics
        assert 'avg_message_length' in metrics
        assert 'conversation_balance' in metrics
        assert metrics['avg_message_length'] > 0
    
    def test_identify_conversation_dynamics(self, analyzer, sample_conversation_df):
        """Test conversation dynamics identification"""
        dynamics = analyzer.identify_conversation_dynamics(sample_conversation_df)
        
        assert 'turn_taking_pattern' in dynamics
        assert 'dominant_speaker' in dynamics
        assert 'interaction_style' in dynamics
    
    def test_empty_conversation_handling(self, analyzer):
        """Test handling of empty conversation data"""
        empty_df = pd.DataFrame(columns=['sender', 'text', 'timestamp'])
        
        windows = analyzer.identify_conversation_windows(empty_df)
        assert len(windows) == 0
        
        patterns = analyzer.analyze_response_patterns(empty_df)
        assert patterns['response_rate'] == 0
    
    def test_single_message_conversation(self, analyzer):
        """Test handling of single message conversation"""
        df = pd.DataFrame({
            'sender': ['You'],
            'text': ['Hello?'],
            'timestamp': [datetime.now()]
        })
        
        windows = analyzer.identify_conversation_windows(df)
        assert len(windows) == 1
        assert len(windows[0]['messages']) == 1
    
    @pytest.mark.parametrize("window_size", [5, 15, 30, 60])
    def test_different_window_sizes(self, analyzer, sample_conversation_df, window_size):
        """Test conversation windows with different sizes"""
        windows = analyzer.identify_conversation_windows(
            sample_conversation_df,
            window_minutes=window_size
        )
        
        assert len(windows) >= 1
        for window in windows:
            assert 'messages' in window
            assert 'window_start' in window
            assert 'window_end' in window