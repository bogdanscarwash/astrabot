"""
Unit tests for style analyzer module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.core.style_analyzer import StyleAnalyzer


@pytest.mark.unit
class TestStyleAnalyzer:
    """Test communication style analysis functionality"""
    
    @pytest.fixture
    def analyzer(self):
        """Create style analyzer instance"""
        return StyleAnalyzer()
    
    @pytest.fixture
    def sample_messages_df(self):
        """Create sample messages DataFrame for style analysis"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        messages = []
        
        # Create a burst sequence
        for i in range(3):
            messages.append({
                'body': f'Hey! Message {i} in burst 😊',
                'from_recipient_id': 2,
                'to_recipient_id': 3,
                'thread_id': 1,
                'date_sent': int((base_time + timedelta(seconds=30*i)).timestamp() * 1000)
            })
        
        # Add some spaced out messages
        for i in range(3, 6):
            messages.append({
                'body': f'This is a longer message that contains more text and demonstrates different communication patterns',
                'from_recipient_id': 2,
                'to_recipient_id': 3,
                'thread_id': 1,
                'date_sent': int((base_time + timedelta(minutes=10*i)).timestamp() * 1000)
            })
            
        return pd.DataFrame(messages)
    
    @pytest.fixture
    def recipients_df(self):
        """Create sample recipients DataFrame"""
        return pd.DataFrame([
            {'_id': 2, 'profile_given_name': 'You', 'profile_family_name': 'User'},
            {'_id': 3, 'profile_given_name': 'Friend', 'profile_family_name': 'Name'}
        ])
    
    def test_analyze_emoji_usage(self, analyzer, sample_messages_df):
        """Test emoji usage analysis"""
        emoji_analysis = analyzer.analyze_emoji_usage(sample_messages_df)
        
        assert 'emoji_frequency' in emoji_analysis
        assert 'messages_with_emojis' in emoji_analysis
        
        assert emoji_analysis['emoji_frequency'] > 0
        assert emoji_analysis['messages_with_emojis'] >= 3  # First 3 messages have emojis
    
    def test_analyze_message_bursts(self, analyzer, sample_messages_df):
        """Test burst pattern analysis"""
        burst_analysis = analyzer.analyze_message_bursts(sample_messages_df)
        
        assert 'total_bursts' in burst_analysis
        assert 'avg_burst_size' in burst_analysis
        assert 'burst_frequency' in burst_analysis
        
        assert burst_analysis['total_bursts'] > 0
        assert burst_analysis['avg_burst_size'] >= 2
    
    def test_analyze_timing_patterns(self, analyzer, sample_messages_df):
        """Test timing pattern analysis"""
        timing_analysis = analyzer.analyze_timing_patterns(sample_messages_df)
        
        assert 'hour_distribution' in timing_analysis
        assert 'peak_hours' in timing_analysis
        assert 'activity_classification' in timing_analysis
        
        # Verify we have hour distribution data
        assert len(timing_analysis['hour_distribution']) > 0
    
    def test_classify_communication_style(self, analyzer):
        """Test style classification"""
        style_data = {
            'avg_message_length': 25.5,
            'burst_frequency': 0.7,
            'emoji_frequency': 0.3
        }
        
        style = analyzer.classify_communication_style(style_data)
        
        assert style in ['rapid_burst_chatter', 'lengthy_texter', 'concise_texter', 
                        'expressive_communicator', 'formal_correspondent']
    
    def test_analyze_all_communication_styles(self, analyzer, sample_messages_df, recipients_df):
        """Test comprehensive style analysis"""
        # Need to add more messages for meaningful analysis
        extended_messages = []
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        
        # Create conversations with different recipients
        for recipient_id in [3, 4, 5]:
            for i in range(60):  # Minimum 50 messages required
                extended_messages.append({
                    'body': f'Message {i} to recipient {recipient_id}' + ('😊' if i % 3 == 0 else ''),
                    'from_recipient_id': 2,
                    'to_recipient_id': recipient_id,
                    'thread_id': recipient_id,
                    'date_sent': int((base_time + timedelta(minutes=i)).timestamp() * 1000)
                })
        
        messages_df = pd.DataFrame(extended_messages)
        
        # Add recipients
        recipients_df = pd.DataFrame([
            {'_id': 2, 'profile_given_name': 'You'},
            {'_id': 3, 'profile_given_name': 'Friend1'},
            {'_id': 4, 'profile_given_name': 'Friend2'},
            {'_id': 5, 'profile_given_name': 'Friend3'}
        ])
        
        styles = analyzer.analyze_all_communication_styles(
            messages_df, recipients_df
        )
        
        assert isinstance(styles, dict)
        assert len(styles) > 0
        
        for recipient_id, style_data in styles.items():
            assert 'style_type' in style_data
            assert 'avg_message_length' in style_data
            assert 'burst_patterns' in style_data
            assert 'emoji_usage' in style_data
    
    def test_empty_message_handling(self, analyzer):
        """Test handling of empty messages"""
        empty_df = pd.DataFrame()
        
        # analyze_emoji_usage should handle empty DataFrame
        emoji_analysis = analyzer.analyze_emoji_usage(empty_df)
        assert emoji_analysis['emoji_frequency'] == 0
        assert emoji_analysis['messages_with_emojis'] == 0
    
    def test_single_message_analysis(self, analyzer):
        """Test analysis with single message"""
        single_message_df = pd.DataFrame([{
            'body': 'Hello world! 😊',
            'from_recipient_id': 2,
            'to_recipient_id': 3,
            'thread_id': 1,
            'date_sent': int(datetime.now().timestamp() * 1000)
        }])
        
        emoji_analysis = analyzer.analyze_emoji_usage(single_message_df)
        assert emoji_analysis['messages_with_emojis'] == 1
        
        # Burst analysis with single message
        burst_analysis = analyzer.analyze_message_bursts(single_message_df)
        assert burst_analysis['total_bursts'] == 0  # No burst with single message