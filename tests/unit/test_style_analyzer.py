"""
Unit tests for style analyzer module
"""

import pytest
import pandas as pd
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
    def sample_messages(self):
        """Create sample messages for style analysis"""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        return [
            {
                'sender': 'You',
                'text': 'Hey! How are you doing? 😊',
                'timestamp': base_time
            },
            {
                'sender': 'You', 
                'text': 'BTW, did you see that article about AI?',
                'timestamp': base_time + timedelta(seconds=30)
            },
            {
                'sender': 'You',
                'text': 'lol yeah that was crazy! 🤯 definitely worth reading',
                'timestamp': base_time + timedelta(minutes=1)
            },
            {
                'sender': 'You',
                'text': 'Actually, let me know what you think when you read it.',
                'timestamp': base_time + timedelta(minutes=2)
            },
            {
                'sender': 'You',
                'text': 'The implications for machine learning are fascinating...',
                'timestamp': base_time + timedelta(minutes=3)
            }
        ]
    
    def test_analyze_message_length_patterns(self, analyzer, sample_messages):
        """Test message length pattern analysis"""
        patterns = analyzer.analyze_message_length_patterns(sample_messages)
        
        assert 'avg_length' in patterns
        assert 'median_length' in patterns
        assert 'length_variance' in patterns
        assert 'short_message_ratio' in patterns
        assert 'long_message_ratio' in patterns
        
        assert patterns['avg_length'] > 0
        assert patterns['median_length'] > 0
    
    def test_analyze_emoji_usage(self, analyzer, sample_messages):
        """Test emoji usage analysis"""
        emoji_analysis = analyzer.analyze_emoji_usage(sample_messages)
        
        assert 'emoji_frequency' in emoji_analysis
        assert 'unique_emojis' in emoji_analysis
        assert 'emoji_per_message' in emoji_analysis
        assert 'most_used_emojis' in emoji_analysis
        
        assert emoji_analysis['emoji_frequency'] > 0
        assert len(emoji_analysis['unique_emojis']) >= 2  # 😊 and 🤯
    
    def test_analyze_language_formality(self, analyzer):
        """Test formality analysis"""
        formal_messages = [
            {'text': 'Good morning. I hope you are doing well.', 'sender': 'You'},
            {'text': 'I would like to discuss the project proposal.', 'sender': 'You'},
            {'text': 'Please let me know your thoughts at your convenience.', 'sender': 'You'}
        ]
        
        informal_messages = [
            {'text': 'hey! what\'s up?', 'sender': 'You'},
            {'text': 'lol that\'s hilarious 😂', 'sender': 'You'},
            {'text': 'gonna be there in 5 mins', 'sender': 'You'}
        ]
        
        formal_score = analyzer.analyze_language_formality(formal_messages)
        informal_score = analyzer.analyze_language_formality(informal_messages)
        
        assert formal_score['formality_score'] > informal_score['formality_score']
        assert formal_score['formality_score'] > 0.6
        assert informal_score['formality_score'] < 0.4
    
    def test_analyze_punctuation_patterns(self, analyzer, sample_messages):
        """Test punctuation pattern analysis"""
        punct_analysis = analyzer.analyze_punctuation_patterns(sample_messages)
        
        assert 'exclamation_ratio' in punct_analysis
        assert 'question_ratio' in punct_analysis
        assert 'ellipsis_usage' in punct_analysis
        assert 'capitalization_consistency' in punct_analysis
        
        # Should detect exclamation marks and questions in sample
        assert punct_analysis['exclamation_ratio'] > 0
        assert punct_analysis['question_ratio'] > 0
    
    def test_analyze_abbreviation_usage(self, analyzer, sample_messages):
        """Test abbreviation and slang analysis"""
        abbrev_analysis = analyzer.analyze_abbreviation_usage(sample_messages)
        
        assert 'abbreviation_frequency' in abbrev_analysis
        assert 'common_abbreviations' in abbrev_analysis
        assert 'internet_slang_usage' in abbrev_analysis
        
        # Should detect "BTW" and "lol" in sample messages
        assert abbrev_analysis['abbreviation_frequency'] > 0
        assert any(abbrev in ['BTW', 'lol'] for abbrev in abbrev_analysis['common_abbreviations'])
    
    def test_analyze_burst_patterns(self, analyzer, sample_messages):
        """Test burst messaging pattern analysis"""
        burst_analysis = analyzer.analyze_burst_patterns(sample_messages)
        
        assert 'burst_frequency' in burst_analysis
        assert 'avg_burst_size' in burst_analysis
        assert 'max_burst_size' in burst_analysis
        assert 'burst_intervals' in burst_analysis
        
        # Sample messages form one burst (sent within minutes)
        assert burst_analysis['burst_frequency'] > 0
        assert burst_analysis['max_burst_size'] >= 5
    
    def test_analyze_response_timing(self, analyzer):
        """Test response timing pattern analysis"""
        conversation = [
            {'sender': 'Alice', 'text': 'Hey!', 'timestamp': datetime(2024, 1, 1, 10, 0, 0)},
            {'sender': 'You', 'text': 'Hi there!', 'timestamp': datetime(2024, 1, 1, 10, 0, 30)},  # 30s
            {'sender': 'Alice', 'text': 'How are you?', 'timestamp': datetime(2024, 1, 1, 10, 1, 0)},
            {'sender': 'You', 'text': 'Good!', 'timestamp': datetime(2024, 1, 1, 10, 1, 5)},  # 5s
            {'sender': 'Alice', 'text': 'Great!', 'timestamp': datetime(2024, 1, 1, 10, 2, 0)},
            {'sender': 'You', 'text': 'Yeah!', 'timestamp': datetime(2024, 1, 1, 10, 5, 0)}  # 3 min
        ]
        
        timing_analysis = analyzer.analyze_response_timing(conversation, user_sender='You')
        
        assert 'avg_response_time' in timing_analysis
        assert 'response_time_variance' in timing_analysis
        assert 'quick_response_ratio' in timing_analysis
        assert 'delayed_response_ratio' in timing_analysis
        
        assert timing_analysis['avg_response_time'] > 0
    
    def test_detect_communication_style(self, analyzer, sample_messages):
        """Test overall communication style detection"""
        style_profile = analyzer.detect_communication_style(sample_messages)
        
        assert 'primary_style' in style_profile
        assert 'style_confidence' in style_profile
        assert 'style_characteristics' in style_profile
        
        # Based on sample messages (casual, friendly, uses emojis)
        expected_styles = ['casual', 'friendly', 'enthusiastic', 'informal']
        assert style_profile['primary_style'] in expected_styles
        assert 0 <= style_profile['style_confidence'] <= 1
    
    def test_analyze_vocabulary_complexity(self, analyzer):
        """Test vocabulary complexity analysis"""
        simple_messages = [
            {'text': 'Hi. How are you?', 'sender': 'You'},
            {'text': 'Good. Thanks.', 'sender': 'You'},
            {'text': 'See you later.', 'sender': 'You'}
        ]
        
        complex_messages = [
            {'text': 'The implementation demonstrates sophisticated algorithmic optimization.', 'sender': 'You'},
            {'text': 'Considering the multifaceted implications of this paradigm shift.', 'sender': 'You'},
            {'text': 'The juxtaposition of these methodologies necessitates careful evaluation.', 'sender': 'You'}
        ]
        
        simple_analysis = analyzer.analyze_vocabulary_complexity(simple_messages)
        complex_analysis = analyzer.analyze_vocabulary_complexity(complex_messages)
        
        assert simple_analysis['complexity_score'] < complex_analysis['complexity_score']
        assert 'avg_word_length' in simple_analysis
        assert 'unique_word_ratio' in simple_analysis
        assert 'rare_word_frequency' in simple_analysis
    
    def test_analyze_topic_transitions(self, analyzer):
        """Test topic transition analysis"""
        messages = [
            {'text': 'How was your vacation?', 'sender': 'You', 'timestamp': datetime.now()},
            {'text': 'Tell me about the beaches!', 'sender': 'You', 
             'timestamp': datetime.now() + timedelta(minutes=1)},
            {'text': 'BTW, did you finish the project?', 'sender': 'You', 
             'timestamp': datetime.now() + timedelta(minutes=2)},
            {'text': 'The code review found some issues.', 'sender': 'You', 
             'timestamp': datetime.now() + timedelta(minutes=3)}
        ]
        
        transition_analysis = analyzer.analyze_topic_transitions(messages)
        
        assert 'transition_frequency' in transition_analysis
        assert 'transition_signals' in transition_analysis
        assert 'topic_persistence' in transition_analysis
        
        # Should detect "BTW" as transition signal
        assert any('BTW' in signal for signal in transition_analysis['transition_signals'])
    
    def test_empty_message_handling(self, analyzer):
        """Test handling of empty message lists"""
        empty_analysis = analyzer.detect_communication_style([])
        
        assert empty_analysis['primary_style'] == 'unknown'
        assert empty_analysis['style_confidence'] == 0
    
    def test_single_message_analysis(self, analyzer):
        """Test analysis with single message"""
        single_message = [{'text': 'Hello world!', 'sender': 'You'}]
        
        style_analysis = analyzer.detect_communication_style(single_message)
        
        assert style_analysis['primary_style'] is not None
        assert style_analysis['style_confidence'] >= 0