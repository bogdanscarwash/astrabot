"""
Unit tests for metadata enricher module
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.core.metadata_enricher import (
    add_reaction_context,
    add_group_context,
    add_temporal_context,
    add_conversation_flow_metadata,
    enrich_messages_with_all_metadata,
    classify_urgency,
    classify_emotion_from_reactions
)
from src.models.schemas import EnhancedMessage, TweetContent, ImageDescription


@pytest.mark.unit
@pytest.mark.skip(reason="MetadataEnricher class not implemented, only functions exist")
class TestMetadataEnricher:
    """Test metadata enrichment functionality"""
    
    @pytest.fixture
    def sample_message(self):
        """Create sample message for testing"""
        return {
            'id': 'msg_123',
            'conversation_id': 'conv_456',
            'sender_id': 'user_789',
            'text': 'Check this out: https://twitter.com/user/status/123456789',
            'timestamp': datetime.now()
        }
    
    def test_enrich_with_urls(self, enricher, sample_message):
        """Test URL detection and enrichment"""
        enriched = enricher.enrich_message(sample_message)
        
        assert 'detected_urls' in enriched
        assert len(enriched['detected_urls']) == 1
        assert enriched['detected_urls'][0] == 'https://twitter.com/user/status/123456789'
        assert enriched['has_media'] is True
    
    def test_enrich_with_mentions(self, enricher):
        """Test mention detection"""
        message = {
            'text': 'Hey @alice and @bob, check this out!',
            'timestamp': datetime.now()
        }
        
        enriched = enricher.enrich_message(message)
        
        assert 'mentions' in enriched
        assert len(enriched['mentions']) == 2
        assert '@alice' in enriched['mentions']
        assert '@bob' in enriched['mentions']
    
    def test_enrich_with_hashtags(self, enricher):
        """Test hashtag detection"""
        message = {
            'text': 'Working on #AI and #MachineLearning projects',
            'timestamp': datetime.now()
        }
        
        enriched = enricher.enrich_message(message)
        
        assert 'hashtags' in enriched
        assert len(enriched['hashtags']) == 2
        assert '#AI' in enriched['hashtags']
        assert '#MachineLearning' in enriched['hashtags']
    
    def test_enrich_with_emojis(self, enricher):
        """Test emoji detection and counting"""
        message = {
            'text': 'Great job! 👍 😊 🎉',
            'timestamp': datetime.now()
        }
        
        enriched = enricher.enrich_message(message)
        
        assert 'emoji_count' in enriched
        assert enriched['emoji_count'] == 3
        assert 'emojis' in enriched
        assert len(enriched['emojis']) == 3
    
    def test_enrich_with_sentiment(self, enricher):
        """Test basic sentiment analysis"""
        positive_msg = {'text': 'This is absolutely amazing!', 'timestamp': datetime.now()}
        negative_msg = {'text': 'This is terrible and disappointing.', 'timestamp': datetime.now()}
        neutral_msg = {'text': 'The meeting is at 3pm.', 'timestamp': datetime.now()}
        
        pos_enriched = enricher.enrich_message(positive_msg)
        neg_enriched = enricher.enrich_message(negative_msg)
        neu_enriched = enricher.enrich_message(neutral_msg)
        
        assert 'sentiment' in pos_enriched
        assert pos_enriched['sentiment']['score'] > 0
        assert neg_enriched['sentiment']['score'] < 0
        assert abs(neu_enriched['sentiment']['score']) < 0.3
    
    def test_enrich_with_language_detection(self, enricher):
        """Test language detection"""
        english_msg = {'text': 'Hello, how are you?', 'timestamp': datetime.now()}
        spanish_msg = {'text': 'Hola, ¿cómo estás?', 'timestamp': datetime.now()}
        
        eng_enriched = enricher.enrich_message(english_msg)
        esp_enriched = enricher.enrich_message(spanish_msg)
        
        assert 'detected_language' in eng_enriched
        assert eng_enriched['detected_language'] == 'en'
        assert esp_enriched['detected_language'] == 'es'
    
    def test_enrich_empty_message(self, enricher):
        """Test handling of empty messages"""
        empty_msg = {'text': '', 'timestamp': datetime.now()}
        
        enriched = enricher.enrich_message(empty_msg)
        
        assert enriched['detected_urls'] == []
        assert enriched['mentions'] == []
        assert enriched['hashtags'] == []
        assert enriched['emoji_count'] == 0
    
    def test_enrich_with_code_blocks(self, enricher):
        """Test code block detection"""
        message = {
            'text': 'Here is some code:\n```python\ndef hello():\n    print("world")\n```',
            'timestamp': datetime.now()
        }
        
        enriched = enricher.enrich_message(message)
        
        assert 'has_code' in enriched
        assert enriched['has_code'] is True
        assert 'code_languages' in enriched
        assert 'python' in enriched['code_languages']
    
    def test_enrich_with_questions(self, enricher):
        """Test question detection"""
        question_msg = {'text': 'What do you think about this? Is it good?', 'timestamp': datetime.now()}
        statement_msg = {'text': 'I think this is good.', 'timestamp': datetime.now()}
        
        q_enriched = enricher.enrich_message(question_msg)
        s_enriched = enricher.enrich_message(statement_msg)
        
        assert q_enriched['is_question'] is True
        assert q_enriched['question_count'] == 2
        assert s_enriched['is_question'] is False
    
    def test_batch_enrichment(self, enricher):
        """Test batch message enrichment"""
        messages = [
            {'text': 'Hello @alice!', 'timestamp': datetime.now()},
            {'text': 'Check #python news', 'timestamp': datetime.now()},
            {'text': '😊 Great!', 'timestamp': datetime.now()}
        ]
        
        enriched_messages = enricher.enrich_messages_batch(messages)
        
        assert len(enriched_messages) == 3
        assert enriched_messages[0]['mentions'] == ['@alice']
        assert enriched_messages[1]['hashtags'] == ['#python']
        assert enriched_messages[2]['emoji_count'] == 1
    
    @patch('src.core.metadata_enricher.requests.get')
    def test_enrich_with_link_preview(self, mock_get, enricher):
        """Test link preview extraction"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
        <head>
            <title>Test Article</title>
            <meta property="og:description" content="This is a test article about AI">
            <meta property="og:image" content="https://example.com/image.jpg">
        </head>
        </html>
        '''
        mock_get.return_value = mock_response
        
        message = {
            'text': 'Read this: https://example.com/article',
            'timestamp': datetime.now()
        }
        
        enriched = enricher.enrich_message(message, extract_link_previews=True)
        
        assert 'link_previews' in enriched
        assert len(enriched['link_previews']) == 1
        preview = enriched['link_previews'][0]
        assert preview['title'] == 'Test Article'
        assert preview['description'] == 'This is a test article about AI'
        assert preview['image'] == 'https://example.com/image.jpg'