"""
Unit tests for Twitter extractor module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.conversation_processor import TwitterExtractor
from src.models.schemas import TweetContent, ImageDescription


@pytest.mark.unit
@pytest.mark.twitter
class TestTwitterExtractor:
    """Test Twitter/X content extraction functionality"""
    
    @pytest.fixture
    def extractor(self):
        """Create TwitterExtractor instance"""
        return TwitterExtractor()
    
    @pytest.fixture
    def sample_tweet_html(self):
        """Sample Nitter HTML response"""
        return """
        <html>
        <body>
            <div class="tweet-content">
                <p>This is a test tweet with #hashtags and @mentions! 🚀</p>
            </div>
            <div class="tweet-stats">
                <span class="tweet-date">Jan 1, 2024 · 12:00 PM UTC</span>
            </div>
            <div class="fullname">
                <a href="/testuser">Test User</a>
            </div>
            <div class="username">@testuser</div>
            <a class="still-image" href="/pic/media%2Ftest.jpg?format=jpg">
                <img src="/pic/media%2Ftest.jpg"/>
            </a>
            <a class="still-image" href="/pic/media%2Ftest2.png?format=png">
                <img src="/pic/media%2Ftest2.png"/>
            </a>
        </body>
        </html>
        """
    
    def test_extract_tweet_id(self, extractor):
        """Test tweet ID extraction from URLs"""
        test_urls = [
            'https://twitter.com/user/status/123456789',
            'https://x.com/user/status/987654321',
            'https://twitter.com/user/status/123456789?s=20',
            'https://x.com/user/status/987654321/photo/1'
        ]
        
        expected_ids = ['123456789', '987654321', '123456789', '987654321']
        
        for url, expected_id in zip(test_urls, expected_ids):
            tweet_id = extractor.extract_tweet_id(url)
            assert tweet_id == expected_id
    
    def test_invalid_urls(self, extractor):
        """Test handling of invalid URLs"""
        invalid_urls = [
            'https://facebook.com/post/123',
            'https://instagram.com/p/abc123',
            'not_a_url',
            'https://twitter.com/user',  # No status
            ''
        ]
        
        for url in invalid_urls:
            tweet_id = extractor.extract_tweet_id(url)
            assert tweet_id is None
    
    @patch('src.core.conversation_processor.requests.get')
    def test_extract_tweet_content(self, mock_get, extractor, sample_tweet_html):
        """Test tweet content extraction"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_tweet_html
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/testuser/status/123456789'
        tweet_content = extractor.extract_tweet_content(url)
        
        assert isinstance(tweet_content, TweetContent)
        assert tweet_content.tweet_id == '123456789'
        assert tweet_content.text == 'This is a test tweet with #hashtags and @mentions! 🚀'
        assert tweet_content.author == 'testuser'
        assert 'hashtags' in tweet_content.hashtags
        assert 'mentions' in tweet_content.mentioned_users
    
    @patch('src.core.conversation_processor.requests.get')
    def test_extract_tweet_images(self, mock_get, extractor, sample_tweet_html):
        """Test tweet image extraction"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = sample_tweet_html
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/testuser/status/123456789'
        images = extractor.extract_tweet_images(url)
        
        assert len(images) == 2
        assert 'pbs.twimg.com' in images[0]
        assert 'pbs.twimg.com' in images[1]
        assert images[0].endswith('.jpg')
        assert images[1].endswith('.png')
    
    @patch('src.core.conversation_processor.requests.get')
    def test_nitter_fallback(self, mock_get, extractor):
        """Test Nitter instance fallback"""
        # First call fails, second succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 503
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.text = '<div class="tweet-content">Success!</div>'
        
        mock_get.side_effect = [mock_response_fail, mock_response_success]
        
        url = 'https://twitter.com/user/status/123'
        tweet_content = extractor.extract_tweet_content(url)
        
        assert tweet_content is not None
        assert tweet_content.text == 'Success!'
        assert mock_get.call_count == 2
    
    @patch('src.core.conversation_processor.requests.get')
    def test_all_nitter_instances_fail(self, mock_get, extractor):
        """Test handling when all Nitter instances fail"""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/user/status/123'
        tweet_content = extractor.extract_tweet_content(url)
        
        assert tweet_content is None
        # Should try all configured Nitter instances
        assert mock_get.call_count >= 2
    
    def test_parse_hashtags(self, extractor):
        """Test hashtag parsing"""
        text = 'This is a #test tweet with #MultipleHashtags and #numbers123'
        hashtags = extractor.parse_hashtags(text)
        
        assert len(hashtags) == 3
        assert 'test' in hashtags
        assert 'MultipleHashtags' in hashtags
        assert 'numbers123' in hashtags
    
    def test_parse_mentions(self, extractor):
        """Test mention parsing"""
        text = 'Hey @alice and @bob_smith, check out @user123!'
        mentions = extractor.parse_mentions(text)
        
        assert len(mentions) == 3
        assert 'alice' in mentions
        assert 'bob_smith' in mentions
        assert 'user123' in mentions
    
    def test_analyze_sentiment(self, extractor):
        """Test sentiment analysis"""
        positive_text = 'This is absolutely amazing! Love it! 🎉'
        negative_text = 'This is terrible and disappointing. Hate it.'
        neutral_text = 'The meeting is scheduled for 3pm today.'
        
        pos_sentiment = extractor.analyze_sentiment(positive_text)
        neg_sentiment = extractor.analyze_sentiment(negative_text)
        neu_sentiment = extractor.analyze_sentiment(neutral_text)
        
        assert pos_sentiment == 'positive'
        assert neg_sentiment == 'negative'
        assert neu_sentiment == 'neutral'
    
    @patch('src.core.conversation_processor.requests.get')
    def test_caching(self, mock_get, extractor):
        """Test response caching"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<div class="tweet-content">Cached content</div>'
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/user/status/123'
        
        # First call
        tweet1 = extractor.extract_tweet_content(url)
        # Second call (should use cache)
        tweet2 = extractor.extract_tweet_content(url)
        
        assert tweet1.text == tweet2.text
        # Should only make one HTTP request due to caching
        assert mock_get.call_count == 1
    
    def test_convert_nitter_to_twitter_image_url(self, extractor):
        """Test Nitter to Twitter image URL conversion"""
        nitter_urls = [
            '/pic/media%2Ftest.jpg?format=jpg',
            '/pic/media%2Fimage.png?format=png',
            '/pic/media%2Fphoto.gif?format=gif'
        ]
        
        for nitter_url in nitter_urls:
            twitter_url = extractor.convert_nitter_to_twitter_image_url(nitter_url)
            assert twitter_url.startswith('https://pbs.twimg.com/media/')
            assert 'test.jpg' in twitter_url or 'image.png' in twitter_url or 'photo.gif' in twitter_url
    
    @patch('src.core.conversation_processor.requests.get')
    def test_thread_detection(self, mock_get, extractor):
        """Test thread/reply detection"""
        thread_html = """
        <div class="timeline-item">
            <div class="tweet-content">This is part of a thread 1/3</div>
            <div class="show-this-thread">Show this thread</div>
        </div>
        """
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = thread_html
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/user/status/123'
        tweet_content = extractor.extract_tweet_content(url)
        
        assert tweet_content.is_thread is True
    
    @patch('src.core.conversation_processor.requests.get')
    def test_retweet_detection(self, mock_get, extractor):
        """Test retweet detection"""
        retweet_html = """
        <div class="timeline-item">
            <div class="retweet-header">Retweeted</div>
            <div class="tweet-content">Original tweet content</div>
        </div>
        """
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = retweet_html
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/user/status/123'
        tweet_content = extractor.extract_tweet_content(url)
        
        assert tweet_content.is_retweet is True
    
    def test_url_validation(self, extractor):
        """Test URL validation"""
        valid_urls = [
            'https://twitter.com/user/status/123',
            'https://x.com/user/status/456',
            'https://mobile.twitter.com/user/status/789'
        ]
        
        invalid_urls = [
            'https://facebook.com/post/123',
            'not-a-url',
            None,
            ''
        ]
        
        for url in valid_urls:
            assert extractor.is_valid_twitter_url(url) is True
        
        for url in invalid_urls:
            assert extractor.is_valid_twitter_url(url) is False
    
    @patch('src.core.conversation_processor.requests.get')
    def test_rate_limiting_handling(self, mock_get, extractor):
        """Test rate limiting handling"""
        mock_response = MagicMock()
        mock_response.status_code = 429  # Too Many Requests
        mock_get.return_value = mock_response
        
        url = 'https://twitter.com/user/status/123'
        tweet_content = extractor.extract_tweet_content(url)
        
        # Should handle rate limiting gracefully
        assert tweet_content is None