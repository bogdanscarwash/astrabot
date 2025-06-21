"""Tests for privacy filter functionality.

This module tests the comprehensive privacy protection system following
the Test-Driven Development (TDD) approach established in the project.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.privacy_filter import (
    PrivacyFilter,
    PrivacyLevel,
    PrivacyPattern,
    classify_signal_data_privacy,
    mask_sensitive_data,
)


class TestPrivacyLevel:
    """Test privacy level enumeration."""

    def test_privacy_levels_exist(self):
        """Test that all required privacy levels are defined."""
        assert PrivacyLevel.CRITICAL.value == "critical"
        assert PrivacyLevel.HIGH.value == "high"
        assert PrivacyLevel.MEDIUM.value == "medium"
        assert PrivacyLevel.LOW.value == "low"

    def test_privacy_level_ordering(self):
        """Test that privacy levels can be compared appropriately."""
        levels = [PrivacyLevel.LOW, PrivacyLevel.MEDIUM, PrivacyLevel.HIGH, PrivacyLevel.CRITICAL]
        level_values = ["low", "medium", "high", "critical"]

        for i, level in enumerate(levels):
            assert level.value == level_values[i]


class TestPrivacyPattern:
    """Test privacy pattern data structure."""

    def test_privacy_pattern_creation(self):
        """Test creating a privacy pattern."""
        pattern = PrivacyPattern(
            name="test_pattern",
            pattern=r"\btest\b",
            replacement="[TEST]",
            privacy_level=PrivacyLevel.CRITICAL,
            description="Test pattern",
        )

        assert pattern.name == "test_pattern"
        assert pattern.pattern == r"\btest\b"
        assert pattern.replacement == "[TEST]"
        assert pattern.privacy_level == PrivacyLevel.CRITICAL
        assert pattern.description == "Test pattern"


class TestPrivacyFilter:
    """Test the main PrivacyFilter class."""

    @pytest.fixture
    def privacy_filter(self):
        """Create a privacy filter instance for testing."""
        return PrivacyFilter(enable_nlp=False)

    @pytest.fixture
    def privacy_filter_with_nlp(self):
        """Create a privacy filter with NLP enabled (mocked)."""
        with patch("src.utils.privacy_filter.spacy") as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            # Mock entity detection
            mock_doc = MagicMock()
            mock_entity = MagicMock()
            mock_entity.label_ = "PERSON"
            mock_entity.start_char = 0
            mock_entity.end_char = 4
            mock_doc.ents = [mock_entity]
            mock_nlp.return_value = mock_doc

            return PrivacyFilter(enable_nlp=True)

    def test_init_without_nlp(self):
        """Test initialization without NLP."""
        pf = PrivacyFilter(enable_nlp=False)
        assert pf.enable_nlp is False
        assert pf.nlp is None
        assert len(pf.patterns) > 0

    def test_init_with_nlp_missing(self):
        """Test initialization with NLP when spaCy is missing."""
        with patch("src.utils.privacy_filter.spacy", None):
            pf = PrivacyFilter(enable_nlp=True)
            assert pf.enable_nlp is False
            assert pf.nlp is None

    def test_patterns_initialization(self, privacy_filter):
        """Test that privacy patterns are properly initialized."""
        pattern_names = [p.name for p in privacy_filter.patterns]

        # Check critical patterns exist
        assert "phone_us" in pattern_names
        assert "email" in pattern_names
        assert "ssn" in pattern_names
        assert "credit_card" in pattern_names
        assert "api_key_openai" in pattern_names
        assert "api_key_anthropic" in pattern_names

        # Check high privacy patterns exist
        assert "ip_address" in pattern_names
        assert "mac_address" in pattern_names


class TestDataClassification:
    """Test data classification functionality."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_classify_empty_text(self, privacy_filter):
        """Test classification of empty or None text."""
        assert privacy_filter.classify_data("") == PrivacyLevel.LOW
        assert privacy_filter.classify_data(None) == PrivacyLevel.LOW
        assert privacy_filter.classify_data(123) == PrivacyLevel.LOW

    def test_classify_phone_numbers(self, privacy_filter):
        """Test classification of phone numbers as critical."""
        phone_numbers = [
            "555-123-4567",
            "(555) 123-4567",
            "555.123.4567",
            "15551234567",
            "+1 555 123 4567",
        ]

        for phone in phone_numbers:
            assert privacy_filter.classify_data(phone) == PrivacyLevel.CRITICAL

    def test_classify_emails(self, privacy_filter):
        """Test classification of email addresses as critical."""
        emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "firstname.lastname@company.org",
        ]

        for email in emails:
            assert privacy_filter.classify_data(email) == PrivacyLevel.CRITICAL

    def test_classify_ssn(self, privacy_filter):
        """Test classification of SSN as critical."""
        ssns = ["123-45-6789", "987-65-4321"]

        for ssn in ssns:
            assert privacy_filter.classify_data(ssn) == PrivacyLevel.CRITICAL

    def test_classify_credit_cards(self, privacy_filter):
        """Test classification of credit card numbers as critical."""
        credit_cards = ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111"]

        for cc in credit_cards:
            assert privacy_filter.classify_data(cc) == PrivacyLevel.CRITICAL

    def test_classify_api_keys(self, privacy_filter):
        """Test classification of API keys as critical."""
        api_keys = [
            "sk-1234567890abcdef1234567890abcdef",
            "sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz",
        ]

        for key in api_keys:
            assert privacy_filter.classify_data(key) == PrivacyLevel.CRITICAL

    def test_classify_ip_addresses(self, privacy_filter):
        """Test classification of IP addresses as high privacy."""
        ip_addresses = ["192.168.1.1", "10.0.0.1", "172.16.0.1", "8.8.8.8"]

        for ip in ip_addresses:
            assert privacy_filter.classify_data(ip) == PrivacyLevel.HIGH

    def test_classify_regular_text(self, privacy_filter):
        """Test classification of regular text as medium privacy."""
        regular_texts = [
            "This is a normal conversation message",
            "Hey, how are you doing today?",
            "Let's meet at the coffee shop",
        ]

        for text in regular_texts:
            assert privacy_filter.classify_data(text) == PrivacyLevel.MEDIUM

    def test_classify_short_text(self, privacy_filter):
        """Test classification of short text as low privacy."""
        short_texts = ["ok", "yes", "no", "123", "test"]

        for text in short_texts:
            assert privacy_filter.classify_data(text) == PrivacyLevel.LOW

    def test_classify_with_context(self, privacy_filter):
        """Test classification with context hints."""
        text = "secret123"

        # Without context
        privacy_filter.classify_data(text)

        # With sensitive context
        sensitive_level = privacy_filter.classify_data(text, context="password field")

        assert sensitive_level == PrivacyLevel.HIGH


class TestAnonymization:
    """Test text anonymization functionality."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_anonymize_empty_text(self, privacy_filter):
        """Test anonymization of empty or None text."""
        assert privacy_filter.anonymize_text("") == ""
        assert privacy_filter.anonymize_text(None) is None
        assert privacy_filter.anonymize_text(123) == 123

    def test_anonymize_phone_numbers(self, privacy_filter):
        """Test anonymization of phone numbers."""
        test_cases = [
            ("Call me at 555-123-4567", "Call me at [PHONE]"),
            ("My number is (555) 123-4567", "My number is [PHONE]"),
            ("Text 555.123.4567 later", "Text [PHONE] later"),
            ("International: +1-555-123-4567", "International: [PHONE]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected, f"Failed for: {original}"

    def test_anonymize_emails(self, privacy_filter):
        """Test anonymization of email addresses."""
        test_cases = [
            ("Email me at user@example.com", "Email me at [EMAIL]"),
            ("Contact: test.email+tag@domain.co.uk", "Contact: [EMAIL]"),
            ("Send to firstname.lastname@company.org", "Send to [EMAIL]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected, f"Failed for: {original}"

    def test_anonymize_ssn(self, privacy_filter):
        """Test anonymization of Social Security Numbers."""
        test_cases = [
            ("SSN: 123-45-6789", "SSN: [SSN]"),
            ("My social is 987-65-4321", "My social is [SSN]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected, f"Failed for: {original}"

    def test_anonymize_credit_cards(self, privacy_filter):
        """Test anonymization of credit card numbers."""
        test_cases = [
            ("Card: 4111 1111 1111 1111", "Card: [CREDIT_CARD]"),
            ("Use 4111-1111-1111-1111", "Use [CREDIT_CARD]"),
            ("Number: 4111111111111111", "Number: [CREDIT_CARD]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected, f"Failed for: {original}"

    def test_anonymize_api_keys(self, privacy_filter):
        """Test anonymization of API keys."""
        test_cases = [
            ("Key: sk-1234567890abcdef1234567890abcdef", "Key: [API_KEY]"),
            ("sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwx234yz", "[API_KEY]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected, f"Failed for: {original}"

    def test_anonymize_multiple_patterns(self, privacy_filter):
        """Test anonymization of text with multiple sensitive patterns."""
        original = "Contact John at john@example.com or call 555-123-4567"
        result = privacy_filter.anonymize_text(original)

        # Should anonymize both email and phone
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "john@example.com" not in result
        assert "555-123-4567" not in result

    def test_anonymize_preserves_structure(self, privacy_filter):
        """Test that anonymization preserves text structure."""
        original = "Hi! My email is user@test.com and phone is 555-1234. Thanks!"
        result = privacy_filter.anonymize_text(original)

        expected = "Hi! My email is [EMAIL] and phone is [PHONE]. Thanks!"
        assert result == expected

    def test_anonymize_case_insensitive(self, privacy_filter):
        """Test that anonymization works regardless of case."""
        test_cases = [
            ("EMAIL: USER@EXAMPLE.COM", "EMAIL: [EMAIL]"),
            ("email: user@example.com", "email: [EMAIL]"),
        ]

        for original, expected in test_cases:
            result = privacy_filter.anonymize_text(original)
            assert result == expected


class TestNLPAnonymization:
    """Test NLP-based anonymization (mocked)."""

    def test_anonymize_with_nlp_person(self):
        """Test NLP-based person name anonymization."""
        with patch("src.utils.privacy_filter.spacy") as mock_spacy:
            # Setup mock
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            mock_doc = MagicMock()
            mock_entity = MagicMock()
            mock_entity.label_ = "PERSON"
            mock_entity.start_char = 0
            mock_entity.end_char = 4
            mock_doc.ents = [mock_entity]
            mock_nlp.return_value = mock_doc

            # Test
            pf = PrivacyFilter(enable_nlp=True)
            result = pf.anonymize_text("John went to the store")

            assert "[PERSON]" in result

    def test_anonymize_with_nlp_organization(self):
        """Test NLP-based organization anonymization."""
        with patch("src.utils.privacy_filter.spacy") as mock_spacy:
            # Setup mock
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            mock_doc = MagicMock()
            mock_entity = MagicMock()
            mock_entity.label_ = "ORG"
            mock_entity.start_char = 0
            mock_entity.end_char = 5
            mock_doc.ents = [mock_entity]
            mock_nlp.return_value = mock_doc

            # Test
            pf = PrivacyFilter(enable_nlp=True)
            result = pf.anonymize_text("Apple released new products")

            assert "[ORGANIZATION]" in result


class TestHashIdentifier:
    """Test identifier hashing functionality."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_hash_identifier_basic(self, privacy_filter):
        """Test basic identifier hashing."""
        identifier = "555-123-4567"
        hashed = privacy_filter.hash_identifier(identifier)

        assert len(hashed) == 8
        assert hashed != identifier
        assert hashed.isalnum()

    def test_hash_identifier_consistency(self, privacy_filter):
        """Test that same identifier produces same hash."""
        identifier = "user@example.com"
        hash1 = privacy_filter.hash_identifier(identifier)
        hash2 = privacy_filter.hash_identifier(identifier)

        assert hash1 == hash2

    def test_hash_identifier_different_salt(self, privacy_filter):
        """Test that different salts produce different hashes."""
        identifier = "555-123-4567"
        hash1 = privacy_filter.hash_identifier(identifier, salt="salt1")
        hash2 = privacy_filter.hash_identifier(identifier, salt="salt2")

        assert hash1 != hash2

    def test_hash_identifier_empty(self, privacy_filter):
        """Test hashing empty identifier."""
        assert privacy_filter.hash_identifier("") == ""
        assert privacy_filter.hash_identifier(None) == ""


class TestDataFrameFiltering:
    """Test DataFrame privacy filtering."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    @pytest.fixture
    def sample_signal_data(self):
        """Create sample Signal data for testing."""
        return pd.DataFrame(
            {
                "_id": [1, 2, 3],
                "thread_id": [101, 102, 103],
                "body": [
                    "Hi! Call me at 555-123-4567",
                    "My email is user@example.com",
                    "Regular message without PII",
                ],
                "e164": ["+15551234567", "+15551234568", "+15551234569"],
                "profile_given_name": ["John", "Jane", "Bob"],
                "profile_family_name": ["Doe", "Smith", "Johnson"],
                "date_sent": [1234567890, 1234567891, 1234567892],
                "read": [1, 0, 1],
            }
        )

    def test_filter_dataframe_critical_data(self, privacy_filter, sample_signal_data):
        """Test filtering DataFrame with critical privacy data."""
        field_mapping = {
            "body": PrivacyLevel.CRITICAL,
            "e164": PrivacyLevel.CRITICAL,
            "profile_given_name": PrivacyLevel.CRITICAL,
            "profile_family_name": PrivacyLevel.CRITICAL,
        }

        filtered = privacy_filter.filter_dataframe(sample_signal_data, field_mapping)

        # Check that phone numbers are masked in messages
        assert "555-123-4567" not in filtered["body"].iloc[0]
        assert "[PHONE]" in filtered["body"].iloc[0]

        # Check that emails are masked in messages
        assert "user@example.com" not in filtered["body"].iloc[1]
        assert "[EMAIL]" in filtered["body"].iloc[1]

        # Check that phone numbers are hashed
        assert all(len(val) == 8 for val in filtered["e164"])
        assert all(val.isalnum() for val in filtered["e164"])

        # Check that names are redacted
        assert all(val == "[REDACTED]" for val in filtered["profile_given_name"])
        assert all(val == "[REDACTED]" for val in filtered["profile_family_name"])

    def test_filter_dataframe_preserves_structure(self, privacy_filter, sample_signal_data):
        """Test that DataFrame structure is preserved after filtering."""
        field_mapping = classify_signal_data_privacy(sample_signal_data)
        filtered = privacy_filter.filter_dataframe(sample_signal_data, field_mapping)

        assert len(filtered) == len(sample_signal_data)
        assert list(filtered.columns) == list(sample_signal_data.columns)
        assert filtered["_id"].equals(sample_signal_data["_id"])
        assert filtered["thread_id"].equals(sample_signal_data["thread_id"])


class TestDifferentialPrivacy:
    """Test differential privacy implementation."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    @pytest.fixture
    def numeric_data(self):
        """Create sample numeric data for testing."""
        return pd.DataFrame(
            {
                "_id": [1, 2, 3, 4, 5],
                "thread_id": [101, 102, 103, 104, 105],
                "message_count": [10, 15, 8, 20, 12],
                "avg_length": [45.2, 38.7, 52.1, 41.9, 48.3],
                "read": [1, 0, 1, 1, 0],
            }
        )

    def test_differential_privacy_adds_noise(self, privacy_filter, numeric_data):
        """Test that differential privacy adds noise to numeric columns."""
        np.random.seed(42)  # For reproducible tests

        noisy_data = privacy_filter.add_differential_privacy(numeric_data, epsilon=1.0)

        # Check that data structure is preserved
        assert len(noisy_data) == len(numeric_data)
        assert list(noisy_data.columns) == list(numeric_data.columns)

        # Check that IDs are not modified
        assert noisy_data["_id"].equals(numeric_data["_id"])
        assert noisy_data["thread_id"].equals(numeric_data["thread_id"])

        # Check that noise was added to other numeric columns
        assert not noisy_data["message_count"].equals(numeric_data["message_count"])
        assert not noisy_data["avg_length"].equals(numeric_data["avg_length"])
        assert not noisy_data["read"].equals(numeric_data["read"])

    def test_differential_privacy_epsilon_effect(self, privacy_filter, numeric_data):
        """Test that smaller epsilon adds more noise."""
        np.random.seed(42)

        low_privacy = privacy_filter.add_differential_privacy(numeric_data, epsilon=0.1)
        np.random.seed(42)
        high_privacy = privacy_filter.add_differential_privacy(numeric_data, epsilon=10.0)

        # Lower epsilon should result in more noise (higher variance)
        low_variance = np.var(low_privacy["message_count"] - numeric_data["message_count"])
        high_variance = np.var(high_privacy["message_count"] - numeric_data["message_count"])

        assert low_variance > high_variance


class TestValidation:
    """Test anonymization validation functionality."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_validate_anonymization_effective(self, privacy_filter):
        """Test validation of effective anonymization."""
        original = "Contact me at user@example.com or 555-123-4567"
        anonymized = privacy_filter.anonymize_text(original)

        validation = privacy_filter.validate_anonymization(original, anonymized)

        assert validation["is_anonymized"] is True
        assert len(validation["patterns_found"]) > 0
        assert len(validation["potential_leaks"]) == 0
        assert validation["anonymization_ratio"] > 0

    def test_validate_anonymization_ineffective(self, privacy_filter):
        """Test validation when anonymization is ineffective."""
        original = "Contact me at user@example.com"
        anonymized = original  # No anonymization applied

        validation = privacy_filter.validate_anonymization(original, anonymized)

        assert validation["is_anonymized"] is False
        assert validation["anonymization_ratio"] == 0.0

    def test_validate_anonymization_partial_leak(self, privacy_filter):
        """Test validation when there are potential leaks."""
        original = "Email: user@example.com and backup@test.com"
        # Manually create partially anonymized text
        anonymized = "Email: [EMAIL] and backup@test.com"

        validation = privacy_filter.validate_anonymization(original, anonymized)

        assert validation["is_anonymized"] is True
        assert len(validation["potential_leaks"]) > 0
        assert "backup@test.com" in validation["potential_leaks"]


class TestPrivacyReport:
    """Test privacy report generation."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    @pytest.fixture
    def mixed_data(self):
        """Create mixed data with various privacy levels."""
        return pd.DataFrame(
            {
                "safe_id": [1, 2, 3],
                "phone_numbers": ["555-123-4567", "555-987-6543", "regular text"],
                "emails": ["user@test.com", "normal message", "another@example.org"],
                "regular_text": ["Hello", "How are you?", "Good morning"],
                "numbers": [10, 20, 30],
            }
        )

    def test_create_privacy_report_structure(self, privacy_filter, mixed_data):
        """Test that privacy report has correct structure."""
        report = privacy_filter.create_privacy_report(mixed_data)

        required_keys = [
            "total_records",
            "columns_analyzed",
            "privacy_classifications",
            "sensitive_data_found",
            "recommendations",
        ]

        for key in required_keys:
            assert key in report

        assert report["total_records"] == len(mixed_data)
        assert set(report["columns_analyzed"]) == set(mixed_data.columns)

    def test_create_privacy_report_classifications(self, privacy_filter, mixed_data):
        """Test privacy level classifications in report."""
        report = privacy_filter.create_privacy_report(mixed_data)

        # Phone number and email columns should be classified as critical
        assert report["privacy_classifications"]["phone_numbers"] == "critical"
        assert report["privacy_classifications"]["emails"] == "critical"

        # Regular text should be medium
        assert report["privacy_classifications"]["regular_text"] == "medium"

    def test_create_privacy_report_recommendations(self, privacy_filter, mixed_data):
        """Test that recommendations are generated for sensitive data."""
        report = privacy_filter.create_privacy_report(mixed_data)

        recommendations = report["recommendations"]
        assert len(recommendations) > 0

        # Should recommend anonymization for critical data
        phone_rec = any("phone_numbers" in rec for rec in recommendations)
        email_rec = any("emails" in rec for rec in recommendations)

        assert phone_rec or email_rec


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""

    def test_mask_sensitive_data_function(self):
        """Test the standalone mask_sensitive_data function."""
        text = "Contact me at user@example.com or 555-123-4567"
        result = mask_sensitive_data(text)

        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "user@example.com" not in result
        assert "555-123-4567" not in result

    def test_classify_signal_data_privacy_function(self):
        """Test the Signal data privacy classification function."""
        sample_df = pd.DataFrame(
            {
                "_id": [1, 2],
                "body": ["message1", "message2"],
                "e164": ["+15551234567", "+15551234568"],
                "profile_given_name": ["John", "Jane"],
                "date_sent": [1234567890, 1234567891],
                "thread_id": [101, 102],
            }
        )

        classification = classify_signal_data_privacy(sample_df)

        # Critical data
        assert classification["body"] == PrivacyLevel.CRITICAL
        assert classification["e164"] == PrivacyLevel.CRITICAL
        assert classification["profile_given_name"] == PrivacyLevel.CRITICAL

        # Medium data
        assert classification["date_sent"] == PrivacyLevel.MEDIUM

        # Low data
        assert classification["_id"] == PrivacyLevel.LOW
        assert classification["thread_id"] == PrivacyLevel.LOW


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_empty_dataframe(self, privacy_filter):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        field_mapping = {}

        result = privacy_filter.filter_dataframe(empty_df, field_mapping)
        assert len(result) == 0

    def test_missing_columns_in_mapping(self, privacy_filter):
        """Test handling when mapping references non-existent columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        field_mapping = {"nonexistent_col": PrivacyLevel.CRITICAL}

        # Should not raise error
        result = privacy_filter.filter_dataframe(df, field_mapping)
        assert result.equals(df)

    def test_none_values_in_dataframe(self, privacy_filter):
        """Test handling of None/NaN values in DataFrame."""
        df = pd.DataFrame(
            {"body": ["message1", None, "message3"], "e164": ["+15551234567", None, "+15551234569"]}
        )
        field_mapping = {"body": PrivacyLevel.CRITICAL, "e164": PrivacyLevel.CRITICAL}

        result = privacy_filter.filter_dataframe(df, field_mapping)

        # Should handle None values gracefully
        assert pd.isna(result["body"].iloc[1])
        assert pd.isna(result["e164"].iloc[1])

    def test_unicode_text_handling(self, privacy_filter):
        """Test handling of Unicode text."""
        unicode_texts = [
            "Contact: üser@exämple.com",
            "Phone: 555-123-4567 с русским текстом",
            "日本語のテキスト user@test.com",
        ]

        for text in unicode_texts:
            result = privacy_filter.anonymize_text(text)
            assert "[EMAIL]" in result or "user@test.com" not in result


class TestPerformance:
    """Test performance characteristics of privacy filter."""

    @pytest.fixture
    def privacy_filter(self):
        return PrivacyFilter(enable_nlp=False)

    def test_large_text_performance(self, privacy_filter):
        """Test performance with large text blocks."""
        # Create large text with embedded sensitive data
        large_text = "Regular text. " * 1000 + " Contact user@example.com " + "More text. " * 1000

        # Should complete without timeout
        result = privacy_filter.anonymize_text(large_text)
        assert "[EMAIL]" in result

    def test_many_patterns_performance(self, privacy_filter):
        """Test performance with text containing many sensitive patterns."""
        # Text with multiple types of sensitive data
        text = """
        Contact info:
        Phone: 555-123-4567
        Email: user@example.com
        SSN: 123-45-6789
        Credit Card: 4111 1111 1111 1111
        API Key: sk-1234567890abcdef1234567890abcdef
        IP: 192.168.1.1
        """

        result = privacy_filter.anonymize_text(text)

        # All patterns should be anonymized
        sensitive_patterns = [
            "555-123-4567",
            "user@example.com",
            "123-45-6789",
            "4111 1111 1111 1111",
            "sk-1234567890abcdef1234567890abcdef",
            "192.168.1.1",
        ]

        for pattern in sensitive_patterns:
            assert pattern not in result


@pytest.mark.integration
class TestPrivacyFilterIntegration:
    """Integration tests for privacy filter with realistic data."""

    def test_realistic_signal_conversation(self):
        """Test with realistic Signal conversation data."""
        conversation_data = pd.DataFrame(
            {
                "_id": range(1, 6),
                "thread_id": [101] * 5,
                "from_recipient_id": [2, 3, 2, 3, 2],
                "to_recipient_id": [3, 2, 3, 2, 3],
                "body": [
                    "Hey! How's it going?",
                    "Good! BTW my new number is 555-123-4567",
                    "Thanks! I'll update my contacts",
                    "Also, email me at john.doe@example.com",
                    "Will do!",
                ],
                "date_sent": [1640995200000 + i * 60000 for i in range(5)],
                "read": [1, 1, 1, 0, 0],
            }
        )

        privacy_filter = PrivacyFilter(enable_nlp=False)
        field_mapping = classify_signal_data_privacy(conversation_data)

        filtered_data = privacy_filter.filter_dataframe(conversation_data, field_mapping)

        # Check that sensitive data is anonymized
        body_text = " ".join(filtered_data["body"].dropna())
        assert "555-123-4567" not in body_text
        assert "john.doe@example.com" not in body_text
        assert "[PHONE]" in body_text
        assert "[EMAIL]" in body_text

        # Check that conversation structure is preserved
        assert len(filtered_data) == len(conversation_data)
        assert filtered_data["thread_id"].equals(conversation_data["thread_id"])

    def test_privacy_pipeline_end_to_end(self):
        """Test complete privacy filtering pipeline."""
        # Simulate processing pipeline
        raw_data = pd.DataFrame(
            {
                "body": [
                    "Hi! My number is 555-123-4567",
                    "Email me at user@test.com",
                    "Regular conversation message",
                    "My SSN is 123-45-6789",
                ],
                "sender_name": ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
                "sender_phone": ["+15551234567", "+15559876543", "+15555551234", "+15552223333"],
            }
        )

        privacy_filter = PrivacyFilter(enable_nlp=False)

        # Step 1: Classify data
        field_mapping = {
            "body": PrivacyLevel.CRITICAL,
            "sender_name": PrivacyLevel.CRITICAL,
            "sender_phone": PrivacyLevel.CRITICAL,
        }

        # Step 2: Filter data
        filtered_data = privacy_filter.filter_dataframe(raw_data, field_mapping)

        # Step 3: Generate privacy report
        report = privacy_filter.create_privacy_report(raw_data)

        # Step 4: Validate results
        validation_results = []
        for i, row in raw_data.iterrows():
            original_body = row["body"]
            filtered_body = filtered_data.iloc[i]["body"]
            validation = privacy_filter.validate_anonymization(original_body, filtered_body)
            validation_results.append(validation)

        # Assertions
        assert all(
            v["is_anonymized"] or v["anonymization_ratio"] == 0.0 for v in validation_results
        )
        assert len(report["recommendations"]) > 0
        assert all(name == "[REDACTED]" for name in filtered_data["sender_name"])
        assert all(len(phone) == 8 and phone.isalnum() for phone in filtered_data["sender_phone"])


if __name__ == "__main__":
    pytest.main([__file__])
