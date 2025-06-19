"""Integration tests for privacy filtering pipeline.

This module tests the complete privacy filtering pipeline with realistic
Signal conversation data, ensuring end-to-end privacy protection.
"""

import pytest
import pandas as pd
import json
import os
from pathlib import Path

from src.utils.privacy_filter import (
    PrivacyFilter,
    PrivacyLevel,
    classify_signal_data_privacy,
    mask_sensitive_data
)


@pytest.mark.integration
class TestPrivacyPipelineIntegration:
    """Test complete privacy filtering pipeline."""
    
    @pytest.fixture
    def sample_sensitive_data(self):
        """Load sample sensitive data from fixtures."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        csv_path = fixtures_dir / "sample_sensitive_data.csv"
        return pd.read_csv(csv_path)
    
    @pytest.fixture
    def privacy_test_cases(self):
        """Load privacy test cases from fixtures."""
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        json_path = fixtures_dir / "privacy_test_cases.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture
    def privacy_filter(self):
        """Create privacy filter instance."""
        return PrivacyFilter(enable_nlp=False)
    
    def test_signal_data_classification_pipeline(self, sample_sensitive_data):
        """Test automatic classification of Signal data."""
        classification = classify_signal_data_privacy(sample_sensitive_data)
        
        # Verify critical data is classified correctly
        assert classification['body'] == PrivacyLevel.CRITICAL
        assert classification['e164'] == PrivacyLevel.CRITICAL
        assert classification['profile_given_name'] == PrivacyLevel.CRITICAL
        assert classification['profile_family_name'] == PrivacyLevel.CRITICAL
        
        # Verify high privacy data
        assert classification['from_recipient_id'] == PrivacyLevel.HIGH
        assert classification['to_recipient_id'] == PrivacyLevel.HIGH
        
        # Verify medium privacy data
        assert classification['date_sent'] == PrivacyLevel.MEDIUM
        assert classification['read'] == PrivacyLevel.MEDIUM
        
        # Verify low privacy data
        assert classification['_id'] == PrivacyLevel.LOW
        assert classification['thread_id'] == PrivacyLevel.LOW
    
    def test_complete_anonymization_pipeline(self, sample_sensitive_data, privacy_filter):
        """Test complete anonymization of sensitive Signal data."""
        # Step 1: Classify data
        field_mapping = classify_signal_data_privacy(sample_sensitive_data)
        
        # Step 2: Apply privacy filtering
        anonymized_data = privacy_filter.filter_dataframe(sample_sensitive_data, field_mapping)
        
        # Step 3: Verify sensitive data is removed
        body_text = " ".join(anonymized_data['body'].dropna())
        
        # Check that specific sensitive data is anonymized
        sensitive_items = [
            "555-123-4567",
            "john.doe@example.com", 
            "j.doe@company.org",
            "123-45-6789",
            "4111 1111 1111 1111",
            "sk-1234567890abcdef1234567890abcdef",
            "192.168.1.100",
            "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        ]
        
        for item in sensitive_items:
            assert item not in body_text, f"Sensitive data '{item}' was not anonymized"
        
        # Check that replacement tokens are present
        anonymization_tokens = ["[PHONE]", "[EMAIL]", "[SSN]", "[CREDIT_CARD]", "[API_KEY]", "[IP_ADDRESS]"]
        token_found = any(token in body_text for token in anonymization_tokens)
        assert token_found, "No anonymization tokens found in processed text"
        
        # Step 4: Verify structure preservation
        assert len(anonymized_data) == len(sample_sensitive_data)
        assert list(anonymized_data.columns) == list(sample_sensitive_data.columns)
        
        # Step 5: Verify non-sensitive data preservation
        assert anonymized_data['_id'].equals(sample_sensitive_data['_id'])
        assert anonymized_data['thread_id'].equals(sample_sensitive_data['thread_id'])
    
    def test_privacy_pattern_validation(self, privacy_test_cases, privacy_filter):
        """Test all privacy patterns with comprehensive test cases."""
        
        for category, test_cases in privacy_test_cases.items():
            for test_case in test_cases:
                input_text = test_case['input']
                expected = test_case['expected']
                should_match = test_case.get('should_match', True)
                
                result = privacy_filter.anonymize_text(input_text)
                
                if should_match:
                    assert result == expected, f"Failed for {category} - {test_case['pattern_type']}: '{input_text}' -> '{result}' != '{expected}'"
                else:
                    # For cases that shouldn't match, result should equal input
                    assert result == input_text, f"False positive for {category} - {test_case['pattern_type']}: '{input_text}' was modified to '{result}'"
    
    def test_privacy_report_generation(self, sample_sensitive_data, privacy_filter):
        """Test comprehensive privacy report generation."""
        report = privacy_filter.create_privacy_report(sample_sensitive_data)
        
        # Verify report structure
        required_keys = [
            'total_records', 'columns_analyzed', 'privacy_classifications',
            'sensitive_data_found', 'recommendations'
        ]
        for key in required_keys:
            assert key in report, f"Missing key '{key}' in privacy report"
        
        # Verify record count
        assert report['total_records'] == len(sample_sensitive_data)
        
        # Verify sensitive data detection
        assert 'body' in report['sensitive_data_found']
        body_stats = report['sensitive_data_found']['body']
        assert body_stats['count'] > 0
        assert body_stats['percentage'] > 0
        
        # Verify recommendations are generated
        assert len(report['recommendations']) > 0
        
        # Check that recommendations mention critical fields
        recommendation_text = " ".join(report['recommendations'])
        assert any(field in recommendation_text for field in ['body', 'e164', 'profile_given_name'])
    
    def test_differential_privacy_integration(self, privacy_filter):
        """Test differential privacy integration with conversation data."""
        # Create sample conversation statistics
        conversation_stats = pd.DataFrame({
            '_id': range(1, 6),
            'thread_id': [101, 102, 103, 104, 105],
            'message_count': [15, 23, 8, 31, 12],
            'avg_message_length': [45.2, 38.7, 52.1, 41.9, 48.3],
            'response_time_avg': [120, 245, 89, 156, 203],
            'emoji_count': [3, 7, 1, 5, 2]
        })
        
        # Apply differential privacy
        private_stats = privacy_filter.add_differential_privacy(conversation_stats, epsilon=1.0)
        
        # Verify structure preservation
        assert len(private_stats) == len(conversation_stats)
        assert list(private_stats.columns) == list(conversation_stats.columns)
        
        # Verify IDs are unchanged
        assert private_stats['_id'].equals(conversation_stats['_id'])
        assert private_stats['thread_id'].equals(conversation_stats['thread_id'])
        
        # Verify noise was added to numeric columns
        numeric_columns = ['message_count', 'avg_message_length', 'response_time_avg', 'emoji_count']
        for col in numeric_columns:
            # Statistical test: noise should make values different
            assert not private_stats[col].equals(conversation_stats[col]), f"No noise detected in column '{col}'"
    
    def test_anonymization_validation_pipeline(self, sample_sensitive_data, privacy_filter):
        """Test validation of anonymization effectiveness."""
        # Process sensitive messages
        sensitive_messages = sample_sensitive_data['body'].dropna()
        validation_results = []
        
        for original_message in sensitive_messages:
            anonymized = privacy_filter.anonymize_text(original_message)
            validation = privacy_filter.validate_anonymization(original_message, anonymized)
            validation_results.append(validation)
        
        # Check validation results
        for i, validation in enumerate(validation_results):
            original = sensitive_messages.iloc[i]
            
            # If sensitive data was detected, anonymization should have occurred
            if validation['patterns_found']:
                assert validation['is_anonymized'], f"Message not anonymized despite patterns found: '{original}'"
                assert len(validation['potential_leaks']) == 0, f"Potential leaks found in: '{original}'"
                
                # Check that all found patterns were fully anonymized
                for pattern_info in validation['patterns_found']:
                    assert pattern_info['fully_anonymized'], f"Pattern '{pattern_info['pattern']}' not fully anonymized in: '{original}'"
    
    def test_conversation_flow_preservation(self, sample_sensitive_data, privacy_filter):
        """Test that conversation flow is preserved after anonymization."""
        field_mapping = classify_signal_data_privacy(sample_sensitive_data)
        anonymized_data = privacy_filter.filter_dataframe(sample_sensitive_data, field_mapping)
        
        # Sort by thread and timestamp to recreate conversation flow
        original_flow = sample_sensitive_data.sort_values(['thread_id', 'date_sent'])
        anonymized_flow = anonymized_data.sort_values(['thread_id', 'date_sent'])
        
        # Verify conversation structure is preserved
        assert len(original_flow) == len(anonymized_flow)
        assert original_flow['thread_id'].equals(anonymized_flow['thread_id'])
        assert original_flow['from_recipient_id'].equals(anonymized_flow['from_recipient_id'])
        assert original_flow['to_recipient_id'].equals(anonymized_flow['to_recipient_id'])
        assert original_flow['date_sent'].equals(anonymized_flow['date_sent'])
        
        # Verify that messages are still readable (not empty after anonymization)
        anonymized_bodies = anonymized_flow['body'].dropna()
        assert all(len(body.strip()) > 0 for body in anonymized_bodies), "Some messages became empty after anonymization"
    
    def test_hash_consistency_across_sessions(self, privacy_filter):
        """Test that identifier hashing is consistent across different sessions."""
        test_identifiers = [
            "+15551234567",
            "user@example.com",
            "john.doe@company.org",
            "+15559876543"
        ]
        
        # Hash identifiers in first session
        first_session_hashes = {id_: privacy_filter.hash_identifier(id_) for id_ in test_identifiers}
        
        # Create new filter instance (simulating new session)
        new_privacy_filter = PrivacyFilter(enable_nlp=False)
        second_session_hashes = {id_: new_privacy_filter.hash_identifier(id_) for id_ in test_identifiers}
        
        # Verify consistency
        for identifier in test_identifiers:
            assert first_session_hashes[identifier] == second_session_hashes[identifier], f"Hash inconsistency for '{identifier}'"
    
    def test_performance_with_large_dataset(self, privacy_filter):
        """Test privacy filter performance with larger datasets."""
        # Create larger dataset (1000 messages)
        large_dataset = []
        message_templates = [
            "Hey! Call me at 555-{:03d}-{:04d}",
            "Email me at user{:03d}@example.com",
            "Regular message without sensitive data #{:d}",
            "My SSN is {:03d}-{:02d}-{:04d}",
            "Credit card: 4111 1111 1111 {:04d}"
        ]
        
        for i in range(1000):
            template = message_templates[i % len(message_templates)]
            if "{:03d}-{:04d}" in template:  # phone
                message = template.format(i % 1000, i % 10000)
            elif "{:03d}@example.com" in template:  # email
                message = template.format(i % 1000)
            elif "#{:d}" in template:  # regular
                message = template.format(i)
            elif "{:03d}-{:02d}-{:04d}" in template:  # SSN
                message = template.format(i % 1000, i % 100, i % 10000)
            elif "1111 {:04d}" in template:  # credit card
                message = template.format(i % 10000)
            
            large_dataset.append({
                '_id': i + 1,
                'thread_id': (i % 50) + 1,
                'body': message,
                'from_recipient_id': (i % 10) + 1,
                'to_recipient_id': ((i + 1) % 10) + 1,
                'date_sent': 1640995200000 + i * 60000
            })
        
        large_df = pd.DataFrame(large_dataset)
        
        # Process large dataset
        field_mapping = {'body': PrivacyLevel.CRITICAL}
        anonymized_large_df = privacy_filter.filter_dataframe(large_df, field_mapping)
        
        # Verify processing completed successfully
        assert len(anonymized_large_df) == len(large_df)
        
        # Spot check anonymization
        sample_bodies = anonymized_large_df['body'].head(20)
        has_anonymization_tokens = any(
            any(token in body for token in ["[PHONE]", "[EMAIL]", "[SSN]", "[CREDIT_CARD]"])
            for body in sample_bodies
        )
        assert has_anonymization_tokens, "No anonymization detected in large dataset sample"
    
    def test_edge_cases_integration(self, privacy_filter):
        """Test edge cases in integrated privacy pipeline."""
        edge_case_data = pd.DataFrame({
            '_id': [1, 2, 3, 4, 5],
            'body': [
                None,  # None value
                "",    # Empty string
                "   ",  # Whitespace only
                "Valid message with email: user@test.com",  # Valid with sensitive data
                "Regular message"  # Valid without sensitive data
            ],
            'sender_name': [
                "John Doe",
                None,
                "",
                "Jane Smith",
                "Bob Johnson"
            ]
        })
        
        field_mapping = {
            'body': PrivacyLevel.CRITICAL,
            'sender_name': PrivacyLevel.CRITICAL
        }
        
        # Should handle edge cases gracefully
        result = privacy_filter.filter_dataframe(edge_case_data, field_mapping)
        
        # Verify structure is preserved
        assert len(result) == len(edge_case_data)
        
        # Verify None values are preserved
        assert pd.isna(result['body'].iloc[0])
        assert pd.isna(result['sender_name'].iloc[1])
        
        # Verify empty strings are preserved
        assert result['body'].iloc[1] == ""
        assert result['sender_name'].iloc[2] == ""
        
        # Verify sensitive data is anonymized
        assert "user@test.com" not in result['body'].iloc[3]
        assert "[EMAIL]" in result['body'].iloc[3]
        
        # Verify names are redacted
        assert result['sender_name'].iloc[0] == '[REDACTED]'
        assert result['sender_name'].iloc[3] == '[REDACTED]'
        assert result['sender_name'].iloc[4] == '[REDACTED]'


@pytest.mark.integration
class TestPrivacyFilterWithRealData:
    """Test privacy filter with realistic conversation patterns."""
    
    def test_realistic_group_conversation(self):
        """Test privacy filtering on a realistic group conversation."""
        group_conversation = pd.DataFrame({
            '_id': range(1, 11),
            'thread_id': [201] * 10,  # Group conversation
            'from_recipient_id': [2, 3, 4, 2, 5, 3, 2, 4, 5, 2],
            'to_recipient_id': [201] * 10,  # Group ID
            'body': [
                "Hey everyone! Planning for the weekend trip",
                "Sounds great! Can everyone share their contact info?",
                "Sure! My number is 555-123-4567",
                "Mine is 555-987-6543. Also email me at john@example.com",
                "I'll send my info privately - don't want to share publicly",
                "Good thinking! BTW, here's the hotel reservation: CONF123456",
                "Thanks! Also, the total cost will be $1,234.56 per person",
                "Perfect. Should I book the rental car with my card 4111 1111 1111 1111?",
                "Yes, but maybe don't share the full number here 😅",
                "Oops! Good point. I'll handle it offline."
            ],
            'date_sent': [1640995200000 + i*300000 for i in range(10)],  # 5 min intervals
            'read': [1] * 10
        })
        
        privacy_filter = PrivacyFilter(enable_nlp=False)
        field_mapping = {'body': PrivacyLevel.CRITICAL}
        
        anonymized_conversation = privacy_filter.filter_dataframe(group_conversation, field_mapping)
        
        # Verify conversation flow is preserved
        assert len(anonymized_conversation) == len(group_conversation)
        
        # Check that sensitive data is anonymized
        body_text = " ".join(anonymized_conversation['body'])
        assert "555-123-4567" not in body_text
        assert "555-987-6543" not in body_text
        assert "john@example.com" not in body_text
        assert "4111 1111 1111 1111" not in body_text
        
        # Check that anonymization tokens are present
        assert "[PHONE]" in body_text
        assert "[EMAIL]" in body_text
        assert "[CREDIT_CARD]" in body_text
        
        # Verify non-sensitive information is preserved
        assert "weekend trip" in body_text
        assert "hotel reservation" in body_text
        assert "$1,234.56" in body_text  # Money amounts should not be anonymized
        assert "CONF123456" in body_text  # Confirmation codes should not be anonymized
    
    def test_multi_language_privacy_filtering(self):
        """Test privacy filtering with multi-language content."""
        multilingual_data = pd.DataFrame({
            '_id': [1, 2, 3, 4],
            'body': [
                "Contact me at user@example.com or 555-123-4567",  # English
                "联系我 user@example.com 或 555-123-4567",  # Chinese
                "Свяжитесь со мной user@example.com или 555-123-4567",  # Russian
                "連絡先: user@example.com または 555-123-4567"  # Japanese
            ]
        })
        
        privacy_filter = PrivacyFilter(enable_nlp=False)
        field_mapping = {'body': PrivacyLevel.CRITICAL}
        
        filtered_data = privacy_filter.filter_dataframe(multilingual_data, field_mapping)
        
        # Verify sensitive data is anonymized regardless of language context
        for body in filtered_data['body']:
            assert "user@example.com" not in body
            assert "555-123-4567" not in body
            assert "[EMAIL]" in body
            assert "[PHONE]" in body
    
    def test_privacy_filter_with_quoted_messages(self):
        """Test privacy filtering with quoted/replied messages."""
        quoted_messages = pd.DataFrame({
            '_id': [1, 2, 3],
            'body': [
                "My phone number is 555-123-4567",
                "> My phone number is 555-123-4567\n\nThanks! I'll save that. Mine is 555-987-6543",
                ">> My phone number is 555-123-4567\n> Thanks! I'll save that. Mine is 555-987-6543\n\nGreat! Now we can coordinate better."
            ],
            'quote_body': [
                None,
                "My phone number is 555-123-4567",
                "Thanks! I'll save that. Mine is 555-987-6543"
            ]
        })
        
        privacy_filter = PrivacyFilter(enable_nlp=False)
        field_mapping = {
            'body': PrivacyLevel.CRITICAL,
            'quote_body': PrivacyLevel.CRITICAL
        }
        
        filtered_data = privacy_filter.filter_dataframe(quoted_messages, field_mapping)
        
        # Verify all phone numbers are anonymized in both body and quoted content
        for body in filtered_data['body'].dropna():
            assert "555-123-4567" not in body
            assert "555-987-6543" not in body
            assert "[PHONE]" in body
        
        for quote in filtered_data['quote_body'].dropna():
            assert "555-123-4567" not in quote and "555-987-6543" not in quote


if __name__ == "__main__":
    pytest.main([__file__])