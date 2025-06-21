"""Privacy filter module for protecting sensitive data in Astrabot.

This module implements comprehensive privacy protection following the
privacy architecture outlined in docs/explanation/privacy-architecture.md.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import spacy
except ImportError:
    spacy = None

from .logging import get_logger

logger = get_logger(__name__)


class PrivacyLevel(Enum):
    """Data privacy classification levels."""

    CRITICAL = "critical"  # 🔴 Highest protection
    HIGH = "high"  # 🟠 Strong protection
    MEDIUM = "medium"  # 🟡 Moderate protection
    LOW = "low"  # 🟢 Basic protection


@dataclass
class PrivacyPattern:
    """Pattern for detecting sensitive data."""

    name: str
    pattern: str
    replacement: str
    privacy_level: PrivacyLevel
    description: str


class PrivacyFilter:
    """Comprehensive privacy filter for Signal conversation data."""

    def __init__(self, enable_nlp: bool = False):
        """Initialize privacy filter.

        Args:
            enable_nlp: Whether to use spaCy for advanced NLP-based filtering
        """
        self.enable_nlp = enable_nlp
        self.nlp = None

        if enable_nlp:
            try:
                if spacy is None:
                    raise ImportError("spacy not available")
                self.nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                logger.warning("spaCy not available, falling back to regex-only filtering")
                self.enable_nlp = False

        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize privacy patterns for different data types."""
        self.patterns = [
            # Critical Privacy Patterns - Order matters! More specific patterns first
            PrivacyPattern(
                name="api_key_openai",
                pattern=r"\bsk-[A-Za-z0-9]{32,}\b",
                replacement="[API_KEY]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="OpenAI API keys",
            ),
            PrivacyPattern(
                name="api_key_anthropic",
                pattern=r"\bsk-ant-[A-Za-z0-9\-_]{32,}\b",
                replacement="[API_KEY]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Anthropic API keys",
            ),
            PrivacyPattern(
                name="credit_card",
                pattern=r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
                replacement="[CREDIT_CARD]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Credit card numbers",
            ),
            PrivacyPattern(
                name="ssn",
                pattern=r"\b\d{3}-\d{2}-\d{4}\b",
                replacement="[SSN]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Social Security Numbers",
            ),
            PrivacyPattern(
                name="phone_international_prefix",
                pattern=r"\+1[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
                replacement="[PHONE]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="US international phone numbers",
            ),
            PrivacyPattern(
                name="phone_us_11digit",
                pattern=r"\b1[0-9]{10}\b",
                replacement="[PHONE]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="US phone numbers with country code (11 digits)",
            ),
            PrivacyPattern(
                name="phone_us",
                pattern=r"\(?[0-9]{3}\)?\s*[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
                replacement="[PHONE]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="US phone numbers",
            ),
            PrivacyPattern(
                name="phone_short",
                pattern=r"\b[0-9]{3}[-.\s]?[0-9]{4}\b",
                replacement="[PHONE]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Short phone numbers (7 digits)",
            ),
            PrivacyPattern(
                name="phone_international",
                pattern=r"\+[1-9]\d{1,14}",
                replacement="[PHONE]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Other international phone numbers (E.164)",
            ),
            PrivacyPattern(
                name="email",
                pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                replacement="[EMAIL]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Email addresses",
            ),
            PrivacyPattern(
                name="bitcoin_address",
                pattern=r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b",
                replacement="[CRYPTO_ADDRESS]",
                privacy_level=PrivacyLevel.CRITICAL,
                description="Bitcoin addresses",
            ),
            # High Privacy Patterns
            PrivacyPattern(
                name="ip_address",
                pattern=r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                replacement="[IP_ADDRESS]",
                privacy_level=PrivacyLevel.HIGH,
                description="IP addresses",
            ),
            PrivacyPattern(
                name="mac_address",
                pattern=r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b",
                replacement="[MAC_ADDRESS]",
                privacy_level=PrivacyLevel.HIGH,
                description="MAC addresses",
            ),
            PrivacyPattern(
                name="url_with_params",
                pattern=r"https?://[^\s]+\?[^\s]*(?:token|key|session|auth)[^\s]*",
                replacement="[PRIVATE_URL]",
                privacy_level=PrivacyLevel.HIGH,
                description="URLs with authentication parameters",
            ),
            # Medium Privacy Patterns
            PrivacyPattern(
                name="timestamp_precise",
                pattern=r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z",
                replacement="[TIMESTAMP]",
                privacy_level=PrivacyLevel.MEDIUM,
                description="Precise timestamps",
            ),
        ]

    def classify_data(self, text: str, context: Optional[str] = None) -> PrivacyLevel:
        """Classify text by privacy level based on content analysis.

        Args:
            text: Text to classify
            context: Additional context for classification

        Returns:
            PrivacyLevel: Highest privacy level found in text
        """
        if not text or not isinstance(text, str):
            return PrivacyLevel.LOW

        # Check for critical patterns first
        for pattern in self.patterns:
            if re.search(pattern.pattern, text, re.IGNORECASE):
                return pattern.privacy_level

        # NLP-based classification for names and entities
        if self.enable_nlp and self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    return PrivacyLevel.CRITICAL
                elif ent.label_ in ["GPE", "LOC", "FAC"]:
                    return PrivacyLevel.HIGH

        # Context-based classification
        if context:
            sensitive_contexts = ["password", "login", "auth", "private", "personal"]
            if any(word in context.lower() for word in sensitive_contexts):
                return PrivacyLevel.HIGH

        # Default to medium for message content, low for metadata
        return PrivacyLevel.MEDIUM if len(text) > 10 else PrivacyLevel.LOW

    def anonymize_text(self, text: str, privacy_level: Optional[PrivacyLevel] = None) -> str:
        """Anonymize text by removing or masking sensitive information.

        Args:
            text: Text to anonymize
            privacy_level: Minimum privacy level to filter (None = filter all)

        Returns:
            Anonymized text
        """
        if not text or not isinstance(text, str):
            return text

        anonymized = text

        # Apply regex patterns
        for pattern in self.patterns:
            if privacy_level is None or pattern.privacy_level.value in [
                PrivacyLevel.CRITICAL.value,
                PrivacyLevel.HIGH.value,
            ]:
                anonymized = re.sub(
                    pattern.pattern, pattern.replacement, anonymized, flags=re.IGNORECASE
                )

        # Apply NLP-based anonymization
        if self.enable_nlp and self.nlp:
            anonymized = self._anonymize_with_nlp(anonymized)

        return anonymized

    def _anonymize_with_nlp(self, text: str) -> str:
        """Use NLP to detect and replace named entities."""
        if not self.nlp:
            return text

        doc = self.nlp(text)

        # Collect replacements in reverse order
        replacements = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON"]:
                replacements.append((ent.start_char, ent.end_char, "[PERSON]"))
            elif ent.label_ in ["ORG"]:
                replacements.append((ent.start_char, ent.end_char, "[ORGANIZATION]"))
            elif ent.label_ in ["GPE", "LOC"]:
                replacements.append((ent.start_char, ent.end_char, "[LOCATION]"))
            elif ent.label_ in ["FAC"]:
                replacements.append((ent.start_char, ent.end_char, "[FACILITY]"))

        # Apply replacements in reverse order to maintain indices
        for start, end, replacement in reversed(replacements):
            text = text[:start] + replacement + text[end:]

        return text

    def hash_identifier(self, identifier: str, salt: str = "astrabot") -> str:
        """Create a consistent hash for identifiers while preserving uniqueness.

        Args:
            identifier: Original identifier (phone, email, etc.)
            salt: Salt for hashing

        Returns:
            Hashed identifier (first 8 chars of SHA256)
        """
        if not identifier:
            return ""

        hash_input = f"{salt}:{identifier}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:8]

    def filter_dataframe(
        self, df: pd.DataFrame, field_mapping: dict[str, PrivacyLevel]
    ) -> pd.DataFrame:
        """Filter a DataFrame according to privacy levels.

        Args:
            df: DataFrame to filter
            field_mapping: Mapping of column names to privacy levels

        Returns:
            Filtered DataFrame with sensitive data anonymized
        """
        filtered_df = df.copy()

        for column, privacy_level in field_mapping.items():
            if column in filtered_df.columns:
                if privacy_level == PrivacyLevel.CRITICAL:
                    # Remove or heavily anonymize critical data
                    if column in ["body", "message_content"]:
                        filtered_df[column] = filtered_df[column].apply(
                            lambda x: self.anonymize_text(x) if pd.notna(x) else x
                        )
                    elif column in ["e164", "phone_number", "sender_phone"]:
                        filtered_df[column] = filtered_df[column].apply(
                            lambda x: self.hash_identifier(str(x)) if pd.notna(x) else x
                        )
                    elif column in [
                        "profile_given_name",
                        "profile_family_name",
                        "sender_name",
                        "system_given_name",
                        "system_family_name",
                    ]:
                        filtered_df[column] = "[REDACTED]"

                elif privacy_level == PrivacyLevel.HIGH:
                    # Partially anonymize high privacy data
                    filtered_df[column] = filtered_df[column].apply(
                        lambda x: self._partial_mask(str(x)) if pd.notna(x) else x
                    )

        return filtered_df

    def _partial_mask(self, value: str) -> str:
        """Partially mask a value showing only first and last characters."""
        if len(value) <= 4:
            return "***"
        return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"

    def add_differential_privacy(self, df: pd.DataFrame, epsilon: float = 1.0) -> pd.DataFrame:
        """Add differential privacy noise to numeric data.

        Args:
            df: DataFrame to add noise to
            epsilon: Privacy parameter (smaller = more private)

        Returns:
            DataFrame with noise added to numeric columns
        """
        noisy_df = df.copy()

        # Add Laplace noise to numeric columns
        for column in df.select_dtypes(include=[np.number]).columns:
            if column not in ["_id", "thread_id"]:  # Don't add noise to IDs
                sensitivity = 1.0  # Assume sensitivity of 1
                scale = sensitivity / epsilon
                noise = np.random.laplace(0, scale, len(df))
                noisy_df[column] = noisy_df[column] + noise

        return noisy_df

    def validate_anonymization(self, original: str, anonymized: str) -> dict[str, Any]:
        """Validate that anonymization was effective.

        Args:
            original: Original text
            anonymized: Anonymized text

        Returns:
            Dictionary with validation results
        """
        results = {
            "is_anonymized": original != anonymized,
            "patterns_found": [],
            "potential_leaks": [],
            "anonymization_ratio": 0.0,
        }

        # Check what patterns were found and replaced
        for pattern in self.patterns:
            original_matches = re.findall(pattern.pattern, original, re.IGNORECASE)
            anonymized_matches = re.findall(pattern.pattern, anonymized, re.IGNORECASE)

            if original_matches:
                results["patterns_found"].append(
                    {
                        "pattern": pattern.name,
                        "count": len(original_matches),
                        "fully_anonymized": len(anonymized_matches) == 0,
                    }
                )

                if anonymized_matches:
                    results["potential_leaks"].extend(anonymized_matches)

        # Calculate anonymization ratio
        if len(original) > 0:
            results["anonymization_ratio"] = (len(original) - len(anonymized)) / len(original)

        return results

    def create_privacy_report(self, df: pd.DataFrame) -> dict[str, Any]:
        """Create a comprehensive privacy report for a dataset.

        Args:
            df: DataFrame to analyze

        Returns:
            Privacy analysis report
        """
        report = {
            "total_records": len(df),
            "columns_analyzed": list(df.columns),
            "privacy_classifications": {},
            "sensitive_data_found": {},
            "recommendations": [],
        }

        for column in df.columns:
            if df[column].dtype == "object":  # Text columns
                sample_values = df[column].dropna().head(100)
                privacy_levels = [self.classify_data(str(val)) for val in sample_values]

                # Most restrictive privacy level found
                max_privacy = max(
                    privacy_levels,
                    key=lambda x: ["low", "medium", "high", "critical"].index(x.value),
                )
                report["privacy_classifications"][column] = max_privacy.value

                # Count sensitive patterns
                sensitive_count = 0
                for val in sample_values:
                    for pattern in self.patterns:
                        if re.search(pattern.pattern, str(val), re.IGNORECASE):
                            sensitive_count += 1
                            break

                if sensitive_count > 0:
                    report["sensitive_data_found"][column] = {
                        "count": sensitive_count,
                        "percentage": (sensitive_count / len(sample_values)) * 100,
                    }

        # Generate recommendations
        for column, classification in report["privacy_classifications"].items():
            if classification in ["critical", "high"]:
                report["recommendations"].append(
                    f"Column '{column}' contains {classification} privacy data - consider anonymization"
                )

        return report


# Convenience functions for common operations
def mask_sensitive_data(text: str) -> str:
    """Quick function to mask sensitive data in text.

    Args:
        text: Text to mask

    Returns:
        Text with sensitive data masked
    """
    filter_instance = PrivacyFilter()
    return filter_instance.anonymize_text(text)


def classify_signal_data_privacy(df: pd.DataFrame) -> dict[str, PrivacyLevel]:
    """Classify Signal DataFrame columns by privacy level.

    Args:
        df: Signal DataFrame

    Returns:
        Dictionary mapping column names to privacy levels
    """
    # Standard Signal schema privacy classifications
    classification = {}

    if "body" in df.columns:
        classification["body"] = PrivacyLevel.CRITICAL
    if "e164" in df.columns:
        classification["e164"] = PrivacyLevel.CRITICAL
    if "profile_given_name" in df.columns:
        classification["profile_given_name"] = PrivacyLevel.CRITICAL
    if "profile_family_name" in df.columns:
        classification["profile_family_name"] = PrivacyLevel.CRITICAL
    if "system_given_name" in df.columns:
        classification["system_given_name"] = PrivacyLevel.CRITICAL
    if "system_family_name" in df.columns:
        classification["system_family_name"] = PrivacyLevel.CRITICAL

    # High privacy
    if "aci" in df.columns:
        classification["aci"] = PrivacyLevel.HIGH
    if "pni" in df.columns:
        classification["pni"] = PrivacyLevel.HIGH
    if "from_recipient_id" in df.columns:
        classification["from_recipient_id"] = PrivacyLevel.HIGH
    if "to_recipient_id" in df.columns:
        classification["to_recipient_id"] = PrivacyLevel.HIGH

    # Medium privacy
    if "date_sent" in df.columns:
        classification["date_sent"] = PrivacyLevel.MEDIUM
    if "date_received" in df.columns:
        classification["date_received"] = PrivacyLevel.MEDIUM
    if "read" in df.columns:
        classification["read"] = PrivacyLevel.MEDIUM

    # Low privacy
    if "_id" in df.columns:
        classification["_id"] = PrivacyLevel.LOW
    if "thread_id" in df.columns:
        classification["thread_id"] = PrivacyLevel.LOW
    if "type" in df.columns:
        classification["type"] = PrivacyLevel.LOW

    return classification
