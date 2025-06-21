"""
Core processing modules for Astrabot.

This package contains the core conversation processing logic.
"""

# Import conversation analyzer functions
from src.core.conversation_analyzer import (
    analyze_conversational_patterns,
    analyze_message_bursts,
    analyze_personal_texting_style,
    create_conversation_windows,
    model_conversation_roles,
    segment_natural_dialogues,
)

# Import conversation processor functions and classes
from src.core.conversation_processor import (
    EnhancedConversationProcessor,
    TwitterExtractor,
    describe_tweet_images,
    describe_tweet_images_with_context,
    extract_tweet_images,
    extract_tweet_text,
    inject_tweet_context,
    preserve_conversation_dynamics,
    process_message_with_structured_content,
    process_message_with_twitter_content,
)

# Import metadata enricher functions
from src.core.metadata_enricher import (
    add_conversation_flow_metadata,
    add_group_context,
    add_reaction_context,
    add_temporal_context,
    classify_emotion_from_reactions,
    classify_urgency,
    enrich_messages_with_all_metadata,
    get_time_period,
)

# Import style analyzer functions
from src.core.style_analyzer import (
    analyze_all_communication_styles,
    analyze_emoji_usage,
    analyze_message_bursts,
    analyze_response_patterns,
    analyze_timing_patterns,
    analyze_your_adaptation_patterns,
    classify_communication_style,
    create_adaptation_context,
)

# Import Q&A extractor functions
# TODO: Implement qa_extractor.py
# from src.core.qa_extractor import (
#     extract_qa_pairs_enhanced,
#     extract_qa_pairs_with_quality_filters,
#     analyze_qa_patterns
# )

__all__ = [
    # Conversation analyzer
    "create_conversation_windows",
    "segment_natural_dialogues",
    "model_conversation_roles",
    "analyze_personal_texting_style",
    "analyze_message_bursts",
    "analyze_conversational_patterns",
    # Conversation processor
    "extract_tweet_text",
    "inject_tweet_context",
    "extract_tweet_images",
    "describe_tweet_images",
    "describe_tweet_images_with_context",
    "process_message_with_twitter_content",
    "process_message_with_structured_content",
    "preserve_conversation_dynamics",
    "TwitterExtractor",
    "EnhancedConversationProcessor",
    # Metadata enricher
    "add_reaction_context",
    "add_group_context",
    "add_temporal_context",
    "get_time_period",
    "classify_urgency",
    "classify_emotion_from_reactions",
    "add_conversation_flow_metadata",
    "enrich_messages_with_all_metadata",
    # Style analyzer
    "analyze_all_communication_styles",
    "classify_communication_style",
    "analyze_emoji_usage",
    "analyze_timing_patterns",
    "analyze_response_patterns",
    "create_adaptation_context",
    "analyze_your_adaptation_patterns",
    # Q&A extractor
    # TODO: Add back when qa_extractor.py is implemented
    # "extract_qa_pairs_enhanced",
    # "extract_qa_pairs_with_quality_filters",
    # "analyze_qa_patterns"
]
