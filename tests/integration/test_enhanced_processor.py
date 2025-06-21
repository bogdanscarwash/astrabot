#!/usr/bin/env python3
"""
Test script for the Enhanced Conversation Processor.

This script tests the new enhanced conversation processor against actual Signal data
to verify all components work together correctly.
"""


import pandas as pd

from src.core.conversation_processor import EnhancedConversationProcessor


def test_enhanced_processor():
    """Test the enhanced conversation processor with sample data."""

    # Initialize processor
    processor = EnhancedConversationProcessor()
    print("✓ Enhanced conversation processor initialized")

    # Create sample Signal data (mimicking actual structure)
    sample_messages = pd.DataFrame(
        [
            {
                "_id": "1",
                "thread_id": "thread_1",
                "from_recipient_id": "2",
                "date_sent": 1640995200000,  # 2022-01-01 timestamp
                "body": "Check out this fascist bullshit 🙄 https://twitter.com/example/status/123",
            },
            {
                "_id": "2",
                "thread_id": "thread_1",
                "from_recipient_id": "3",
                "date_sent": 1640995260000,  # 1 minute later
                "body": "Yeah that's absolutely wild 😂",
            },
            {
                "_id": "3",
                "thread_id": "thread_1",
                "from_recipient_id": "2",
                "date_sent": 1640995280000,  # 20 seconds later
                "body": "The dialectical analysis shows",
            },
            {
                "_id": "4",
                "thread_id": "thread_1",
                "from_recipient_id": "2",
                "date_sent": 1640995300000,  # 20 seconds later (burst)
                "body": "how capitalism creates these conditions",
            },
            {
                "_id": "5",
                "thread_id": "thread_1",
                "from_recipient_id": "2",
                "date_sent": 1640995320000,  # 20 seconds later (burst continues)
                "body": "and the material base drives the superstructure 🤔",
            },
        ]
    )

    sample_recipients = pd.DataFrame(
        [{"_id": "2", "profile_given_name": "You"}, {"_id": "3", "profile_given_name": "Alice"}]
    )

    print("✓ Sample data created")

    # Test message processing
    print("\n--- Testing Message Processing ---")
    context_messages = []
    for _, row in sample_messages.iterrows():
        signal_msg = processor.process_signal_message(
            row, context_messages[-3:] if context_messages else None
        )
        context_messages.append(signal_msg)

        print(f"Message {signal_msg.message_id}:")
        print(f"  Topic: {signal_msg.topic_category}")
        print(f"  Tone: {signal_msg.emotional_tone}")
        print(f"  Type: {signal_msg.message_type}")
        print(f"  Emojis: {signal_msg.emoji_list}")
        print(f"  URLs: {signal_msg.url_list}")
        print(f"  Academic: {signal_msg.academic_language}")
        print(f"  Continuation: {signal_msg.is_continuation}")

    # Test burst detection
    print("\n--- Testing Burst Detection ---")
    burst_sequences = processor.detect_burst_sequences(context_messages)
    print(f"Detected {len(burst_sequences)} burst sequences")
    for i, burst in enumerate(burst_sequences):
        print(
            f"Burst {i+1}: {burst.message_count} messages, {burst.duration_seconds:.1f}s duration"
        )
        print(f"  Topic: {burst.topic_category}, Tone: {burst.emotional_tone}")

    # Test conversation windows
    print("\n--- Testing Conversation Windows ---")
    windows = processor.create_conversation_windows(
        context_messages, window_size=4, your_recipient_id="2"
    )
    print(f"Created {len(windows)} conversation windows")
    for i, window in enumerate(windows):
        print(f"Window {i+1}:")
        print(f"  Duration: {window.duration_minutes:.1f} minutes")
        print(f"  Mood: {window.dominant_mood}")
        print(f"  Primary topic: {window.primary_topic}")
        print(f"  Speakers: {window.unique_speakers}")
        print(f"  Dynamics: {window.conversation_dynamics}")

    # Test full thread processing
    print("\n--- Testing Full Thread Processing ---")
    try:
        thread = processor.process_full_conversation_thread(
            sample_messages, "thread_1", sample_recipients, your_recipient_id="2"
        )

        if thread:
            print(f"Thread processed successfully:")
            print(f"  Participants: {thread.participants}")
            print(f"  Total messages: {thread.total_messages}")
            print(f"  Duration: {thread.total_duration_days:.2f} days")
            print(f"  Dominant topics: {[t.value for t in thread.dominant_topics[:3]]}")
            print(f"  Relationship: {thread.relationship_dynamic}")
            print(f"  Windows: {len(thread.windows)}")

            # Test training data generation
            print("\n--- Testing Training Data Generation ---")
            training_data = processor.generate_training_data(thread, your_recipient_id="2")
            print(f"Generated {len(training_data)} training examples")

            if training_data:
                example = training_data[0]
                print("Sample training example:")
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            print("❌ Thread processing returned None")

    except Exception as e:
        print(f"❌ Error in thread processing: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ Enhanced processor test completed!")


if __name__ == "__main__":
    test_enhanced_processor()
