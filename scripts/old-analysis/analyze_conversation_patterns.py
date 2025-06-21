#!/usr/bin/env python3
"""
Conversation Pattern Analysis Script

Analyzes Signal conversation data to extract:
- Emoji usage statistics and patterns
- Common topics and themes
- Message timing and burst patterns
- Communication style characteristics
- Emotional expression patterns

Usage:
    python scripts/analyze_conversation_patterns.py
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import emoji
import pandas as pd


def load_signal_data(
    data_dir: str = "data/raw/signal-flatfiles",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Signal CSV files."""
    base_path = Path(data_dir)

    # Load main data files
    messages_df = pd.read_csv(base_path / "signal.csv")
    recipients_df = pd.read_csv(base_path / "recipient.csv")
    reactions_df = pd.read_csv(base_path / "reaction.csv")

    return messages_df, recipients_df, reactions_df


def extract_emojis_from_text(text: str) -> list[str]:
    """Extract all emojis from text."""
    if pd.isna(text) or not isinstance(text, str):
        return []
    return [char for char in text if char in emoji.EMOJI_DATA]


def analyze_emoji_usage(messages_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze emoji usage patterns."""
    print("🔍 Analyzing emoji usage patterns...")

    # Extract all emojis
    all_emojis = []
    emoji_contexts = defaultdict(list)

    for _, row in messages_df.iterrows():
        body = row.get("body", "")
        if pd.isna(body):
            continue

        emojis = extract_emojis_from_text(body)
        all_emojis.extend(emojis)

        # Store context for each emoji
        for em in emojis:
            emoji_contexts[em].append(
                {
                    "message": body,
                    "sender": row.get("from_recipient_id"),
                    "timestamp": row.get("date_sent"),
                }
            )

    # Count emoji frequencies
    emoji_counts = Counter(all_emojis)

    # Analyze emoji by sender
    emoji_by_sender = defaultdict(Counter)
    for _, row in messages_df.iterrows():
        body = row.get("body", "")
        sender = row.get("from_recipient_id")
        if pd.isna(body):
            continue
        emojis = extract_emojis_from_text(body)
        emoji_by_sender[sender].update(emojis)

    return {
        "total_emojis": len(all_emojis),
        "unique_emojis": len(emoji_counts),
        "most_common": emoji_counts.most_common(20),
        "by_sender": dict(emoji_by_sender),
        "contexts": dict(emoji_contexts),
    }


def analyze_message_timing(messages_df: pd.DataFrame) -> dict[str, Any]:
    """Analyze message timing and burst patterns."""
    print("⏰ Analyzing message timing patterns...")

    # Convert timestamps
    messages_df["datetime"] = pd.to_datetime(messages_df["date_sent"], unit="ms")

    # Sort by thread and time
    sorted_msgs = messages_df.sort_values(["thread_id", "date_sent"])

    burst_sequences = []
    response_times = []

    # Analyze by thread
    for thread_id in sorted_msgs["thread_id"].unique():
        thread_msgs = sorted_msgs[sorted_msgs["thread_id"] == thread_id].copy()

        if len(thread_msgs) < 2:
            continue

        # Calculate time gaps
        thread_msgs["time_gap"] = thread_msgs["date_sent"].diff() / 1000  # seconds

        # Identify burst sequences (messages < 2 minutes apart)
        current_burst = []
        for _, row in thread_msgs.iterrows():
            time_gap = row.get("time_gap", float("inf"))

            if time_gap < 120 or len(current_burst) == 0:  # < 2 minutes or first message
                current_burst.append(row.to_dict())
            else:
                if len(current_burst) >= 3:  # Only consider bursts of 3+ messages
                    burst_sequences.append(
                        {
                            "thread_id": thread_id,
                            "length": len(current_burst),
                            "duration": current_burst[-1]["date_sent"]
                            - current_burst[0]["date_sent"],
                            "messages": current_burst,
                        }
                    )
                current_burst = [row.to_dict()]

        # Don't forget the last burst
        if len(current_burst) >= 3:
            burst_sequences.append(
                {
                    "thread_id": thread_id,
                    "length": len(current_burst),
                    "duration": current_burst[-1]["date_sent"] - current_burst[0]["date_sent"],
                    "messages": current_burst,
                }
            )

        # Calculate response times
        for i in range(1, len(thread_msgs)):
            prev_sender = thread_msgs.iloc[i - 1]["from_recipient_id"]
            curr_sender = thread_msgs.iloc[i]["from_recipient_id"]

            if prev_sender != curr_sender:  # Different senders = response
                response_time = thread_msgs.iloc[i]["time_gap"]
                if response_time < 3600:  # Only count responses within 1 hour
                    response_times.append(response_time)

    return {
        "total_bursts": len(burst_sequences),
        "avg_burst_length": (
            sum(b["length"] for b in burst_sequences) / len(burst_sequences)
            if burst_sequences
            else 0
        ),
        "longest_burst": (
            max(burst_sequences, key=lambda x: x["length"]) if burst_sequences else None
        ),
        "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
        "burst_sequences": burst_sequences[:10],  # Top 10 for analysis
        "response_time_stats": {
            "count": len(response_times),
            "avg": sum(response_times) / len(response_times) if response_times else 0,
            "median": sorted(response_times)[len(response_times) // 2] if response_times else 0,
        },
    }


def extract_topics_and_themes(messages_df: pd.DataFrame) -> dict[str, Any]:
    """Extract common topics and themes from messages."""
    print("📝 Extracting topics and themes...")

    # Political/theory keywords
    political_keywords = [
        "fascism",
        "fascist",
        "theory",
        "marxist",
        "communist",
        "capitalism",
        "dialectical",
        "material",
        "bourgeois",
        "proletariat",
        "revolution",
        "nazi",
        "zionism",
        "imperialism",
        "nato",
        "cia",
        "fbi",
    ]

    # Personal/casual keywords
    personal_keywords = [
        "food",
        "cooking",
        "eat",
        "hungry",
        "dinner",
        "lunch",
        "coffee",
        "sleep",
        "tired",
        "work",
        "job",
        "money",
        "family",
        "friend",
    ]

    # Internet culture keywords
    internet_keywords = [
        "lmao",
        "omg",
        "lol",
        "wtf",
        "tbh",
        "ngl",
        "fr",
        "bruh",
        "meme",
        "tweet",
        "twitter",
        "tiktok",
        "reddit",
        "viral",
    ]

    # Extract messages with valid bodies
    messages = []
    for _, row in messages_df.iterrows():
        body = row.get("body", "")
        if pd.isna(body) or not isinstance(body, str) or len(body.strip()) == 0:
            continue
        messages.append(body.lower())

    # Count keyword occurrences
    political_count = sum(1 for msg in messages if any(kw in msg for kw in political_keywords))
    personal_count = sum(1 for msg in messages if any(kw in msg for kw in personal_keywords))
    internet_count = sum(1 for msg in messages if any(kw in msg for kw in internet_keywords))

    # Extract URLs (Twitter links especially)
    url_pattern = r"https?://[^\s]+"
    urls = []
    twitter_urls = []

    for _, row in messages_df.iterrows():
        body = row.get("body", "")
        if pd.isna(body):
            continue
        found_urls = re.findall(url_pattern, body)
        urls.extend(found_urls)
        twitter_urls.extend([url for url in found_urls if "twitter.com" in url or "x.com" in url])

    # Language patterns
    profanity_pattern = r"\b(fuck|shit|damn|hell|ass|bitch)\w*"
    academic_pattern = r"\b(dialectical|materialist|bourgeois|proletarian|hegemony|ideology)\b"

    profanity_msgs = [msg for msg in messages if re.search(profanity_pattern, msg)]
    academic_msgs = [msg for msg in messages if re.search(academic_pattern, msg)]

    return {
        "total_messages_analyzed": len(messages),
        "topic_distribution": {
            "political_theory": political_count,
            "personal_casual": personal_count,
            "internet_culture": internet_count,
        },
        "url_sharing": {
            "total_urls": len(urls),
            "twitter_urls": len(twitter_urls),
            "url_percentage": len(urls) / len(messages) * 100 if messages else 0,
        },
        "language_patterns": {
            "profanity_usage": len(profanity_msgs),
            "academic_language": len(academic_msgs),
            "mixed_register": len(set(profanity_msgs) & set(academic_msgs)),
        },
        "sample_academic_messages": academic_msgs[:5],
        "sample_casual_messages": [
            msg for msg in messages if any(kw in msg for kw in internet_keywords)
        ][:5],
    }


def analyze_sender_patterns(
    messages_df: pd.DataFrame, recipients_df: pd.DataFrame
) -> dict[str, Any]:
    """Analyze communication patterns by sender."""
    print("👥 Analyzing sender communication patterns...")

    # Get sender info
    sender_lookup = recipients_df.set_index("_id").to_dict("index")

    sender_stats = defaultdict(
        lambda: {
            "message_count": 0,
            "avg_length": 0,
            "total_chars": 0,
            "emoji_count": 0,
            "url_count": 0,
            "burst_count": 0,
        }
    )

    # Analyze each message
    for _, row in messages_df.iterrows():
        sender_id = row.get("from_recipient_id")
        body = row.get("body", "")

        if pd.isna(body) or not isinstance(body, str):
            continue

        stats = sender_stats[sender_id]
        stats["message_count"] += 1
        stats["total_chars"] += len(body)
        stats["emoji_count"] += len(extract_emojis_from_text(body))
        stats["url_count"] += len(re.findall(r"https?://[^\s]+", body))

    # Calculate averages
    for sender_id, stats in sender_stats.items():
        if stats["message_count"] > 0:
            stats["avg_length"] = stats["total_chars"] / stats["message_count"]
            stats["sender_name"] = sender_lookup.get(sender_id, {}).get(
                "profile_given_name", f"User_{sender_id}"
            )

    return dict(sender_stats)


def generate_conversation_report(
    messages_df: pd.DataFrame, recipients_df: pd.DataFrame, reactions_df: pd.DataFrame
) -> dict[str, Any]:
    """Generate comprehensive conversation analysis report."""
    print("📊 Generating comprehensive conversation analysis report...")

    # Run all analyses
    emoji_analysis = analyze_emoji_usage(messages_df)
    timing_analysis = analyze_message_timing(messages_df)
    topic_analysis = extract_topics_and_themes(messages_df)
    sender_analysis = analyze_sender_patterns(messages_df, recipients_df)

    # Basic stats
    total_messages = len(messages_df[messages_df["body"].notna()])
    unique_senders = messages_df["from_recipient_id"].nunique()
    unique_threads = messages_df["thread_id"].nunique()
    date_range = (
        pd.to_datetime(messages_df["date_sent"].min(), unit="ms"),
        pd.to_datetime(messages_df["date_sent"].max(), unit="ms"),
    )

    return {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_messages": total_messages,
            "unique_senders": unique_senders,
            "unique_threads": unique_threads,
            "date_range": [d.isoformat() for d in date_range],
        },
        "emoji_analysis": emoji_analysis,
        "timing_analysis": timing_analysis,
        "topic_analysis": topic_analysis,
        "sender_analysis": sender_analysis,
    }


def main():
    """Main analysis function."""
    print("🚀 Starting Signal conversation pattern analysis...")

    try:
        # Load data
        messages_df, recipients_df, reactions_df = load_signal_data()
        print(
            f"✅ Loaded {len(messages_df)} messages, {len(recipients_df)} recipients, {len(reactions_df)} reactions"
        )

        # Generate report
        report = generate_conversation_report(messages_df, recipients_df, reactions_df)

        # Save report
        output_path = Path("data/processed/conversation_patterns_analysis.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📝 Analysis complete! Report saved to {output_path}")

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)

        emoji_data = report["emoji_analysis"]
        timing_data = report["timing_analysis"]
        topic_data = report["topic_analysis"]

        print(f"📊 Total Messages: {report['metadata']['total_messages']:,}")
        print(f"🎭 Total Emojis: {emoji_data['total_emojis']:,}")
        print(f"🔥 Burst Sequences: {timing_data['total_bursts']}")
        print(f"🔗 URLs Shared: {topic_data['url_sharing']['total_urls']}")
        print(f"🐦 Twitter Links: {topic_data['url_sharing']['twitter_urls']}")

        print("\n🎭 Top Emojis:")
        for emoji_char, count in emoji_data["most_common"][:10]:
            print(f"  {emoji_char} : {count}")

        print(f"\n📈 Topic Distribution:")
        for topic, count in topic_data["topic_distribution"].items():
            print(f"  {topic}: {count} messages")

        print("\n✨ Analysis saved successfully!")

    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
