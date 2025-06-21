#!/usr/bin/env python3
import csv
import re


def final_pattern_analysis():
    print("=== ADDITIONAL COMMUNICATION PATTERNS ===\n")

    # Look for specific patterns
    punctuation_creativity = []

    with open("data/raw/signal-flatfiles/signal.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        messages = []

        for row in reader:
            body = row.get("body", "").strip()
            if body and len(body) > 1 and not body.startswith("http"):
                messages.append(body)
                if len(messages) >= 1000:
                    break

    # Look for creative punctuation
    print("🎨 CREATIVE PUNCTUATION & EXPRESSION:")
    for msg in messages:
        if "!!!" in msg or "???" in msg:
            punctuation_creativity.append(msg)
        elif re.search(r"[.]{3,}", msg):
            punctuation_creativity.append(msg)

    for i, example in enumerate(punctuation_creativity[:8], 1):
        truncated = example[:100] + ("..." if len(example) > 100 else "")
        print(f"  {i}. {truncated}")

    print("\n📱 TEXTING HABITS:")
    # Count specific texting patterns
    lowercase_starts = sum(1 for msg in messages if msg[0].islower() and len(msg) > 5)
    no_punctuation = sum(1 for msg in messages if not re.search(r"[.!?]$", msg) and len(msg) > 10)
    single_word = sum(1 for msg in messages if len(msg.split()) == 1 and len(msg) > 2)

    print(
        f"  Messages starting lowercase: {lowercase_starts} ({lowercase_starts/len(messages)*100:.1f}%)"
    )
    print(
        f"  Messages without ending punctuation: {no_punctuation} ({no_punctuation/len(messages)*100:.1f}%)"
    )
    print(f"  Single-word messages: {single_word} ({single_word/len(messages)*100:.1f}%)")

    print("\n🎭 CONVERSATIONAL PERSONALITY TRAITS:")
    # Look for personality indicators
    intellectual_words = [
        "theory",
        "analysis",
        "fundamentally",
        "dialectic",
        "consciousness",
        "material",
        "fascist",
        "revolutionary",
        "thesis",
    ]
    casual_words = ["lol", "haha", "omg", "wtf", "damn", "shit", "fuck"]
    affectionate_words = [
        "love",
        "beautiful",
        "sweet",
        "gorgeous",
        "cute",
        "adorable",
        "baby",
        "honey",
    ]

    intellectual_count = sum(
        1 for msg in messages for word in intellectual_words if word in msg.lower()
    )
    casual_count = sum(1 for msg in messages for word in casual_words if word in msg.lower())
    affectionate_count = sum(
        1 for msg in messages for word in affectionate_words if word in msg.lower()
    )

    print(f"  Intellectual/Academic language: {intellectual_count} instances")
    print(f"  Casual/Informal language: {casual_count} instances")
    print(f"  Affectionate language: {affectionate_count} instances")

    print("\n💭 STREAM OF CONSCIOUSNESS EXAMPLES:")
    # Find messages that show stream of consciousness
    stream_examples = []
    for msg in messages:
        if ("..." in msg or "," in msg) and len(msg) > 80:
            stream_examples.append(msg)

    for i, example in enumerate(stream_examples[:5], 1):
        truncated = example[:120] + ("..." if len(example) > 120 else "")
        print(f"  {i}. {truncated}")

    print("\n🗨️ UNIQUE CONVERSATION QUIRKS:")
    quirks_found = []
    for msg in messages:
        msg_lower = msg.lower()
        if "lollllll" in msg_lower:
            quirks_found.append('Extended "lol" variations')
        if msg_lower.startswith("bitch "):
            quirks_found.append('Casual use of "bitch" as address')
        if "fuck with" in msg_lower and "fucking" not in msg_lower:
            quirks_found.append('"fuck with" meaning "enjoy/like"')
        if re.search(r"\b\w+ass\b", msg_lower):
            quirks_found.append("X-ass construction (adjective intensifier)")
        if "bestie" in msg_lower:
            quirks_found.append('Use of "bestie" as endearment')
        if msg_lower.strip() == "bet":
            quirks_found.append('"Bet" as standalone agreement')

    unique_quirks = list(set(quirks_found))
    for quirk in unique_quirks:
        print(f"  • {quirk}")


if __name__ == "__main__":
    final_pattern_analysis()
