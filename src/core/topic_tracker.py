"""
Topic Tracking Module for Astrabot.

This module provides topic analysis and tracking capabilities including:
- Political discussion threading and categorization
- Food/personal life topic detection
- Meme/humor context analysis
- Current events discussion patterns
- Topic transition analysis
"""

import re
from collections import Counter, defaultdict
from typing import Any, Optional
from urllib.parse import urlparse

import pandas as pd

from src.models.conversation_schemas import TopicTransition
from src.models.schemas import TopicCategory
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TopicTracker:
    """Advanced topic detection and tracking for conversation analysis."""

    def __init__(self):
        """Initialize topic tracker with keyword mappings based on Signal data analysis."""
        self._init_topic_keywords()
        self._init_academic_indicators()
        self._init_humor_patterns()
        logger.info("TopicTracker initialized with Signal data patterns")

    def _init_topic_keywords(self) -> None:
        """Initialize topic keyword mappings based on actual Signal conversation analysis."""
        self.topic_keywords = {
            TopicCategory.POLITICAL_THEORY: {
                "primary": [
                    "dialectical",
                    "materialist",
                    "marxist",
                    "communist",
                    "capitalism",
                    "bourgeois",
                    "proletariat",
                    "class struggle",
                    "surplus value",
                    "commodity",
                    "labor power",
                    "means of production",
                    "base and superstructure",
                    "ideology",
                    "hegemony",
                    "praxis",
                    "revolutionary",
                    "revolution",
                ],
                "secondary": [
                    "theory",
                    "analysis",
                    "framework",
                    "paradigm",
                    "methodology",
                    "thesis",
                    "antithesis",
                    "synthesis",
                    "critique",
                    "discourse",
                ],
            },
            TopicCategory.POLITICS: {
                "primary": [
                    "fascism",
                    "fascist",
                    "nazis",
                    "nazi",
                    "authoritarianism",
                    "totalitarian",
                    "revanchist",
                    "supranationalist",
                    "militarism",
                    "ultranationalism",
                    "biden",
                    "trump",
                    "election",
                    "congress",
                    "senate",
                    "democrat",
                    "republican",
                    "liberal",
                    "conservative",
                    "progressive",
                    "policy",
                    "legislation",
                    "nato",
                    "ukraine",
                    "russia",
                    "china",
                    "palestine",
                    "israel",
                    "zionism",
                ],
                "secondary": [
                    "government",
                    "political",
                    "vote",
                    "campaign",
                    "candidate",
                    "party",
                    "politician",
                    "democratic",
                    "republic",
                ],
            },
            TopicCategory.CURRENT_EVENTS: {
                "primary": [
                    "news",
                    "breaking",
                    "happened",
                    "today",
                    "yesterday",
                    "recent",
                    "update",
                    "development",
                    "situation",
                    "event",
                    "report",
                    "intelligence",
                    "cia",
                    "fbi",
                    "nsa",
                    "surveillance",
                    "classified",
                ],
                "secondary": ["media", "press", "journalist", "article", "story", "coverage"],
            },
            TopicCategory.FOOD: {
                "primary": [
                    "food",
                    "cooking",
                    "recipe",
                    "eat",
                    "eating",
                    "dinner",
                    "lunch",
                    "breakfast",
                    "restaurant",
                    "kitchen",
                    "cook",
                    "meal",
                    "hungry",
                    "delicious",
                    "taste",
                    "flavor",
                    "spice",
                    "arepas",
                    "coffee",
                    "drink",
                    "drinking",
                ],
                "secondary": [
                    "chef",
                    "cuisine",
                    "dish",
                    "ingredient",
                    "bake",
                    "baking",
                    "grill",
                    "grilling",
                    "fry",
                    "boil",
                    "steam",
                ],
            },
            TopicCategory.PERSONAL_LIFE: {
                "primary": [
                    "work",
                    "job",
                    "career",
                    "family",
                    "friend",
                    "friends",
                    "relationship",
                    "home",
                    "house",
                    "travel",
                    "vacation",
                    "health",
                    "exercise",
                    "sleep",
                    "tired",
                    "money",
                    "weekend",
                    "today",
                    "tomorrow",
                ],
                "secondary": ["personal", "private", "life", "living", "daily", "routine"],
            },
            TopicCategory.TECHNOLOGY: {
                "primary": [
                    "computer",
                    "software",
                    "programming",
                    "code",
                    "tech",
                    "app",
                    "website",
                    "internet",
                    "digital",
                    "algorithm",
                    "ai",
                    "machine learning",
                    "phone",
                    "smartphone",
                    "laptop",
                    "server",
                    "database",
                ],
                "secondary": ["technical", "system", "platform", "online", "cyber", "virtual"],
            },
            TopicCategory.ACADEMIC: {
                "primary": [
                    "philosophy",
                    "history",
                    "economics",
                    "sociology",
                    "psychology",
                    "literature",
                    "science",
                    "research",
                    "study",
                    "education",
                    "university",
                    "college",
                    "professor",
                    "student",
                    "academic",
                ],
                "secondary": ["intellectual", "scholarly", "theoretical", "analytical", "critical"],
            },
            TopicCategory.ENTERTAINMENT: {
                "primary": [
                    "movie",
                    "film",
                    "tv",
                    "show",
                    "television",
                    "music",
                    "song",
                    "band",
                    "artist",
                    "game",
                    "gaming",
                    "book",
                    "read",
                    "reading",
                    "novel",
                    "story",
                ],
                "secondary": ["entertainment", "fun", "enjoy", "watch", "listen", "play"],
            },
            TopicCategory.MEMES: {
                "primary": [
                    "meme",
                    "viral",
                    "trending",
                    "based",
                    "cringe",
                    "sus",
                    "salty",
                    "triggered",
                    "woke",
                    "karen",
                    "boomer",
                    "ok boomer",
                    "big mood",
                ],
                "secondary": ["funny", "hilarious", "joke", "humor", "comedy", "ridiculous"],
            },
            TopicCategory.SOCIAL_MEDIA: {
                "primary": [
                    "twitter",
                    "tweet",
                    "retweet",
                    "tiktok",
                    "instagram",
                    "facebook",
                    "reddit",
                    "youtube",
                    "social media",
                    "post",
                    "share",
                    "viral",
                ],
                "secondary": ["follow", "follower", "like", "comment", "subscribe", "tag"],
            },
        }

    def _init_academic_indicators(self) -> None:
        """Initialize academic language indicators."""
        self.academic_indicators = [
            "analysis",
            "theory",
            "framework",
            "paradigm",
            "methodology",
            "thesis",
            "argument",
            "critique",
            "synthesis",
            "dialectic",
            "discourse",
            "praxis",
            "furthermore",
            "moreover",
            "however",
            "nevertheless",
            "consequently",
            "therefore",
            "thus",
            "hence",
            "accordingly",
            "subsequently",
        ]

    def _init_humor_patterns(self) -> None:
        """Initialize humor and internet culture patterns."""
        self.humor_patterns = {
            "internet_slang": [
                "lmao",
                "lol",
                "lmfao",
                "rofl",
                "omg",
                "wtf",
                "bruh",
                "fr",
                "ngl",
                "tbh",
                "imo",
                "imho",
                "smh",
                "fml",
                "yolo",
                "af",
                "lowkey",
                "highkey",
            ],
            "humor_expressions": [
                "haha",
                "hehe",
                "lololol",
                "ahahaha",
                "bahahaha",
                "dead",
                "dying",
                "crying",
                "weak",
                "can't even",
                "i can't",
                "im dead",
            ],
            "sarcasm_indicators": [
                "yeah right",
                "sure thing",
                "oh really",
                "how shocking",
                "what a surprise",
                "totally",
                "absolutely",
                "definitely",
                "obviously",
                "clearly",
            ],
        }

    def detect_message_topics(
        self, message: str, context_messages: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Detect topics in a single message with optional context."""
        if not isinstance(message, str) or not message.strip():
            return {"primary_topic": TopicCategory.OTHER, "confidence": 0.0, "keywords_matched": []}

        message_lower = message.lower()
        topic_scores = defaultdict(float)
        matched_keywords = defaultdict(list)

        # Score each topic category
        for topic, keyword_groups in self.topic_keywords.items():
            score = 0.0

            # Primary keywords (higher weight)
            for keyword in keyword_groups["primary"]:
                if keyword in message_lower:
                    score += 2.0
                    matched_keywords[topic].append(keyword)

            # Secondary keywords (lower weight)
            for keyword in keyword_groups["secondary"]:
                if keyword in message_lower:
                    score += 1.0
                    matched_keywords[topic].append(keyword)

            # Length bonus for longer matches
            if matched_keywords[topic]:
                score += len(set(matched_keywords[topic])) * 0.5

            topic_scores[topic] = score

        # Special handling for academic language
        academic_count = sum(
            1 for indicator in self.academic_indicators if indicator in message_lower
        )
        if academic_count >= 2:
            topic_scores[TopicCategory.ACADEMIC] += academic_count * 1.5

        # URL detection for social media
        url_pattern = r"https?://[^\s]+"
        urls = re.findall(url_pattern, message)
        if urls:
            for url in urls:
                domain = urlparse(url).netloc.lower()
                if any(platform in domain for platform in ["twitter.com", "x.com", "t.co"]):
                    topic_scores[TopicCategory.SOCIAL_MEDIA] += 3.0
                elif any(
                    platform in domain for platform in ["youtube.com", "youtu.be", "tiktok.com"]
                ):
                    topic_scores[TopicCategory.ENTERTAINMENT] += 2.0
                elif any(platform in domain for platform in ["reddit.com"]):
                    topic_scores[TopicCategory.MEMES] += 2.0

        # Humor detection
        humor_score = 0.0
        for category, indicators in self.humor_patterns.items():
            matches = sum(1 for indicator in indicators if indicator in message_lower)
            humor_score += matches

        if humor_score > 0:
            topic_scores[TopicCategory.HUMOR] += humor_score * 1.5

        # Determine primary topic
        if not topic_scores:
            primary_topic = TopicCategory.OTHER
            confidence = 0.0
        else:
            primary_topic = max(topic_scores.items(), key=lambda x: x[1])[0]
            confidence = min(topic_scores[primary_topic] / 5.0, 1.0)  # Normalize to 0-1

        return {
            "primary_topic": primary_topic,
            "confidence": confidence,
            "topic_scores": dict(topic_scores),
            "keywords_matched": dict(matched_keywords),
            "academic_indicators": academic_count,
            "humor_score": humor_score,
            "contains_urls": len(urls) > 0,
            "url_count": len(urls),
        }

    def track_topic_transitions(
        self, messages_df: pd.DataFrame, time_threshold_minutes: int = 15
    ) -> list[TopicTransition]:
        """Track how topics transition within conversations."""
        logger.info("Tracking topic transitions in conversations")

        transitions = []

        # Group by thread and analyze topic flow
        for thread_id in messages_df["thread_id"].unique():
            thread_messages = messages_df[messages_df["thread_id"] == thread_id].sort_values(
                "date_sent"
            )

            if len(thread_messages) < 3:
                continue

            # Analyze topics in sequence
            previous_topic = None
            previous_time = None

            for _, message in thread_messages.iterrows():
                body = message.get("body", "")
                timestamp = pd.to_datetime(message.get("date_sent"), unit="ms")

                if not isinstance(body, str) or not body.strip():
                    continue

                # Detect topic
                topic_analysis = self.detect_message_topics(body)
                current_topic = topic_analysis["primary_topic"]

                # Check for transition
                if (
                    previous_topic
                    and current_topic != previous_topic
                    and current_topic != TopicCategory.OTHER
                    and topic_analysis["confidence"] > 0.3
                ):

                    # Calculate time gap
                    time_gap = (
                        (timestamp - previous_time).total_seconds() / 60 if previous_time else 0
                    )

                    # Determine transition method
                    transition_method = self._classify_transition_method(
                        body, topic_analysis, time_gap, time_threshold_minutes
                    )

                    # Calculate smoothness (how naturally the topic changed)
                    smoothness = self._calculate_transition_smoothness(
                        previous_topic, current_topic, transition_method, time_gap
                    )

                    transition = TopicTransition(
                        from_topic=previous_topic,
                        to_topic=current_topic,
                        transition_method=transition_method,
                        trigger_message=body[:100] + "..." if len(body) > 100 else body,
                        transition_smoothness=smoothness,
                    )

                    transitions.append(transition)

                if topic_analysis["confidence"] > 0.3:
                    previous_topic = current_topic
                    previous_time = timestamp

        logger.info(f"Tracked {len(transitions)} topic transitions")
        return transitions

    def _classify_transition_method(
        self, message: str, topic_analysis: dict, time_gap: float, threshold: int
    ) -> str:
        """Classify how a topic transition occurred."""
        # URL-triggered transition
        if topic_analysis.get("contains_urls", False):
            return "media_triggered"

        # Question-triggered transition
        if "?" in message:
            return "question_triggered"

        # Time-based classification
        if time_gap > threshold:
            return "time_gap"
        elif time_gap < 2:
            return "abrupt"
        else:
            return "gradual"

    def _calculate_transition_smoothness(
        self, from_topic: TopicCategory, to_topic: TopicCategory, method: str, time_gap: float
    ) -> float:
        """Calculate how smooth a topic transition was (0-1)."""
        base_smoothness = 0.5

        # Related topics transition more smoothly
        related_groups = [
            {TopicCategory.POLITICS, TopicCategory.POLITICAL_THEORY, TopicCategory.CURRENT_EVENTS},
            {TopicCategory.ACADEMIC, TopicCategory.POLITICAL_THEORY, TopicCategory.PHILOSOPHY},
            {TopicCategory.MEMES, TopicCategory.HUMOR, TopicCategory.SOCIAL_MEDIA},
            {TopicCategory.FOOD, TopicCategory.PERSONAL_LIFE},
            {TopicCategory.TECHNOLOGY, TopicCategory.WORK},
        ]

        for group in related_groups:
            if from_topic in group and to_topic in group:
                base_smoothness += 0.3
                break

        # Transition method affects smoothness
        method_modifiers = {
            "gradual": 0.2,
            "question_triggered": 0.1,
            "media_triggered": -0.1,
            "abrupt": -0.3,
            "time_gap": 0.0,
        }

        base_smoothness += method_modifiers.get(method, 0.0)

        # Time gap affects smoothness
        if time_gap < 1:  # Very quick transition
            base_smoothness -= 0.2
        elif 1 <= time_gap <= 5:  # Natural timing
            base_smoothness += 0.1

        return max(0.0, min(1.0, base_smoothness))

    def analyze_conversation_topics(
        self, messages_df: pd.DataFrame, window_size: int = 10
    ) -> dict[str, Any]:
        """Analyze topics across entire conversation dataset."""
        logger.info("Analyzing conversation topics across dataset")

        topic_distribution = Counter()
        topic_by_sender = defaultdict(Counter)
        topic_by_thread = defaultdict(Counter)
        academic_discussions = []
        political_discussions = []

        # Analyze each message
        for _, message in messages_df.iterrows():
            body = message.get("body", "")
            sender_id = message.get("from_recipient_id")
            thread_id = message.get("thread_id")

            if not isinstance(body, str) or not body.strip():
                continue

            topic_analysis = self.detect_message_topics(body)
            primary_topic = topic_analysis["primary_topic"]

            if topic_analysis["confidence"] > 0.3:
                topic_distribution[primary_topic] += 1
                topic_by_sender[sender_id][primary_topic] += 1
                topic_by_thread[thread_id][primary_topic] += 1

                # Collect academic discussions
                if (
                    primary_topic == TopicCategory.ACADEMIC
                    or topic_analysis["academic_indicators"] >= 2
                ):
                    academic_discussions.append(
                        {
                            "message_id": message.get("_id"),
                            "sender_id": sender_id,
                            "thread_id": thread_id,
                            "text": body,
                            "academic_score": topic_analysis["academic_indicators"],
                            "timestamp": message.get("date_sent"),
                        }
                    )

                # Collect political discussions
                if primary_topic in [TopicCategory.POLITICS, TopicCategory.POLITICAL_THEORY]:
                    political_discussions.append(
                        {
                            "message_id": message.get("_id"),
                            "sender_id": sender_id,
                            "thread_id": thread_id,
                            "text": body,
                            "topic": primary_topic,
                            "confidence": topic_analysis["confidence"],
                            "keywords": topic_analysis["keywords_matched"].get(primary_topic, []),
                            "timestamp": message.get("date_sent"),
                        }
                    )

        # Track transitions
        transitions = self.track_topic_transitions(messages_df)

        return {
            "topic_distribution": dict(topic_distribution),
            "topic_by_sender": {str(k): dict(v) for k, v in topic_by_sender.items()},
            "topic_by_thread": {str(k): dict(v) for k, v in topic_by_thread.items()},
            "academic_discussions": academic_discussions,
            "political_discussions": political_discussions,
            "topic_transitions": [t.__dict__ for t in transitions],
            "transition_patterns": self._analyze_transition_patterns(transitions),
            "topic_engagement_scores": self._calculate_topic_engagement(topic_by_sender),
        }

    def _analyze_transition_patterns(self, transitions: list[TopicTransition]) -> dict[str, Any]:
        """Analyze patterns in topic transitions."""
        if not transitions:
            return {}

        # Most common transition pairs
        transition_pairs = Counter()
        transition_methods = Counter()
        smoothness_scores = []

        for transition in transitions:
            pair = (transition.from_topic, transition.to_topic)
            transition_pairs[pair] += 1
            transition_methods[transition.transition_method] += 1
            smoothness_scores.append(transition.transition_smoothness)

        avg_smoothness = sum(smoothness_scores) / len(smoothness_scores) if smoothness_scores else 0

        return {
            "most_common_transitions": transition_pairs.most_common(10),
            "transition_methods": dict(transition_methods),
            "avg_transition_smoothness": avg_smoothness,
            "total_transitions": len(transitions),
        }

    def _calculate_topic_engagement(self, topic_by_sender: dict) -> dict[str, float]:
        """Calculate how engaged each sender is with different topics."""
        engagement_scores = {}

        for sender_id, topics in topic_by_sender.items():
            total_messages = sum(topics.values())
            if total_messages == 0:
                continue

            # Calculate entropy to measure topic diversity
            topic_probs = [count / total_messages for count in topics.values()]
            entropy = -sum(p * (p.bit_length() - 1) for p in topic_probs if p > 0)

            # Calculate political engagement
            political_messages = topics.get(TopicCategory.POLITICS, 0) + topics.get(
                TopicCategory.POLITICAL_THEORY, 0
            )
            political_engagement = political_messages / total_messages

            # Calculate academic engagement
            academic_engagement = topics.get(TopicCategory.ACADEMIC, 0) / total_messages

            engagement_scores[str(sender_id)] = {
                "topic_diversity": entropy,
                "political_engagement": political_engagement,
                "academic_engagement": academic_engagement,
                "total_messages": total_messages,
                "dominant_topics": sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3],
            }

        return engagement_scores

    def generate_topic_summary(self, topic_analysis: dict[str, Any]) -> str:
        """Generate a natural language summary of topic analysis."""
        distribution = topic_analysis["topic_distribution"]
        transitions = topic_analysis.get("transition_patterns", {})

        if not distribution:
            return "No significant topics detected in conversations."

        # Find dominant topics
        sorted_topics = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        top_topics = sorted_topics[:3]

        summary_parts = []

        # Main topics
        topic_names = [topic.value.replace("_", " ").title() for topic, _ in top_topics]
        summary_parts.append(f"Primary discussion topics: {', '.join(topic_names)}")

        # Political engagement
        political_count = distribution.get(TopicCategory.POLITICS, 0) + distribution.get(
            TopicCategory.POLITICAL_THEORY, 0
        )
        total_messages = sum(distribution.values())
        if political_count > 0:
            political_pct = (political_count / total_messages) * 100
            summary_parts.append(
                f"Political discussions comprise {political_pct:.1f}% of conversations"
            )

        # Academic content
        academic_count = len(topic_analysis.get("academic_discussions", []))
        if academic_count > 0:
            summary_parts.append(
                f"{academic_count} messages contain academic or theoretical language"
            )

        # Transitions
        if transitions.get("total_transitions", 0) > 0:
            avg_smoothness = transitions.get("avg_transition_smoothness", 0)
            smoothness_desc = (
                "smooth"
                if avg_smoothness > 0.6
                else "abrupt" if avg_smoothness < 0.4 else "moderate"
            )
            summary_parts.append(f"Topic transitions are generally {smoothness_desc}")

        return ". ".join(summary_parts) + "."
