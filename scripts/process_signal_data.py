#!/usr/bin/env python3
"""
Script to process Signal data and create training dataset.

Usage:
    python scripts/process_signal_data.py --input data/raw/signal-flatfiles --output outputs/training_data.json
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm import create_training_data_from_signal
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process Signal data and create training dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/signal-flatfiles",
        help="Path to Signal CSV files directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/training_data.json",
        help="Output path for training data JSON",
    )
    parser.add_argument(
        "--recipient-id",
        type=int,
        default=2,
        help="Your recipient ID in Signal (default: 2)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of training examples to create",
    )
    parser.add_argument(
        "--no-twitter",
        action="store_true",
        help="Disable Twitter content extraction",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging

        get_logger().setLevel(logging.DEBUG)

    # Validate input directory
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    # Check for required files
    messages_path = input_path / "signal.csv"
    recipients_path = input_path / "recipient.csv"

    if not messages_path.exists():
        logger.error(f"Messages file not found: {messages_path}")
        sys.exit(1)

    if not recipients_path.exists():
        logger.error(f"Recipients file not found: {recipients_path}")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process data
    logger.info("Starting Signal data processing...")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Your recipient ID: {args.recipient_id}")

    if args.max_examples:
        logger.info(f"Max examples: {args.max_examples}")

    if args.no_twitter:
        logger.info("Twitter content extraction: DISABLED")

    try:
        result = create_training_data_from_signal(
            messages_csv_path=str(messages_path),
            recipients_csv_path=str(recipients_path),
            output_path=str(output_path),
            your_recipient_id=args.recipient_id,
            include_twitter=not args.no_twitter,
            max_examples=args.max_examples,
        )

        if result["success"]:
            logger.info("✅ Processing complete!")
            logger.info(f"Created {result['total_examples']} training examples")
            logger.info(f"Output saved to: {result['output_path']}")

            # Show style analysis
            style = result["style_analysis"]
            logger.info("\n📊 Your Communication Style:")
            logger.info(f"  Average message length: {style['avg_message_length']:.1f} chars")
            logger.info(f"  Preferred style: {style['preferred_length']}")
            logger.info(f"  Burst frequency: {style['burst_patterns']['burst_frequency']:.2%}")
            logger.info(f"  Emoji usage: {style['emoji_usage']['emoji_usage_rate']}")
        else:
            logger.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
