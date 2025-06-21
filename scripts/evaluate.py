#!/usr/bin/env python3
"""
Model evaluation script for Astrabot.

This script evaluates trained models on various metrics including:
- Perplexity
- Style similarity
- Response quality
- Conversation coherence
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Astrabot model")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data JSON file",
    )

    parser.add_argument(
        "--eval-mode",
        choices=["perplexity", "style", "interactive", "all"],
        default="all",
        help="Evaluation mode",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="evaluation_results.json",
        help="Output file for evaluation results",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def load_model(model_path: str):
    """Load the trained model and tokenizer."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def evaluate_perplexity(model, tokenizer, test_data: list[dict[str, Any]]) -> float:
    """Calculate perplexity on test data."""
    import torch

    logger.info("Evaluating perplexity...")

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for example in test_data:
            text = example.get("text", "")
            if not text:
                continue

            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Perplexity: {perplexity:.2f}")
    return perplexity


def evaluate_style_similarity(
    model, tokenizer, test_data: list[dict[str, Any]]
) -> dict[str, float]:
    """Evaluate how well the model captures communication style."""
    logger.info("Evaluating style similarity...")

    style_metrics = {
        "avg_response_length": 0,
        "burst_detection_accuracy": 0,
        "emoji_usage_similarity": 0,
        "vocabulary_overlap": 0,
    }

    generated_responses = []
    expected_responses = []

    for example in test_data[:50]:  # Limit for efficiency
        input_text = example.get("input", "")
        expected = example.get("output", "")

        if not input_text or not expected:
            continue

        # Generate response
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(input_text) :].strip()  # Remove input from output

        generated_responses.append(generated)
        expected_responses.append(expected)

    # Calculate style metrics
    if generated_responses:
        # Average response length
        avg_gen_len = sum(len(r) for r in generated_responses) / len(generated_responses)
        avg_exp_len = sum(len(r) for r in expected_responses) / len(expected_responses)
        style_metrics["avg_response_length"] = 1 - abs(avg_gen_len - avg_exp_len) / max(
            avg_gen_len, avg_exp_len
        )

        # Burst detection (check for [NEXT] or [CONTINUE] tokens)
        burst_gen = sum(1 for r in generated_responses if "[NEXT]" in r or "[CONTINUE]" in r)
        burst_exp = sum(1 for r in expected_responses if "[NEXT]" in r or "[CONTINUE]" in r)
        style_metrics["burst_detection_accuracy"] = 1 - abs(burst_gen - burst_exp) / max(
            len(generated_responses), 1
        )

        # Emoji usage
        import re

        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]"
        )

        emoji_gen = sum(1 for r in generated_responses if emoji_pattern.search(r))
        emoji_exp = sum(1 for r in expected_responses if emoji_pattern.search(r))
        style_metrics["emoji_usage_similarity"] = 1 - abs(emoji_gen - emoji_exp) / max(
            len(generated_responses), 1
        )

        # Vocabulary overlap
        from collections import Counter

        gen_words = Counter(" ".join(generated_responses).lower().split())
        exp_words = Counter(" ".join(expected_responses).lower().split())

        common_words = set(gen_words.keys()) & set(exp_words.keys())
        all_words = set(gen_words.keys()) | set(exp_words.keys())

        style_metrics["vocabulary_overlap"] = len(common_words) / len(all_words) if all_words else 0

    logger.info(f"Style metrics: {style_metrics}")
    return style_metrics


def interactive_evaluation(model, tokenizer):
    """Interactive evaluation mode for manual testing."""
    logger.info("Starting interactive evaluation mode...")
    print("\nInteractive Evaluation Mode")
    print("Type 'quit' to exit")
    print("-" * 50)

    import torch

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        # Generate response
        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(user_input) :].strip()

        print(f"Model: {response}")


def save_results(results: dict[str, Any], output_file: str):
    """Save evaluation results to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting Astrabot model evaluation")

    # Load model
    model, tokenizer = load_model(args.model_path)

    results = {
        "model_path": args.model_path,
        "evaluation_mode": args.eval_mode,
    }

    # Load test data if provided
    test_data = []
    if args.test_data:
        with open(args.test_data) as f:
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} test examples")
        test_data = test_data[: args.max_samples]

    # Run evaluations
    if args.eval_mode in ["perplexity", "all"] and test_data:
        results["perplexity"] = evaluate_perplexity(model, tokenizer, test_data)

    if args.eval_mode in ["style", "all"] and test_data:
        results["style_metrics"] = evaluate_style_similarity(model, tokenizer, test_data)

    if args.eval_mode == "interactive":
        interactive_evaluation(model, tokenizer)

    # Save results
    if args.eval_mode != "interactive":
        save_results(results, args.output_file)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    # Ensure we have torch for evaluation
    try:
        import torch
    except ImportError:
        print("PyTorch is required for model evaluation. Please install it:")
        print("pip install torch")
        sys.exit(1)

    main()
