"""
Training data formatting for Astrabot.

This module provides functions to format conversational training data for various
model training frameworks, particularly Unsloth and Hugging Face models.
"""

from typing import Any

import pandas as pd
from datasets import Dataset


def format_conversational_for_training(training_data: list[dict[str, Any]], tokenizer) -> Dataset:
    """
    Format conversational training data for Unsloth/model training.

    Handles different conversation types appropriately:
    - Burst sequences are formatted to preserve multi-message nature
    - Role-based responses use appropriate system prompts
    - Conversation windows maintain natural flow

    Args:
        training_data: List of conversational training examples
        tokenizer: The tokenizer to use for formatting

    Returns:
        Dataset ready for training
    """
    formatted_data = []

    for example in training_data:
        metadata = example["metadata"]

        # Create appropriate system prompt based on conversation type
        if metadata["type"] == "burst_sequence":
            system_prompt = "You are an AI that naturally sends multiple messages in quick succession when expressing complex thoughts or emotions. Use [NEXT] to separate messages in a burst."
        elif metadata["type"] == "conversation_starter":
            system_prompt = "You are an AI that initiates conversations naturally and engagingly."
        elif metadata["type"] == "role_based_response":
            role = metadata["role"]
            role_descriptions = {
                "conversation_driver": "You lead conversations with engaging topics and questions.",
                "responsive_participant": "You respond thoughtfully to others' messages.",
                "active_engager": "You actively participate in discussions with enthusiasm.",
                "balanced_conversationalist": "You maintain balanced, natural conversation flow.",
            }
            system_prompt = (
                f"You are an AI that {role_descriptions.get(role, 'communicates naturally')}."
            )
        else:
            system_prompt = "You are an AI that communicates in a natural, conversational style."

        # Build the conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Add metadata for potential filtering or weighting during training
        formatted_data.append({"text": text, "metadata": metadata})

    return Dataset.from_list(formatted_data)


def format_for_alpaca(training_data: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Format training data in Alpaca/Vicuna instruction format.

    Args:
        training_data: List of training examples

    Returns:
        List of examples in Alpaca format
    """
    alpaca_data = []

    for example in training_data:
        # Map our format to Alpaca format
        alpaca_example = {
            "instruction": example.get("instruction", "Continue the conversation naturally"),
            "input": example.get("input", ""),
            "output": example.get("output", ""),
        }

        # Handle special cases based on metadata
        if "metadata" in example:
            metadata = example["metadata"]

            # Add context about conversation style if burst sequence
            if metadata.get("type") == "burst_sequence":
                alpaca_example["instruction"] += " (Send multiple short messages if appropriate)"

            # Add timing context if available
            if "response_delay" in metadata:
                delay = metadata["response_delay"]
                if delay < 60:
                    alpaca_example["instruction"] += " (Respond immediately)"
                elif delay < 3600:
                    alpaca_example["instruction"] += " (Quick response)"

        alpaca_data.append(alpaca_example)

    return alpaca_data


def format_for_chat_completion(
    training_data: list[dict[str, Any]], include_system_prompt: bool = True
) -> list[dict[str, list]]:
    """
    Format training data for OpenAI-style chat completion format.

    Args:
        training_data: List of training examples
        include_system_prompt: Whether to include system prompts

    Returns:
        List of examples in chat completion format
    """
    chat_data = []

    for example in training_data:
        messages = []

        # Add system prompt if requested
        if include_system_prompt:
            system_content = (
                "You are a helpful assistant that communicates in a natural, conversational style."
            )

            # Customize based on metadata
            if "metadata" in example:
                metadata = example["metadata"]
                if metadata.get("type") == "burst_sequence":
                    system_content += " You sometimes send multiple short messages in succession."
                elif metadata.get("role") == "conversation_driver":
                    system_content += (
                        " You enjoy leading conversations and asking engaging questions."
                    )

            messages.append({"role": "system", "content": system_content})

        # Add user and assistant messages
        messages.append({"role": "user", "content": example.get("input", "")})
        messages.append({"role": "assistant", "content": example.get("output", "")})

        chat_data.append({"messages": messages})

    return chat_data


def split_burst_sequences(training_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Split burst sequences into individual messages while maintaining context.

    This is useful for training models that don't support the [NEXT] token.

    Args:
        training_data: List of training examples

    Returns:
        List with burst sequences expanded into individual examples
    """
    expanded_data = []

    for example in training_data:
        if example.get("metadata", {}).get("type") == "burst_sequence":
            # Split the output by [NEXT] token
            messages = example["output"].split(" [NEXT] ")

            if len(messages) > 1:
                # Create an example for each message in the burst
                for i, message in enumerate(messages):
                    new_example = example.copy()
                    new_example["output"] = message
                    new_example["metadata"] = example["metadata"].copy()
                    new_example["metadata"]["burst_position"] = i + 1
                    new_example["metadata"]["burst_total"] = len(messages)

                    # Update instruction to indicate position in burst
                    if i == 0:
                        new_example["instruction"] = "Start a burst of messages"
                    elif i == len(messages) - 1:
                        new_example["instruction"] = "Send the final message in the burst"
                    else:
                        new_example["instruction"] = "Continue the burst sequence"

                    expanded_data.append(new_example)
            else:
                # Not actually a burst, just add as-is
                expanded_data.append(example)
        else:
            # Not a burst sequence, add as-is
            expanded_data.append(example)

    return expanded_data


def filter_by_quality(
    training_data: list[dict[str, Any]],
    min_output_length: int = 10,
    max_output_length: int = 1000,
    min_input_length: int = 5,
) -> list[dict[str, Any]]:
    """
    Filter training examples by quality criteria.

    Args:
        training_data: List of training examples
        min_output_length: Minimum output length
        max_output_length: Maximum output length
        min_input_length: Minimum input length

    Returns:
        Filtered list of training examples
    """
    filtered_data = []

    for example in training_data:
        input_text = example.get("input", "")
        output_text = example.get("output", "")

        # Check length criteria
        if (
            len(input_text) >= min_input_length
            and min_output_length <= len(output_text) <= max_output_length
        ):

            # Additional quality checks
            if output_text.lower() not in ["ok", "okay", "yes", "no", "lol", "haha"]:
                filtered_data.append(example)

    return filtered_data


def create_weighted_dataset(
    training_data: list[dict[str, Any]], weight_by: str = "type"
) -> Dataset:
    """
    Create a weighted dataset where certain types of examples appear more frequently.

    Args:
        training_data: List of training examples
        weight_by: Metadata field to use for weighting

    Returns:
        Weighted dataset
    """
    # Define weights for different types
    weights = {
        "conversation_window": 1.0,
        "burst_sequence": 1.5,  # Emphasize burst patterns
        "role_based_response": 1.2,
        "conversation_starter": 2.0,  # Emphasize conversation initiation
        "single_message": 0.8,
        "long_form": 1.0,
        "double_tap": 1.0,
    }

    weighted_data = []

    for example in training_data:
        # Get the weight for this example
        example_type = example.get("metadata", {}).get(weight_by, "default")
        weight = weights.get(example_type, 1.0)

        # Duplicate examples based on weight
        num_copies = int(weight)
        for _ in range(num_copies):
            weighted_data.append(example)

        # Add partial copy based on decimal part
        if weight % 1 > 0 and pd.np.random.random() < (weight % 1):
            weighted_data.append(example)

    return Dataset.from_list(weighted_data)


def prepare_for_unsloth(
    training_data: list[dict[str, Any]], tokenizer, max_seq_length: int = 2048
) -> Dataset:
    """
    Prepare training data specifically for Unsloth fine-tuning.

    Args:
        training_data: List of training examples
        tokenizer: The tokenizer to use
        max_seq_length: Maximum sequence length

    Returns:
        Dataset formatted for Unsloth
    """
    # First format the conversations
    dataset = format_conversational_for_training(training_data, tokenizer)

    # Apply quality filtering
    filtered_examples = filter_by_quality([ex for ex in dataset])

    # Create final dataset
    final_dataset = Dataset.from_list(filtered_examples)

    # Add length filtering
    def filter_by_length(example):
        return len(tokenizer.encode(example["text"])) <= max_seq_length

    final_dataset = final_dataset.filter(filter_by_length)

    return final_dataset
