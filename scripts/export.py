#!/usr/bin/env python3
"""
Model export script for Astrabot.

This script exports trained models to various formats:
- Hugging Face format
- GGUF format for llama.cpp
- ONNX format for deployment
- Merged model (LoRA + base)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export Astrabot models to various formats")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model",
    )

    parser.add_argument(
        "--export-format",
        choices=["hf", "gguf", "onnx", "merged", "all"],
        default="hf",
        help="Export format",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./exports",
        help="Output directory for exported models",
    )

    parser.add_argument(
        "--quantization",
        choices=["none", "int8", "int4", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"],
        default="none",
        help="Quantization method for GGUF export",
    )

    parser.add_argument(
        "--merge-with-base",
        action="store_true",
        help="Merge LoRA weights with base model",
    )

    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push exported model to Hugging Face Hub",
    )

    parser.add_argument(
        "--hub-repo",
        type=str,
        help="Hugging Face Hub repository name",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Make Hub repository private",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def export_to_hf(model_path: Path, output_path: Path):
    """Export model in Hugging Face format."""
    logger.info("Exporting to Hugging Face format...")

    hf_output = output_path / "huggingface"
    hf_output.mkdir(parents=True, exist_ok=True)

    # Copy model files
    for file in model_path.iterdir():
        if file.is_file():
            shutil.copy2(file, hf_output / file.name)

    logger.info(f"Exported HF model to {hf_output}")
    return hf_output


def export_to_gguf(model_path: Path, output_path: Path, quantization: str = "q4_0"):
    """Export model to GGUF format for llama.cpp."""
    logger.info(f"Exporting to GGUF format with {quantization} quantization...")

    try:
        import subprocess
    except ImportError:
        logger.error("subprocess module not available")
        return None

    gguf_output = output_path / "gguf"
    gguf_output.mkdir(parents=True, exist_ok=True)

    # Check if llama.cpp convert script is available
    convert_script = Path.home() / "llama.cpp" / "convert.py"
    if not convert_script.exists():
        logger.warning("llama.cpp convert.py not found. Please install llama.cpp first.")
        logger.warning("git clone https://github.com/ggerganov/llama.cpp")
        return None

    # Convert to GGUF
    output_file = gguf_output / f"model-{quantization}.gguf"

    cmd = [
        sys.executable,
        str(convert_script),
        str(model_path),
        "--outfile",
        str(output_file),
        "--outtype",
        quantization,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Exported GGUF model to {output_file}")
            return output_file
        else:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Error during GGUF conversion: {e}")
        return None


def export_to_onnx(model_path: Path, output_path: Path):
    """Export model to ONNX format."""
    logger.info("Exporting to ONNX format...")

    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing required dependencies for ONNX export: {e}")
        logger.error("Please install: pip install optimum[onnxruntime]")
        return None

    onnx_output = output_path / "onnx"
    onnx_output.mkdir(parents=True, exist_ok=True)

    try:
        # Load model
        AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Export to ONNX
        ort_model = ORTModelForCausalLM.from_pretrained(
            model_path, export=True, provider="CPUExecutionProvider"
        )

        # Save ONNX model
        ort_model.save_pretrained(onnx_output)
        tokenizer.save_pretrained(onnx_output)

        logger.info(f"Exported ONNX model to {onnx_output}")
        return onnx_output

    except Exception as e:
        logger.error(f"Error during ONNX export: {e}")
        return None


def merge_lora_weights(model_path: Path, output_path: Path):
    """Merge LoRA weights with base model."""
    logger.info("Merging LoRA weights with base model...")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        return None

    merged_output = output_path / "merged"
    merged_output.mkdir(parents=True, exist_ok=True)

    try:
        # Load base model and LoRA weights
        logger.info("Loading model and LoRA weights...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )

        # If this is a LoRA model, merge it
        if (model_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info("Merging LoRA weights...")
            model = model.merge_and_unload()
        else:
            model = base_model
            logger.info("Model does not appear to use LoRA, copying as-is")

        # Save merged model
        model.save_pretrained(merged_output)

        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.save_pretrained(merged_output)

        logger.info(f"Saved merged model to {merged_output}")
        return merged_output

    except Exception as e:
        logger.error(f"Error during merge: {e}")
        return None


def push_to_hub(model_path: Path, repo_name: str, private: bool = True):
    """Push model to Hugging Face Hub."""
    logger.info(f"Pushing model to Hugging Face Hub: {repo_name}")

    try:
        from huggingface_hub import HfApi, create_repo
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        logger.error(f"Missing required dependencies: {e}")
        logger.error("Please install: pip install huggingface-hub")
        return False

    try:
        # Create repo if it doesn't exist
        HfApi()
        try:
            create_repo(repo_name, private=private, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")

        # Load and push model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model.push_to_hub(repo_name, private=private)
        tokenizer.push_to_hub(repo_name, private=private)

        logger.info(f"Successfully pushed model to https://huggingface.co/{repo_name}")
        return True

    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")
        return False


def main():
    """Main export pipeline."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logger.info("Starting Astrabot model export")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export based on format
    exported_paths = []

    if args.export_format in ["hf", "all"]:
        hf_path = export_to_hf(model_path, output_path)
        if hf_path:
            exported_paths.append(("huggingface", hf_path))

    if args.export_format in ["gguf", "all"]:
        gguf_path = export_to_gguf(model_path, output_path, args.quantization)
        if gguf_path:
            exported_paths.append(("gguf", gguf_path))

    if args.export_format in ["onnx", "all"]:
        onnx_path = export_to_onnx(model_path, output_path)
        if onnx_path:
            exported_paths.append(("onnx", onnx_path))

    if args.export_format in ["merged", "all"] or args.merge_with_base:
        merged_path = merge_lora_weights(model_path, output_path)
        if merged_path:
            exported_paths.append(("merged", merged_path))

    # Push to hub if requested
    if args.push_to_hub and args.hub_repo:
        # Use merged model if available, otherwise use original
        push_path = model_path
        for format_name, path in exported_paths:
            if format_name == "merged":
                push_path = path
                break

        push_to_hub(push_path, args.hub_repo, args.private)

    # Summary
    logger.info("\nExport Summary:")
    logger.info("=" * 50)
    for format_name, path in exported_paths:
        logger.info(f"{format_name}: {path}")

    if not exported_paths:
        logger.warning("No models were successfully exported")
    else:
        logger.info(f"\nAll exports saved to: {output_path}")


if __name__ == "__main__":
    main()
