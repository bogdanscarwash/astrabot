#!/usr/bin/env python3
"""
Configuration management for Astrabot.
Handles environment variables securely using python-dotenv.
"""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Navigate from src/utils/ to project root
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env if it exists
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # Try to load .env.example as fallback (with warning)
    example_env = PROJECT_ROOT / ".env.example"
    if example_env.exists():
        print("⚠️  Warning: .env file not found, using .env.example")
        print("   Copy .env.example to .env and update with your values")
        load_dotenv(example_env)


class Config:
    """Centralized configuration management."""

    # API Keys (sensitive - never log these!)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")

    # Model Configuration
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Signal Data
    YOUR_RECIPIENT_ID: int = int(os.getenv("YOUR_RECIPIENT_ID", "2"))
    SIGNAL_BACKUP_PATH: Optional[str] = os.getenv("SIGNAL_BACKUP_PATH")

    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output")))

    # Feature Flags
    ENABLE_IMAGE_PROCESSING: bool = os.getenv("ENABLE_IMAGE_PROCESSING", "true").lower() == "true"
    ENABLE_BATCH_PROCESSING: bool = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "10"))

    # Development
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Rate Limiting
    API_RATE_LIMIT: int = int(os.getenv("API_RATE_LIMIT", "60"))
    API_RATE_WINDOW: int = int(os.getenv("API_RATE_WINDOW", "60"))

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return getattr(cls, key, default)

    @classmethod
    def require(cls, key: str) -> Any:
        """Get a required configuration value, raise if not set."""
        value = getattr(cls, key, None)
        if value is None:
            raise ValueError(
                f"Required configuration '{key}' is not set. "
                f"Please set it in your .env file or as an environment variable."
            )
        return value

    @classmethod
    def has_openai(cls) -> bool:
        """Check if OpenAI API is configured."""
        return cls.OPENAI_API_KEY is not None

    @classmethod
    def has_anthropic(cls) -> bool:
        """Check if Anthropic API is configured."""
        return cls.ANTHROPIC_API_KEY is not None

    @classmethod
    def validate(cls) -> dict:
        """Validate configuration and return status."""
        status = {
            "openai_configured": cls.has_openai(),
            "anthropic_configured": cls.has_anthropic(),
            "data_dir_exists": cls.DATA_DIR.exists(),
            "output_dir_exists": cls.OUTPUT_DIR.exists(),
            "debug_mode": cls.DEBUG,
        }

        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        return status

    @classmethod
    def print_status(cls):
        """Print configuration status (hiding sensitive values)."""
        print("🔧 Astrabot Configuration Status")
        print("=" * 40)

        # API Keys (show only if configured, not the actual values)
        print(f"OpenAI API: {'✅ Configured' if cls.has_openai() else '❌ Not configured'}")
        print(f"Anthropic API: {'✅ Configured' if cls.has_anthropic() else '❌ Not configured'}")

        # Non-sensitive configuration
        print(f"Your Recipient ID: {cls.YOUR_RECIPIENT_ID}")
        print(f"Debug Mode: {'ON' if cls.DEBUG else 'OFF'}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Image Processing: {'Enabled' if cls.ENABLE_IMAGE_PROCESSING else 'Disabled'}")
        print(f"Batch Processing: {'Enabled' if cls.ENABLE_BATCH_PROCESSING else 'Disabled'}")
        print(f"Max Batch Size: {cls.MAX_BATCH_SIZE}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"Output Directory: {cls.OUTPUT_DIR}")


# Create a singleton instance
config = Config()

# Convenience exports
OPENAI_API_KEY = config.OPENAI_API_KEY
ANTHROPIC_API_KEY = config.ANTHROPIC_API_KEY
YOUR_RECIPIENT_ID = config.YOUR_RECIPIENT_ID


if __name__ == "__main__":
    # When run directly, show configuration status
    config.print_status()
    print("\n📋 Validation Results:")
    for key, value in config.validate().items():
        status = "✅" if value else "❌"
        print(f"  {status} {key}: {value}")
