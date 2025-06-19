#!/usr/bin/env python3
"""
Example script showing pyenv-compatible Python script structure.

This script demonstrates:
1. Proper shebang for pyenv compatibility
2. Module imports that work with the project structure
3. Command-line argument handling
4. Main function pattern
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path for imports
# This ensures we can import our project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import project modules
try:
    from conversation_utilities import extract_tweet_text
    from structured_schemas import TweetContent
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project directory")
    sys.exit(1)


def check_python_version():
    """Check and display Python version information"""
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path[0]}")
    
    # Check if we're using pyenv
    if ".pyenv" in sys.executable:
        print("✅ Running under pyenv")
    else:
        print("⚠️  Not running under pyenv (might be system Python)")


def main():
    """Main function"""
    print("🐍 Pyenv-Compatible Python Script Example")
    print("=" * 50)
    
    check_python_version()
    
    print("\nProject modules are available for use!")
    print("This script can now use all project functionality.")
    
    # Example usage
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"\nTesting with URL: {url}")
        result = extract_tweet_text(url)
        if result:
            print(f"Extracted tweet ID: {result.get('tweet_id', 'N/A')}")
    else:
        print("\nUsage: ./example-pyenv-script.py [twitter-url]")


if __name__ == "__main__":
    main()