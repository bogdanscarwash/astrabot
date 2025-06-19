#!/usr/bin/env bash
# Setup script for Astrabot development environment
# Compatible with pyenv and Debian 12

set -e  # Exit on error

echo "🤖 Astrabot Environment Setup"
echo "============================="

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "❌ pyenv is not installed!"
    echo "Please install pyenv first. See docs/pyenv-setup.md for instructions."
    exit 1
fi

# Check current directory
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ This script must be run from the Astrabot project root directory"
    exit 1
fi

# Detect Python version to use
if [[ -f ".python-version" ]]; then
    PYTHON_VERSION=$(cat .python-version)
    echo "📄 Found .python-version file: $PYTHON_VERSION"
    
    # Check if it's a valid Python version (not "qwen")
    if [[ "$PYTHON_VERSION" == "qwen" ]]; then
        echo "⚠️  .python-version contains 'qwen', using default Python 3.11.9"
        PYTHON_VERSION="3.11.9"
    fi
else
    PYTHON_VERSION="3.11.9"
    echo "📝 No .python-version file found, using default: $PYTHON_VERSION"
fi

# Check if Python version is installed
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "📦 Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
fi

# Set local Python version
echo "🐍 Setting local Python version to $PYTHON_VERSION"
pyenv local "$PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "🔧 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "✨ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
if [[ -f "requirements.txt" ]]; then
    echo "📚 Installing requirements..."
    pip install -r requirements.txt
fi

# Install test dependencies
echo "🧪 Installing test dependencies..."
pip install pytest pytest-cov pytest-mock black flake8 mypy

# Make scripts executable
echo "🔐 Making scripts executable..."
for script in test_runner.py test_conversation_utilities.py test_structured_outputs.py; do
    if [[ -f "$script" ]]; then
        chmod +x "$script"
        echo "  ✓ $script"
    fi
done

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  ./test_runner.py all"
echo "  # or"
echo "  make test"
echo ""
echo "Happy coding! 🚀"