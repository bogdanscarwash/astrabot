#!/bin/bash
# Development environment setup script for Astrabot

set -e

echo "🚀 Setting up Astrabot development environment..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "✅ Python $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev]" || pip install pytest pytest-cov pytest-mock black flake8 mypy pre-commit

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,cache,logs}
mkdir -p outputs/{models,checkpoints,reports}
mkdir -p logs

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
else
    echo "✅ .env file exists"
fi

# Run tests to verify setup
echo "🧪 Running tests..."
pytest tests/unit/test_utils_logging.py -v || echo "⚠️  Some tests failed - this is expected if you haven't set up everything yet"

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run 'source venv/bin/activate' to activate the environment"
echo "3. Run 'make test' to run all tests"
echo "4. Run 'jupyter notebook' to start working with notebooks"