#!/bin/bash
# Setup script for linting and formatting tools

set -e

echo "Setting up linting and formatting tools..."

# Install dependencies
echo "Installing linting and formatting tools..."
uv add --dev black flake8 flake8-docstrings flake8-bugbear isort mypy autoflake

# Install pre-commit if not already installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    uv add --dev pre-commit
    pre-commit install
fi

# Create a VSCode settings file for linting and formatting
VSCODE_SETTINGS_DIR=".vscode"
VSCODE_SETTINGS_FILE="$VSCODE_SETTINGS_DIR/settings.json"

if [ ! -d "$VSCODE_SETTINGS_DIR" ]; then
    mkdir -p "$VSCODE_SETTINGS_DIR"
fi

if [ ! -f "$VSCODE_SETTINGS_FILE" ]; then
    echo "Creating VSCode settings for linting and formatting..."
    cat > "$VSCODE_SETTINGS_FILE" <<EOL
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.linting.flake8Args": [
        "--config=.flake8"
    ],
    "python.linting.mypyArgs": [
        "--config-file=pyproject.toml"
    ],
    "isort.args": ["--profile", "black", "--line-length", "100"],
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
EOL
else
    echo "VSCode settings file already exists. Skipping creation."
fi

echo "✓ Linting and formatting tools setup complete!"
echo
echo "To fix existing issues, run:"
echo "  just fix-issues"
echo
echo "To run all code quality checks:"
echo "  just all"
echo
echo "For automatic checks on commit:"
echo "  pre-commit install"
