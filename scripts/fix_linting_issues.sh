#!/bin/bash
# Script to fix common linting issues automatically

set -e

echo "Fixing linting issues in the codebase..."

# Step 1: Fix unused imports
echo "Step 1: Removing unused imports..."
autoflake --remove-all-unused-imports --recursive --in-place src/ tests/ scripts/

# Step 2: Fix unused variables
echo "Step 2: Removing unused variables..."
autoflake --remove-unused-variables --recursive --in-place src/ tests/ scripts/

# Step 3: Fix line length and formatting issues
echo "Step 3: Formatting code with black..."
black --line-length 100 src/ tests/ scripts/

# Step 4: Fix import order
echo "Step 4: Sorting imports with isort..."
isort src/ tests/ scripts/

# Step 5: Run flake8 to check remaining issues
echo "Step 5: Running flake8 to check remaining issues..."
flake8 src/ tests/ scripts/ --ignore=E501 || {
  echo "Some issues remain. Check the output above."
  echo "You may need to manually fix some issues like:"
  echo "- E402: Module level import not at top of file"
  echo "- E722: Do not use bare except"
  echo "- E741: Ambiguous variable names (like 'l')"
  echo "- F541: f-string missing placeholders"
  echo "Note: E501 (line too long) errors are now ignored as Black handles line length"
}

echo "Linting fixes applied. Run 'just lint' to check if any issues remain."
