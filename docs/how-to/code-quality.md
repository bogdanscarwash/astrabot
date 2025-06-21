# Code Quality Guidelines

This guide explains how to maintain code quality standards in the Astrabot project using the provided linting, formatting, and type checking tools.

## Setup

To set up all linting and formatting tools:

```bash
just setup-linting
```

This will:

1. Install all necessary tools (black, flake8, isort, mypy, autoflake)
2. Configure pre-commit hooks
3. Set up VSCode settings for automatic formatting and linting

## Common Commands

### Fix Issues Automatically

To automatically fix common issues (unused imports, unused variables, formatting, etc.):

```bash
just fix-issues
```

### Format Code

To format code with black and isort:

```bash
just format
```

### Lint Code

To check for linting issues with flake8:

```bash
just lint
```

### Type Check

To run type checking with mypy:

```bash
just type-check
```

### Run All Checks

To run all code quality checks:

```bash
just all
```

## Pre-commit Hooks

Pre-commit hooks run automatically when you commit code. To install:

```bash
pre-commit install
```

To run manually:

```bash
just pre-commit
```

## Understanding Linting Errors

Common error codes:

- **E402**: Module level import not at top of file
- **F401**: Import unused
- **F841**: Local variable unused
- **E226**: Missing whitespace around arithmetic operator
- **F541**: f-string missing placeholders
- **E722**: Do not use bare 'except'
- **E741**: Ambiguous variable name (like 'l')
- **F811**: Redefinition of unused name

> **Note**: E501 (line too long) errors are now ignored since Black handles line length formatting automatically.

## Ignoring Specific Issues

Sometimes you may need to ignore specific issues. You can:

1. Add a comment at the end of the line: `# noqa: E402` (for specific error codes)
2. Use per-file-ignores in `.flake8` (already configured for common patterns)
3. Wrap code with:
   ```python
   # flake8: noqa
   code_to_ignore
   # flake8: qa
   ```

The project is already configured to ignore:

- E501: Line too long (handled by Black)
- E203, W503: Black-compatible whitespace rules
- F401: Unused imports in **init**.py files
- F841: Unused variables in test files
- E402: Module level imports not at top of file

## Editor Integration

### VSCode

The setup script configures VSCode to:

- Format on save with black
- Sort imports with isort
- Show linting errors from flake8
- Show type errors from mypy

### Other Editors

For other editors, configure them to use:

- black for formatting (line length 100)
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Continuous Integration

The GitHub Actions workflow runs these checks on every PR. Fix any issues before merging.
