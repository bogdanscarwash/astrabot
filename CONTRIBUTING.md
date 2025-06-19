# Contributing to Astrabot

Thank you for your interest in contributing to Astrabot! This document provides guidelines and best practices for development.

## Development Setup

### Prerequisites

- Python 3.8+
- Docker (for Signal backup processing)
- CUDA-capable GPU (recommended for training)
- Git with pre-commit hooks

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/astrabot.git
cd astrabot
```

2. Run the development setup script:
```bash
./scripts/dev-setup.sh
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Development Workflow

### Code Style

- We use Black for Python formatting (line length: 100)
- Flake8 for linting
- isort for import sorting
- mypy for type checking

All of these are automatically run via pre-commit hooks.

### Test-Driven Development (TDD)

We follow strict TDD principles:

1. **Write tests first** - Create comprehensive unit tests before implementation
2. **Red-Green-Refactor** - Tests fail → Make them pass → Improve code
3. **Test structure**:
   ```python
   class TestFeature(unittest.TestCase):
       def setUp(self):
           # Test setup
           
       def test_basic_functionality(self):
           # Core feature tests
           
       def test_edge_cases(self):
           # Boundary condition tests
           
       def test_security_features(self):
           # Security-related tests
   ```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
make test-file

# Run with coverage
make test-coverage

# Run only unit tests
make test-unit
```

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Write tests for your feature
3. Implement the feature
4. Ensure all tests pass
5. Run linting and formatting:
   ```bash
   make lint
   make format
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code refactoring
- `style:` Code style changes
- `chore:` Maintenance tasks

## Project Structure

```
astrabot/
├── src/           # Source code
│   ├── core/      # Core functionality
│   ├── extractors/# Data extraction
│   ├── llm/       # LLM-related code
│   ├── models/    # Data models
│   └── utils/     # Utilities
├── tests/         # Test suite
├── docs/          # Documentation
├── notebooks/     # Jupyter notebooks
└── scripts/       # Development scripts
```

## Documentation

We use the Diátaxis framework:

- **Tutorials** (`docs/tutorials/`): Learning-oriented guides
- **How-to guides** (`docs/how-to/`): Task-oriented instructions
- **Reference** (`docs/reference/`): Technical documentation
- **Explanation** (`docs/explanation/`): Conceptual documentation

When adding new features, please update relevant documentation.

## Code Review Guidelines

### For Contributors

- Ensure all tests pass
- Add tests for new functionality
- Update documentation
- Follow code style guidelines
- Keep commits focused and atomic

### For Reviewers

- Check test coverage
- Verify documentation updates
- Ensure code follows project patterns
- Test the changes locally
- Provide constructive feedback

## Security

- Never commit sensitive data
- Use environment variables for secrets
- Follow the principle of least privilege
- Report security issues privately

## Performance Considerations

- Profile code for performance bottlenecks
- Use batch processing where appropriate
- Consider memory usage for large datasets
- Document performance characteristics

## Questions or Issues?

- Check existing issues first
- Use issue templates when creating new issues
- Join discussions in the issues section
- Be respectful and constructive

## License

By contributing to Astrabot, you agree that your contributions will be licensed under the MIT License.