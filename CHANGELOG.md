# Changelog

All notable changes to Astrabot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project restructuring with organized directory structure
- Modern Python packaging with pyproject.toml
- Diátaxis documentation framework
- Pre-commit hooks for code quality
- GitHub Actions CI/CD workflows
- Structured output support for tweet and image processing
- Batch image processing with conversation context tracking
- Enhanced conversational training data creation system
- Natural dialogue flow capture instead of forced Q&A format
- Conversation role modeling and style preservation

### Changed
- Migrated from Q&A extraction to conversational training approach
- Moved all modules to proper src/ directory structure
- Updated imports to reflect new module locations
- Enhanced .gitignore for better privacy protection
- Improved Twitter/X content extraction with structured outputs

### Fixed
- Broken image description example in notebook
- Incorrect role assignment in transform_to_conversations()
- Import paths after module reorganization

## [0.1.0] - 2024-01-01

### Added
- Initial project setup
- Signal backup processing with Docker
- Basic conversation extraction
- Twitter/X link content extraction
- Image description using vision APIs
- Unsloth integration for model fine-tuning
- Test-driven development framework