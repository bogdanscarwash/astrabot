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
- **New Makefile with comprehensive development commands**
- **Interactive test commands (test-file, test-one)**
- **Code quality automation (format, lint, type-check, all)**
- **Docker and data processing commands**
- **Training pipeline automation**
- **Development environment setup automation**
- **Documentation generation commands**
- **Pre-commit hook integration**
- **Comprehensive reference documentation for Signal data schema**
- **Detailed utilities reference documentation (config.py, logging.py)**
- **Enhanced getting started tutorial with step-by-step guidance**
- **Documentation index following Diátaxis framework**
- **In-depth Signal CSV schema analysis with privacy levels**
- **Data relationship diagrams and usage examples**

### Changed
- Migrated from Q&A extraction to conversational training approach
- Moved all modules to proper src/ directory structure
- Updated imports to reflect new module locations
- Enhanced .gitignore for better privacy protection
- Improved Twitter/X content extraction with structured outputs
- **Updated project overview in CLAUDE.md with current architecture**
- **Enhanced testing commands and code quality workflow**
- **Improved Signal backup processing and training pipeline documentation**
- **Updated environment setup and development workflow**
- **Refined project structure and data processing flow documentation**
- **Restructured getting started tutorial with troubleshooting section**
- **Added model selection guide and training monitoring tips**
- **Improved documentation navigation and cross-references**

### Fixed
- Broken image description example in notebook
- Incorrect role assignment in transform_to_conversations()
- Import paths after module reorganization
- **Updated author information in pyproject.toml**
- **Enhanced pytest configuration for better test discovery**
- **Improved setup.cfg with comprehensive tool configurations**

## [0.1.0] - 2024-01-01

### Added
- Initial project setup
- Signal backup processing with Docker
- Basic conversation extraction
- Twitter/X link content extraction
- Image description using vision APIs
- Unsloth integration for model fine-tuning
- Test-driven development framework