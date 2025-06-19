# Astrabot Architecture

This document explains the architectural design and key decisions behind Astrabot.

## Overview

Astrabot is designed as a modular system for creating personalized AI models from Signal messenger conversation history. The architecture emphasizes privacy, extensibility, and maintainability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│                  (Jupyter Notebooks)                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                    Core Pipeline                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Extract   │→ │   Process    │→ │     Train     │  │
│  │   Signal    │  │ Conversations│  │     Model     │  │
│  │    Data     │  │              │  │               │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                    Data Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Signal    │  │   Processed  │  │   Training    │  │
│  │  Flatfiles  │  │     Data     │  │   Datasets    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Extraction Layer (`docker/signalbackup-tools/`)

**Purpose**: Safely extract conversation data from encrypted Signal backups.

**Key Features**:
- Dockerized for isolation and security
- Converts binary backup format to CSV files
- Preserves message metadata and relationships

**Design Decision**: Using Docker ensures the extraction process is isolated from the main system, protecting both the backup data and the host system.

### 2. Core Processing (`src/core/`)

**Purpose**: Transform raw Signal data into structured training data.

**Components**:
- `conversation_processor.py`: Main processing logic
  - Conversation window extraction
  - Natural dialogue segmentation
  - Style preservation
  - Role modeling

**Design Patterns**:
- **Pipeline Pattern**: Data flows through transformation stages
- **Strategy Pattern**: Different processing strategies for different training modes

### 3. Data Models (`src/models/`)

**Purpose**: Define structured data schemas for consistency and validation.

**Components**:
- `conversation_schemas.py`: Pydantic models for structured data
  - `TweetContent`: Structured tweet data
  - `ImageDescription`: Structured image descriptions
  - `EnhancedMessage`: Complete message with extracted content

**Design Decision**: Using Pydantic provides automatic validation, serialization, and documentation of data structures.

### 4. Extractors (`src/extractors/`)

**Purpose**: Extract and enhance content from external sources.

**Components**:
- `twitter_extractor.py`: Twitter/X content extraction
  - Tweet text extraction
  - Image URL extraction
  - Batch processing support

**Design Pattern**: **Adapter Pattern** - Adapts various external data sources to internal formats.

### 5. LLM Integration (`src/llm/`)

**Purpose**: Handle model training and fine-tuning.

**Components**:
- `training_data_creator.py`: Creates formatted training data
- `prompts/`: Prompt templates for different models
- `configs/`: Model and training configurations

**Design Decision**: Separating LLM-specific code allows easy swapping of base models and training frameworks.

### 6. Utilities (`src/utils/`)

**Purpose**: Shared functionality across the system.

**Components**:
- `config.py`: Centralized configuration management
- `logging.py`: Structured logging with privacy features

**Design Patterns**:
- **Singleton Pattern**: Logger instance
- **Factory Pattern**: Configuration loading

## Data Flow Architecture

### 1. Extraction Phase
```
Signal Backup → Docker Container → CSV Files → Raw Data Storage
```

### 2. Processing Phase
```
Raw CSVs → Message Filtering → Conversation Segmentation → Style Analysis
```

### 3. Enhancement Phase
```
Messages → URL Detection → Content Extraction → Structured Data → Enhanced Messages
```

### 4. Training Data Creation
```
Enhanced Messages → Training Strategy → Formatted Examples → Dataset
```

### 5. Model Training
```
Dataset → Tokenization → Model Fine-tuning → Checkpoint Saving → Export
```

## Privacy Architecture

### Data Protection Layers

1. **Extraction Isolation**: Docker container for backup processing
2. **Sensitive Data Masking**: Automatic masking in logs
3. **Local Processing**: No cloud dependencies for core functionality
4. **Configurable Privacy**: Blocked contact handling

### Security Considerations

- API keys stored in environment variables
- No sensitive data in version control
- Comprehensive .gitignore patterns
- Optional encryption for exported data

## Scalability Considerations

### Horizontal Scaling

- Batch processing for API calls
- Parallel message processing
- Chunked dataset creation

### Vertical Scaling

- Configurable model sizes
- Gradient accumulation for large batches
- Memory-efficient LoRA training

## Extension Points

### 1. New Data Sources
Add extractors in `src/extractors/` following the existing pattern.

### 2. New Training Strategies
Extend `TrainingDataCreator` with new methods.

### 3. New Model Types
Add configurations in `src/llm/configs/`.

### 4. Custom Processing
Add processors in `src/core/` following the pipeline pattern.

## Technology Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Model library
- **Unsloth**: Efficient fine-tuning

### Data Processing
- **Pandas**: Data manipulation
- **Pydantic**: Data validation
- **Beautiful Soup**: Web scraping

### Development Tools
- **pytest**: Testing framework
- **Black**: Code formatting
- **Docker**: Containerization
- **pre-commit**: Git hooks

## Design Principles

1. **Modularity**: Each component has a single responsibility
2. **Privacy First**: User data protection at every layer
3. **Extensibility**: Easy to add new features
4. **Testability**: Comprehensive test coverage
5. **Documentation**: Clear documentation at all levels

## Future Architecture Considerations

### Planned Enhancements

1. **Plugin System**: Dynamic loading of extractors
2. **Distributed Training**: Multi-GPU support
3. **Model Registry**: Version management for trained models
4. **Web Interface**: Browser-based UI option

### Potential Optimizations

1. **Caching Layer**: Reduce API calls
2. **Stream Processing**: Handle larger datasets
3. **Async Processing**: Improve extraction speed
4. **Model Quantization**: Deployment optimization