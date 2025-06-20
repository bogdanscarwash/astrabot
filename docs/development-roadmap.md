# Astrabot Development Roadmap

This comprehensive roadmap guides developers through the complete process of setting up Astrabot for data processing, testing compliance, and model fine-tuning.

## Overview

Astrabot is a privacy-first AI language model system designed to process Signal messenger conversations while maintaining strict privacy standards. This roadmap outlines the three critical phases needed to achieve a fully functional system.

## Phase 1: Data Processing Pipeline

### 1.1 Signal Backup Extraction

**Objective**: Extract conversation data from Signal Desktop backups using Docker-based tools.

**Prerequisites**:
- Docker installed and running
- Signal Desktop backup files available
- Sufficient disk space for processing

**Steps**:

1. **Set up Signal extraction environment**:
   ```bash
   # Build Docker container for Signal processing
   docker build -t signal-extractor ./docker/signal-extractor
   
   # Mount backup directory and run extraction
   docker run -v /path/to/signal/backup:/data signal-extractor
   ```

2. **Extract conversation data**:
   - Use `signal-backup-decode` tool within Docker container
   - Export conversations to structured CSV format
   - Validate data integrity and completeness

3. **Data validation**:
   ```python
   from src.data.signal_processor import SignalProcessor
   
   processor = SignalProcessor()
   conversations = processor.load_backup('/path/to/extracted/data')
   processor.validate_data_integrity(conversations)
   ```

**Expected Output**: Clean CSV files containing conversation threads, messages, and metadata.

### 1.2 Conversation Processing

**Objective**: Transform raw Signal data into structured conversation windows for analysis.

**Implementation**:

1. **Initialize ConversationProcessor**:
   ```python
   from src.data.conversation_processor import ConversationProcessor
   
   processor = ConversationProcessor(
       window_size_minutes=30,
       min_messages_per_window=3
   )
   ```

2. **Create conversation windows**:
   ```python
   # Process CSV data into conversation windows
   windows = processor.create_conversation_windows(csv_data)
   
   # Apply relationship dynamic analysis
   analyzed_windows = processor.analyze_relationship_dynamics(windows)
   ```

3. **Data enrichment**:
   - Apply emoji analysis using `EmojiAnalyzer`
   - Detect message timing patterns
   - Classify conversation topics and emotional tones

**Expected Output**: Structured `ConversationWindow` objects with rich metadata.

### 1.3 Twitter Content Enhancement

**Objective**: Enhance conversation context with Twitter content using privacy-preserving methods.

**Implementation**:

1. **Set up TwitterExtractor**:
   ```python
   from src.data.twitter_extractor import TwitterExtractor
   
   extractor = TwitterExtractor(
       nitter_instances=['nitter.net', 'nitter.it'],
       rate_limit_delay=2.0
   )
   ```

2. **Extract Twitter context**:
   ```python
   # Find Twitter URLs in conversations
   twitter_urls = extractor.find_twitter_references(conversation_data)
   
   # Fetch content via Nitter (privacy-preserving)
   enhanced_data = extractor.enhance_conversations(
       conversation_data, 
       twitter_urls
   )
   ```

3. **Content integration**:
   - Merge Twitter content with conversation context
   - Maintain privacy by using Nitter instances
   - Handle rate limiting and error recovery

**Expected Output**: Conversations enriched with relevant Twitter context while preserving privacy.

### 1.4 Data Anonymization

**Objective**: Ensure all personal identifiers are removed or anonymized before training.

**Implementation**:

1. **Apply anonymization pipeline**:
   ```python
   from src.privacy.anonymizer import ConversationAnonymizer
   
   anonymizer = ConversationAnonymizer()
   anonymized_data = anonymizer.anonymize_conversations(processed_data)
   ```

2. **Validation checks**:
   ```python
   # Run privacy validation
   privacy_validator = PrivacyValidator()
   privacy_validator.check_for_sensitive_data(anonymized_data)
   ```

**Expected Output**: Fully anonymized conversation data ready for training.

## Phase 2: TDD Test Compliance

### 2.1 Fix Existing Test Issues

**Objective**: Resolve all failing tests to establish a solid testing foundation.

**Current Issues**:

1. **Fix undefined variable in adaptive trainer tests**:
   ```python
   # File: tests/unit/test_adaptive_trainer.py
   # Lines 217, 227: Fix undefined 'mock_trainer_class'
   
   @patch('src.training.adaptive_trainer.TrainerClass')
   def test_adaptive_trainer_initialization(self, mock_trainer_class):
       mock_trainer_instance = MagicMock()
       mock_trainer_class.return_value = mock_trainer_instance
       
       # Test implementation
       trainer = AdaptiveTrainer()
       trainer.initialize()
       
       mock_trainer_class.assert_called_once()
   ```

2. **Update deprecated GitHub Actions**:
   - Replace `actions/upload-artifact@v3` with `@v4`
   - Update workflow configurations in `.github/workflows/`

### 2.2 Comprehensive Test Suite Execution

**Objective**: Ensure all test categories pass consistently.

**Test Categories**:

1. **Unit Tests**:
   ```bash
   make test-unit
   # Or: pytest tests/unit/ -v
   ```

2. **Integration Tests**:
   ```bash
   make test-integration
   # Or: pytest tests/integration/ -v
   ```

3. **Privacy Tests**:
   ```bash
   make test-privacy
   # Or: pytest tests/privacy/ -v
   ```

4. **End-to-End Tests**:
   ```bash
   make test-e2e
   # Or: pytest tests/e2e/ -v
   ```

### 2.3 Test Coverage and Quality

**Objective**: Achieve comprehensive test coverage across all modules.

**Implementation**:

1. **Measure current coverage**:
   ```bash
   pytest --cov=src --cov-report=html tests/
   ```

2. **Target coverage goals**:
   - Core modules: 90%+ coverage
   - Data processing: 85%+ coverage
   - Privacy modules: 95%+ coverage

3. **Add missing tests**:
   - Test edge cases and error conditions
   - Add property-based testing for data validation
   - Implement performance benchmarks

### 2.4 Continuous Integration

**Objective**: Establish robust CI/CD pipeline for automated testing.

**Implementation**:

1. **GitHub Actions workflow**:
   ```yaml
   # .github/workflows/test.yml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: [3.8, 3.9, 3.10, 3.11]
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v4
         - run: make setup
         - run: make test
         - run: make test-privacy
   ```

2. **Quality gates**:
   - All tests must pass
   - Code coverage thresholds met
   - Linting and formatting checks pass
   - Privacy validation successful

## Phase 3: Model Fine-tuning Pipeline

### 3.1 Data Preparation

**Objective**: Prepare anonymized conversation data for model training.

**Implementation**:

1. **Load and filter data**:
   ```python
   from scripts.train_qwen3 import load_and_prepare_data
   
   # Load processed conversation data
   conversations_df = load_and_prepare_data(
       data_path='data/processed/conversations.csv',
       min_messages=5,
       max_window_hours=24
   )
   ```

2. **Data quality validation**:
   ```python
   # Validate data structure and content
   assert 'body' in conversations_df.columns
   assert 'from_recipient_id' in conversations_df.columns
   assert conversations_df['body'].notna().all()
   
   # Check for privacy compliance
   privacy_validator.validate_training_data(conversations_df)
   ```

3. **Data splitting**:
   ```python
   from sklearn.model_selection import train_test_split
   
   train_df, val_df = train_test_split(
       conversations_df, 
       test_size=0.2, 
       random_state=42
   )
   ```

### 3.2 Training Data Creation

**Objective**: Generate diverse training examples using multiple formats.

**Implementation**:

1. **Create mixed dataset**:
   ```python
   from src.llm.training_data_creator import TrainingDataCreator
   
   creator = TrainingDataCreator()
   
   # Generate different training formats
   conversational_data = creator.create_conversational_examples(train_df)
   adaptive_data = creator.create_adaptive_examples(train_df)
   burst_data = creator.create_burst_examples(train_df)
   
   # Combine datasets
   mixed_dataset = creator.combine_datasets([
       conversational_data,
       adaptive_data, 
       burst_data
   ])
   ```

2. **Data format validation**:
   ```python
   # Ensure proper format for Qwen3 training
   for example in mixed_dataset:
       assert 'messages' in example
       assert len(example['messages']) >= 2
       assert example['messages'][0]['role'] == 'user'
       assert example['messages'][-1]['role'] == 'assistant'
   ```

### 3.3 Qwen3 Model Fine-tuning

**Objective**: Fine-tune Qwen3 model using prepared conversation data.

**Prerequisites**:
- CUDA-compatible GPU with sufficient VRAM
- Transformers library with Qwen3 support
- Adequate disk space for model checkpoints

**Implementation**:

1. **Model setup**:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from peft import LoraConfig, get_peft_model
   
   # Load base model with 4-bit quantization
   model = AutoModelForCausalLM.from_pretrained(
       "Qwen/Qwen2.5-7B-Instruct",
       load_in_4bit=True,
       device_map="auto"
   )
   
   # Configure LoRA
   lora_config = LoraConfig(
       r=16,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.1
   )
   
   model = get_peft_model(model, lora_config)
   ```

2. **Training configuration**:
   ```python
   from transformers import TrainingArguments, Trainer
   
   training_args = TrainingArguments(
       output_dir="./checkpoints/qwen3-astrabot",
       num_train_epochs=3,
       per_device_train_batch_size=4,
       gradient_accumulation_steps=4,
       warmup_steps=100,
       learning_rate=2e-4,
       fp16=True,
       logging_steps=10,
       save_steps=500,
       eval_steps=500,
       evaluation_strategy="steps",
       save_total_limit=3,
       load_best_model_at_end=True,
   )
   ```

3. **Training execution**:
   ```python
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       tokenizer=tokenizer,
   )
   
   # Start training
   trainer.train()
   
   # Save final model
   trainer.save_model("./models/qwen3-astrabot-final")
   ```

### 3.4 Multi-stage Training

**Objective**: Implement progressive training for optimal results.

**Implementation**:

1. **Stage 1: Base conversation understanding**:
   ```python
   # Train on general conversation patterns
   stage1_data = filter_conversational_data(mixed_dataset)
   train_model_stage(model, stage1_data, epochs=2)
   ```

2. **Stage 2: Adaptive response training**:
   ```python
   # Fine-tune on adaptive conversation examples
   stage2_data = filter_adaptive_data(mixed_dataset)
   train_model_stage(model, stage2_data, epochs=1)
   ```

3. **Stage 3: Burst conversation handling**:
   ```python
   # Specialize in rapid conversation patterns
   stage3_data = filter_burst_data(mixed_dataset)
   train_model_stage(model, stage3_data, epochs=1)
   ```

### 3.5 Model Evaluation and Validation

**Objective**: Validate model performance and privacy compliance.

**Implementation**:

1. **Performance metrics**:
   ```python
   from src.evaluation.model_evaluator import ModelEvaluator
   
   evaluator = ModelEvaluator()
   metrics = evaluator.evaluate_model(
       model, 
       test_dataset,
       metrics=['perplexity', 'bleu', 'conversation_coherence']
   )
   ```

2. **Privacy validation**:
   ```python
   # Test model for privacy leakage
   privacy_tester = PrivacyTester()
   privacy_results = privacy_tester.test_model_privacy(model)
   
   assert privacy_results['pii_leakage_score'] < 0.01
   assert privacy_results['memorization_score'] < 0.05
   ```

3. **Conversation quality assessment**:
   ```python
   # Test conversation quality
   quality_assessor = ConversationQualityAssessor()
   quality_scores = quality_assessor.assess_conversations(
       model, 
       test_conversations
   )
   ```

## Success Criteria

### Phase 1 Completion
- [ ] Signal backups successfully extracted and processed
- [ ] Conversation windows created with proper metadata
- [ ] Twitter content integrated while preserving privacy
- [ ] All data anonymized and privacy-validated

### Phase 2 Completion
- [ ] All unit tests passing (100%)
- [ ] Integration tests passing (100%)
- [ ] Privacy tests passing (100%)
- [ ] Code coverage above target thresholds
- [ ] CI/CD pipeline operational

### Phase 3 Completion
- [ ] Training data prepared and validated
- [ ] Qwen3 model successfully fine-tuned
- [ ] Multi-stage training completed
- [ ] Model performance meets quality benchmarks
- [ ] Privacy compliance verified

## Troubleshooting

### Common Issues

1. **Memory issues during training**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use gradient checkpointing

2. **Privacy validation failures**:
   - Review anonymization pipeline
   - Check for data leakage in training examples
   - Validate PII removal

3. **Test failures**:
   - Check data dependencies
   - Verify environment setup
   - Review mock configurations

### Performance Optimization

1. **Data processing**:
   - Use multiprocessing for large datasets
   - Implement data streaming for memory efficiency
   - Cache processed results

2. **Model training**:
   - Use mixed precision training
   - Implement gradient accumulation
   - Optimize data loading pipeline

## Next Steps

After completing this roadmap:

1. **Model deployment**: Set up inference pipeline for production use
2. **Monitoring**: Implement model performance monitoring
3. **Continuous improvement**: Establish feedback loop for model updates
4. **Documentation**: Create user guides and API documentation
5. **Security audit**: Conduct comprehensive security review

## Resources

- [Qwen3 Fine-tuning Guide](docs/how-to/fine-tune-qwen3.md)
- [Privacy Guidelines](docs/privacy/guidelines.md)
- [Testing Best Practices](docs/reference/testing.md)
- [API Reference](docs/reference/api.md)

---

**Last Updated**: June 19, 2025  
**Version**: 1.0  
**Maintainer**: Astrabot Development Team
