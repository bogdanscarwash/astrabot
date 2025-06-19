"""
Unit tests for adaptive trainer module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import torch

from src.llm.adaptive_trainer import AdaptiveTrainer


@pytest.mark.unit
@pytest.mark.llm
class TestAdaptiveTrainer:
    """Test adaptive training functionality"""
    
    @pytest.fixture
    def mock_messages_df(self):
        """Create mock messages DataFrame"""
        import pandas as pd
        from datetime import datetime
        return pd.DataFrame([
            {
                'thread_id': 1,
                'from_recipient_id': 2,
                'to_recipient_id': 3,
                'body': 'Test message',
                'date_sent': int(datetime.now().timestamp() * 1000)
            }
        ])
    
    @pytest.fixture
    def trainer(self):
        """Create AdaptiveTrainer instance"""
        return AdaptiveTrainer(your_recipient_id=2)
    
    @pytest.fixture
    def sample_training_config(self):
        """Create sample training configuration"""
        return {
            'model_name': 'Qwen/Qwen2.5-3B',
            'max_seq_length': 2048,
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'learning_rate': 2e-5,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 1,
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 500,
            'evaluation_strategy': 'steps',
            'eval_steps': 100
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        return [
            {
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'Hi there! How are you?'}
                ],
                'metadata': {
                    'conversation_id': 'conv_1',
                    'partner': 'Alice',
                    'style': 'casual'
                }
            },
            {
                'messages': [
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': 'What is Python?'},
                    {'role': 'assistant', 'content': 'Python is a programming language.'}
                ],
                'metadata': {
                    'conversation_id': 'conv_2', 
                    'partner': 'Bob',
                    'style': 'formal'
                }
            }
        ]
    
    def test_prepare_training_config(self, trainer, sample_training_config):
        """Test training configuration preparation"""
        config = trainer.prepare_training_config(sample_training_config)
        
        assert 'model_name' in config
        assert 'max_seq_length' in config
        assert 'lora_config' in config
        assert 'training_args' in config
        
        # LoRA config should be properly structured
        lora_config = config['lora_config']
        assert 'r' in lora_config
        assert 'alpha' in lora_config
        assert 'target_modules' in lora_config
    
    def test_create_conversation_partner_adaptations(self, trainer, sample_training_data):
        """Test creating partner-specific adaptations"""
        adaptations = trainer.create_conversation_partner_adaptations(sample_training_data)
        
        assert len(adaptations) > 0
        
        # Should have adaptations for different partners
        partners = set()
        for adaptation in adaptations:
            if 'partner' in adaptation.get('metadata', {}):
                partners.add(adaptation['metadata']['partner'])
        
        assert len(partners) >= 2  # Alice and Bob
    
    def test_style_adaptive_training_data(self, trainer, sample_training_data):
        """Test style-adaptive training data creation"""
        style_data = trainer.create_style_adaptive_data(sample_training_data)
        
        assert len(style_data) > 0
        
        # Should include style context in system messages
        for example in style_data:
            system_msg = example['messages'][0]
            assert system_msg['role'] == 'system'
            # Should mention style adaptation
            assert 'style' in system_msg['content'].lower() or 'adapt' in system_msg['content'].lower()
    
    def test_conversation_context_weighting(self, trainer, sample_training_data):
        """Test conversation context weighting"""
        weighted_data = trainer.apply_conversation_context_weighting(sample_training_data)
        
        assert len(weighted_data) == len(sample_training_data)
        
        # Should add weight information
        for example in weighted_data:
            assert 'weight' in example or 'sample_weight' in example
    
    @patch('src.llm.adaptive_trainer.AutoModelForCausalLM')
    @patch('src.llm.adaptive_trainer.AutoTokenizer')
    def test_load_model_and_tokenizer(self, mock_tokenizer, mock_model, trainer):
        """Test model and tokenizer loading"""
        mock_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        model, tokenizer = trainer.load_model_and_tokenizer(
            model_name="unsloth/Qwen2.5-3B",
            load_in_4bit=True
        )
        
        assert model is not None
        assert tokenizer is not None
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
    
    def test_tokenize_training_data(self, trainer, sample_training_data):
        """Test training data tokenization"""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "tokenized text"
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.eos_token_id = 2
        
        tokenized_data = trainer.tokenize_training_data(
            sample_training_data,
            mock_tokenizer,
            max_length=512
        )
        
        assert len(tokenized_data) == len(sample_training_data)
        assert all('input_ids' in example for example in tokenized_data)
        assert all('attention_mask' in example for example in tokenized_data)
    
    def test_calculate_adaptive_loss_weights(self, trainer, sample_training_data):
        """Test adaptive loss weight calculation"""
        weights = trainer.calculate_adaptive_loss_weights(sample_training_data)
        
        assert len(weights) == len(sample_training_data)
        assert all(isinstance(w, (int, float)) and w > 0 for w in weights)
        
        # Weights should vary based on conversation characteristics
        assert not all(w == weights[0] for w in weights)
    
    def test_create_partner_specific_datasets(self, trainer, sample_training_data):
        """Test creating partner-specific datasets"""
        partner_datasets = trainer.create_partner_specific_datasets(sample_training_data)
        
        assert isinstance(partner_datasets, dict)
        assert len(partner_datasets) > 0
        
        # Should have datasets for different partners
        assert 'Alice' in partner_datasets or 'Bob' in partner_datasets
        
        for partner, dataset in partner_datasets.items():
            assert len(dataset) > 0
            # All examples should be for the same partner
            for example in dataset:
                metadata = example.get('metadata', {})
                if 'partner' in metadata:
                    assert metadata['partner'] == partner
    
    def test_temporal_adaptation_weights(self, trainer, sample_training_data):
        """Test temporal adaptation weighting"""
        # Add timestamps to training data
        import datetime
        for i, example in enumerate(sample_training_data):
            example['metadata']['timestamp'] = datetime.datetime.now() - datetime.timedelta(days=i*30)
        
        temporal_weights = trainer.calculate_temporal_weights(sample_training_data)
        
        assert len(temporal_weights) == len(sample_training_data)
        
        # More recent conversations should have higher weights
        assert temporal_weights[0] >= temporal_weights[1]
    
    @patch('src.llm.adaptive_trainer.Trainer')
    def test_create_trainer_with_adaptations(self, mock_trainer_class, trainer, sample_training_data):
        """Test creating trainer with adaptive features"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_train_dataset = MagicMock()
        
        # Mock trainer instance
        mock_trainer_instance = MagicMock()
        mock_trainer_class.return_value = mock_trainer_instance
        
        adaptive_trainer = trainer.create_adaptive_trainer(
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataset=mock_train_dataset,
            training_args={'output_dir': '/tmp/test'}
        )
        
        assert adaptive_trainer is not None
        mock_trainer_class.assert_called_once()
    
    def test_conversation_style_detection(self, trainer, sample_training_data):
        """Test conversation style detection"""
        styles = trainer.detect_conversation_styles(sample_training_data)
        
        assert isinstance(styles, dict)
        assert len(styles) > 0
        
        # Should detect different styles
        detected_styles = set(styles.values())
        assert len(detected_styles) > 1  # Should detect multiple styles
    
    def test_adaptive_learning_rate_scheduling(self, trainer):
        """Test adaptive learning rate scheduling"""
        # Mock optimizer and scheduler
        mock_optimizer = MagicMock()
        mock_scheduler = trainer.create_adaptive_lr_scheduler(
            optimizer=mock_optimizer,
            adaptation_strategy='conversation_aware'
        )
        
        assert mock_scheduler is not None
    
    def test_conversation_quality_filtering(self, trainer, sample_training_data):
        """Test conversation quality filtering"""
        # Add quality scores to training data
        for example in sample_training_data:
            example['metadata']['quality_score'] = 0.8 if 'Alice' in str(example) else 0.3
        
        filtered_data = trainer.filter_by_conversation_quality(
            sample_training_data,
            min_quality_score=0.5
        )
        
        # Should filter out low quality conversations
        assert len(filtered_data) < len(sample_training_data)
        
        for example in filtered_data:
            quality = example['metadata'].get('quality_score', 1.0)
            assert quality >= 0.5
    
    def test_multi_stage_training_preparation(self, trainer, sample_training_data):
        """Test multi-stage training preparation"""
        stages = trainer.prepare_multi_stage_training(sample_training_data)
        
        assert isinstance(stages, list)
        assert len(stages) > 0
        
        # Each stage should have training data and config
        for stage in stages:
            assert 'data' in stage
            assert 'config' in stage
            assert len(stage['data']) > 0
    
    def test_empty_data_handling(self, trainer):
        """Test handling of empty training data"""
        empty_data = []
        
        # Should handle empty data gracefully
        adaptations = trainer.create_conversation_partner_adaptations(empty_data)
        assert len(adaptations) == 0
        
        weights = trainer.calculate_adaptive_loss_weights(empty_data)
        assert len(weights) == 0
    
    def test_memory_efficient_training_preparation(self, trainer, sample_training_data):
        """Test memory-efficient training data preparation"""
        # Create larger dataset
        large_dataset = sample_training_data * 100
        
        efficient_data = trainer.prepare_memory_efficient_training(
            large_dataset,
            batch_size=8,
            max_memory_gb=2.0
        )
        
        # Should handle large datasets efficiently
        assert len(efficient_data) <= len(large_dataset)
        assert isinstance(efficient_data, (list, torch.utils.data.Dataset))