#!/usr/bin/env python3
"""
Qwen3 Training Script for Astrabot

This script implements advanced fine-tuning for Qwen3 models using personal
conversation data, incorporating reasoning capabilities and style adaptation.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

from src.llm.training_data_creator import TrainingDataCreator
from src.llm.adaptive_trainer import AdaptiveTrainer
from src.core.style_analyzer import analyze_all_communication_styles
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Qwen3Trainer:
    """Handles Qwen3 model training with advanced features."""
    
    def __init__(self, config_path: str = None):
        """Initialize the Qwen3 trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self.training_data = []
        self.style_analysis = {}
        
    def _load_config(self, config_path: str = None) -> Dict:
        """Load training configuration from YAML file."""
        if config_path is None:
            config_path = project_root / "configs" / "training_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def load_model_and_tokenizer(self):
        """Load Qwen3 model and tokenizer with Unsloth optimizations."""
        logger.info(f"Loading model: {self.config['model']['name']}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config['model']['name'],
            max_seq_length=self.config['model']['max_seq_length'],
            dtype=self.config['model']['dtype'],
            load_in_4bit=self.config['model']['load_in_4bit'],
            load_in_8bit=self.config['model']['load_in_8bit'],
        )
        
        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config['lora']['r'],
            target_modules=self.config['lora']['target_modules'],
            lora_alpha=self.config['lora']['alpha'],
            lora_dropout=self.config['lora']['dropout'],
            bias=self.config['lora']['bias'],
            use_gradient_checkpointing=self.config['lora']['use_gradient_checkpointing'],
            random_state=self.config['lora']['random_state'],
            use_rslora=self.config['lora']['use_rslora'],
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
    def load_and_prepare_data(
        self, 
        messages_path: str, 
        recipients_path: str,
        your_recipient_id: int = 2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare Signal conversation data."""
        logger.info("Loading conversation data...")
        
        messages_df = pd.read_csv(messages_path)
        recipients_df = pd.read_csv(recipients_path)
        
        # Filter and clean messages
        messages_df = messages_df[
            (messages_df['body'].notna()) & 
            (messages_df['body'].str.len() >= self.config['dataset']['min_message_length'])
        ]
        
        logger.info(f"Loaded {len(messages_df)} messages from {len(recipients_df)} recipients")
        return messages_df, recipients_df
    
    def create_mixed_dataset(
        self,
        messages_df: pd.DataFrame,
        recipients_df: pd.DataFrame,
        your_recipient_id: int = 2
    ) -> List[Dict]:
        """Create mixed training dataset with multiple formats."""
        logger.info("Creating mixed training dataset...")
        
        creator = TrainingDataCreator(your_recipient_id)
        adaptive_trainer = AdaptiveTrainer(your_recipient_id)
        all_examples = []
        
        # 1. Conversational data
        if self.config['dataset']['modes']['conversational']['enabled']:
            logger.info("Creating conversational training data...")
            conv_data = creator.create_conversational_training_data(
                messages_df, 
                recipients_df,
                context_window=self.config['dataset']['modes']['conversational']['context_window'],
                include_metadata=self.config['dataset']['modes']['conversational']['include_metadata']
            )
            weight = self.config['dataset']['modes']['conversational']['weight']
            num_examples = int(len(conv_data) * weight)
            all_examples.extend(conv_data[:num_examples])
            logger.info(f"Added {num_examples} conversational examples")
        
        # 2. Adaptive training data
        if self.config['dataset']['modes']['adaptive']['enabled']:
            logger.info("Creating adaptive training data...")
            # Analyze communication styles first
            communication_styles = analyze_all_communication_styles(
                messages_df, recipients_df, your_recipient_id
            )
            self.style_analysis = communication_styles
            
            adaptive_data = adaptive_trainer.create_adaptive_training_data(
                messages_df, recipients_df, communication_styles
            )
            weight = self.config['dataset']['modes']['adaptive']['weight']
            num_examples = int(len(adaptive_data) * weight)
            all_examples.extend(adaptive_data[:num_examples])
            logger.info(f"Added {num_examples} adaptive examples")
        
        # 3. Burst sequence data
        if self.config['dataset']['modes']['burst_sequence']['enabled']:
            logger.info("Creating burst sequence data...")
            burst_data = creator.create_burst_sequence_data(
                messages_df.to_dict('records'), 
                your_recipient_id
            )
            weight = self.config['dataset']['modes']['burst_sequence']['weight']
            num_examples = int(len(burst_data) * weight)
            all_examples.extend(burst_data[:num_examples])
            logger.info(f"Added {num_examples} burst sequence examples")
        
        # 4. Q&A data
        if self.config['dataset']['modes']['qa']['enabled']:
            logger.info("Creating Q&A training data...")
            qa_data = creator.create_qa_training_data(
                messages_df.to_dict('records'),
                your_recipient_id
            )
            weight = self.config['dataset']['modes']['qa']['weight']
            num_examples = int(len(qa_data) * weight)
            all_examples.extend(qa_data[:num_examples])
            logger.info(f"Added {num_examples} Q&A examples")
        
        # Shuffle if configured
        if self.config['dataset']['shuffle']:
            import random
            random.seed(self.config['dataset']['seed'])
            random.shuffle(all_examples)
        
        logger.info(f"Total training examples: {len(all_examples)}")
        return all_examples
    
    def apply_chat_template(self, examples: List[Dict]) -> List[str]:
        """Apply Qwen3 chat template to training examples."""
        logger.info("Applying Qwen3 chat template...")
        
        formatted_examples = []
        
        for example in examples:
            # Handle different example formats
            if 'messages' in example:
                # Already in chat format
                text = self.tokenizer.apply_chat_template(
                    example['messages'],
                    tokenize=False,
                    add_generation_prompt=False
                )
            elif 'instruction' in example and 'output' in example:
                # Convert to chat format
                messages = []
                
                # Add system message if instruction is complex
                if 'adapt' in example.get('instruction', '').lower():
                    messages.append({
                        'role': 'system',
                        'content': example['instruction']
                    })
                    if 'input' in example:
                        messages.append({
                            'role': 'user',
                            'content': example['input']
                        })
                else:
                    # Simple format
                    messages.append({
                        'role': 'user',
                        'content': example.get('input', example['instruction'])
                    })
                
                messages.append({
                    'role': 'assistant',
                    'content': example['output']
                })
                
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                # Skip invalid examples
                logger.warning(f"Skipping invalid example format: {example.keys()}")
                continue
            
            formatted_examples.append({
                'text': text,
                'metadata': example.get('metadata', {})
            })
        
        logger.info(f"Formatted {len(formatted_examples)} examples with chat template")
        return formatted_examples
    
    def create_reasoning_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Create synthetic reasoning examples for Qwen3."""
        logger.info("Creating reasoning dataset...")
        
        # This would ideally load from a reasoning dataset
        # For now, create some synthetic examples
        reasoning_examples = []
        
        templates = [
            {
                'messages': [
                    {'role': 'user', 'content': 'Solve this step by step: If x + 5 = 12, what is x?'},
                    {'role': 'assistant', 'content': '<think>\nI need to solve for x in the equation x + 5 = 12.\nTo isolate x, I\'ll subtract 5 from both sides:\nx + 5 - 5 = 12 - 5\nx = 7\n</think>\n\nTo solve for x in the equation x + 5 = 12, I need to isolate x by subtracting 5 from both sides:\n\nx + 5 - 5 = 12 - 5\nx = 7\n\nTherefore, x = 7.'}
                ]
            },
            {
                'messages': [
                    {'role': 'user', 'content': 'Explain why the sky appears blue.'},
                    {'role': 'assistant', 'content': '<think>\nThe sky appears blue due to a phenomenon called Rayleigh scattering. I should explain:\n1. White light from the sun contains all colors\n2. When it hits air molecules, shorter wavelengths scatter more\n3. Blue has a short wavelength, so it scatters most\n4. This scattered blue light reaches our eyes from all directions\n</think>\n\nThe sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters Earth\'s atmosphere, it collides with gas molecules. Blue light has a shorter wavelength than other colors, causing it to scatter more than red or yellow light. This scattered blue light reaches our eyes from all directions, making the entire sky appear blue during the day.'}
                ]
            }
        ]
        
        # Expand templates with variations
        for i in range(min(num_examples, len(templates))):
            reasoning_examples.append(templates[i % len(templates)])
        
        return reasoning_examples
    
    def prepare_final_dataset(self, training_examples: List[Dict]) -> Dataset:
        """Prepare final dataset with optional reasoning data."""
        all_data = training_examples
        
        # Add reasoning data if configured
        if self.config['reasoning']['enabled'] and self.config['reasoning']['ratio'] > 0:
            num_reasoning = int(len(training_examples) * self.config['reasoning']['ratio'])
            reasoning_data = self.create_reasoning_dataset(num_reasoning)
            
            # Apply chat template to reasoning data
            reasoning_formatted = self.apply_chat_template(reasoning_data)
            all_data.extend(reasoning_formatted)
            
            logger.info(f"Added {len(reasoning_formatted)} reasoning examples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(all_data)
        
        # Apply deduplication if configured
        if self.config['dataset']['deduplicate']:
            logger.info("Deduplicating dataset...")
            unique_texts = set()
            filtered_data = []
            
            for item in all_data:
                text = item.get('text', '')
                if text and text not in unique_texts:
                    unique_texts.add(text)
                    filtered_data.append(item)
            
            dataset = Dataset.from_list(filtered_data)
            logger.info(f"Deduplicated to {len(dataset)} examples")
        
        return dataset
    
    def train(self, dataset: Dataset, output_dir: str = "./output"):
        """Train the model using SFTTrainer."""
        logger.info("Starting training...")
        
        # Determine if we should use fp16 or bf16
        if torch.cuda.is_available():
            if is_bfloat16_supported():
                use_fp16 = False
                use_bf16 = True
            else:
                use_fp16 = True
                use_bf16 = False
        else:
            use_fp16 = False
            use_bf16 = False
        
        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            optim=self.config['training']['optim'],
            weight_decay=self.config['training']['weight_decay'],
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            dataloader_pin_memory=self.config['training']['dataloader_pin_memory'],
            group_by_length=self.config['training']['group_by_length'],
            report_to=self.config['logging']['report_to'],
            logging_dir=self.config['logging']['logging_dir'],
            seed=self.config['lora']['random_state'],
            dataset_text_field="text",
            max_seq_length=self.config['model']['max_seq_length'],
            packing=False,  # Qwen3 works better without packing
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        
        # Show memory stats
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            logger.info(f"{start_gpu_memory} GB of memory reserved.")
        
        # Train
        trainer_stats = trainer.train()
        
        # Show final stats
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
            
            logger.info(f"Training completed in {trainer_stats.metrics['train_runtime']:.2f} seconds")
            logger.info(f"Peak memory: {used_memory} GB ({used_percentage}% of max)")
            logger.info(f"Memory for training: {used_memory_for_lora} GB ({lora_percentage}% of max)")
        
        return trainer_stats
    
    def save_model(self, output_dir: str = "./output"):
        """Save the trained model and tokenizer."""
        logger.info(f"Saving model to {output_dir}")
        
        # Save LoRA adapters
        if self.config['output']['save_lora_only']:
            lora_dir = Path(output_dir) / "lora_model"
            self.model.save_pretrained(lora_dir)
            self.tokenizer.save_pretrained(lora_dir)
            logger.info(f"Saved LoRA adapters to {lora_dir}")
        
        # Save merged models if configured
        if self.config['output']['save_merged_16bit']:
            merged_dir = Path(output_dir) / "merged_16bit"
            self.model.save_pretrained_merged(merged_dir, self.tokenizer, save_method="merged_16bit")
            logger.info(f"Saved merged 16-bit model to {merged_dir}")
        
        if self.config['output']['save_merged_4bit']:
            merged_dir = Path(output_dir) / "merged_4bit"
            self.model.save_pretrained_merged(merged_dir, self.tokenizer, save_method="merged_4bit")
            logger.info(f"Saved merged 4-bit model to {merged_dir}")
        
        # Save to Hub if configured
        if self.config['output']['push_to_hub'] and self.config['output']['hub_model_id']:
            logger.info(f"Pushing to Hub: {self.config['output']['hub_model_id']}")
            self.model.push_to_hub(self.config['output']['hub_model_id'])
            self.tokenizer.push_to_hub(self.config['output']['hub_model_id'])
        
        # Save GGUF if configured
        if self.config['output']['gguf']['enabled']:
            for method in self.config['output']['gguf']['quantization_methods']:
                gguf_path = Path(output_dir) / f"gguf_{method}"
                logger.info(f"Saving GGUF {method} to {gguf_path}")
                self.model.save_pretrained_gguf(
                    gguf_path, 
                    self.tokenizer, 
                    quantization_method=method
                )
    
    def test_model(self, test_prompts: List[str] = None):
        """Test the trained model with sample prompts."""
        logger.info("Testing trained model...")
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        if test_prompts is None:
            test_prompts = [
                "Hello! How are you doing today?",
                "Can you explain quantum computing in simple terms?",
                "What's your favorite way to spend a weekend?"
            ]
        
        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            
            # Determine if this needs reasoning
            needs_reasoning = any(word in prompt.lower() for word in ['explain', 'solve', 'why', 'how'])
            
            messages = [{"role": "user", "content": prompt}]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            
            # Use appropriate generation settings
            if needs_reasoning and self.config['reasoning']['enabled']:
                generation_config = self.config['reasoning']
            else:
                generation_config = self.config['chat']
            
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            outputs = self.model.generate(
                inputs,
                streamer=streamer,
                max_new_tokens=512,
                temperature=generation_config['temperature'],
                top_p=generation_config['top_p'],
                top_k=generation_config['top_k'],
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            print("\n" + "-"*50)
    
    def run_multi_stage_training(
        self,
        messages_df: pd.DataFrame,
        recipients_df: pd.DataFrame,
        your_recipient_id: int = 2,
        output_dir: str = "./output"
    ):
        """Run multi-stage training if configured."""
        if 'stages' not in self.config or not self.config['stages']:
            logger.info("No multi-stage configuration found, running single-stage training")
            return
        
        logger.info("Starting multi-stage training...")
        
        adaptive_trainer = AdaptiveTrainer(your_recipient_id)
        training_data = self.create_mixed_dataset(messages_df, recipients_df, your_recipient_id)
        
        # Prepare stage-specific datasets
        stage_datasets = adaptive_trainer.multi_stage_training_preparation(training_data)
        
        for stage_config in self.config['stages']:
            stage_name = stage_config['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting stage: {stage_name} - {stage_config['description']}")
            logger.info(f"{'='*60}")
            
            # Get stage-specific data
            if stage_name in stage_datasets:
                stage_data = stage_datasets[stage_name]
            else:
                logger.warning(f"No data for stage {stage_name}, skipping...")
                continue
            
            # Apply chat template
            formatted_data = self.apply_chat_template(stage_data)
            dataset = Dataset.from_list(formatted_data)
            
            # Update learning rate for this stage
            original_lr = self.config['training']['learning_rate']
            self.config['training']['learning_rate'] = stage_config['learning_rate']
            self.config['training']['num_train_epochs'] = stage_config['epochs']
            
            # Train this stage
            stage_output = Path(output_dir) / f"stage_{stage_name}"
            self.train(dataset, str(stage_output))
            
            # Restore original config
            self.config['training']['learning_rate'] = original_lr
            
            logger.info(f"Completed stage: {stage_name}")
        
        logger.info("Multi-stage training completed!")


def main():
    """Main training pipeline for Qwen3."""
    parser = argparse.ArgumentParser(description="Train Qwen3 model on personal conversations")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--messages",
        type=str,
        default="data/raw/signal-flatfiles/signal.csv",
        help="Path to messages CSV file"
    )
    
    parser.add_argument(
        "--recipients", 
        type=str,
        default="data/raw/signal-flatfiles/recipient.csv",
        help="Path to recipients CSV file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--your-id",
        type=int,
        default=2,
        help="Your recipient ID in Signal database"
    )
    
    parser.add_argument(
        "--multi-stage",
        action="store_true",
        help="Run multi-stage training"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test model after training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = Qwen3Trainer(args.config)
    
    # Load model
    trainer.load_model_and_tokenizer()
    
    # Load data
    messages_df, recipients_df = trainer.load_and_prepare_data(
        args.messages, 
        args.recipients,
        args.your_id
    )
    
    if args.multi_stage:
        # Run multi-stage training
        trainer.run_multi_stage_training(
            messages_df,
            recipients_df,
            args.your_id,
            args.output
        )
    else:
        # Create training dataset
        training_examples = trainer.create_mixed_dataset(
            messages_df,
            recipients_df,
            args.your_id
        )
        
        # Apply chat template
        formatted_examples = trainer.apply_chat_template(training_examples)
        
        # Prepare final dataset
        dataset = trainer.prepare_final_dataset(formatted_examples)
        
        # Train
        trainer.train(dataset, args.output)
    
    # Save model
    trainer.save_model(args.output)
    
    # Test if requested
    if args.test:
        trainer.test_model()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()