"""
LLM Fine-tuning with PyTorch - Mini Project
=============================================
A complete pipeline for fine-tuning a transformer-based language model
on custom text data using PyTorch and Hugging Face Transformers.

Project Structure:
- Data loading and preprocessing
- Custom PyTorch dataset
- Model initialization (GPT-2 or DistilGPT-2)
- Training loop with validation
- Inference and evaluation
- Checkpointing and model saving
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom PyTorch Dataset for language modeling."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_length: int = 512,
        stride: int = 256
    ):
        """
        Args:
            file_path: Path to text file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            stride: Stride for creating overlapping samples
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Loaded text: {len(text)} characters")
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        logger.info(f"Tokenized: {len(self.tokens)} tokens")
        
        # Create samples with stride
        self.samples = []
        for i in range(0, len(self.tokens) - max_length, stride):
            sample = self.tokens[i:i + max_length]
            if len(sample) == max_length:
                self.samples.append(sample)
        
        logger.info(f"Created {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        input_ids = torch.tensor(sample[:-1], dtype=torch.long)
        labels = torch.tensor(sample[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class LLMTrainer:
    """Main trainer class for fine-tuning language models."""
    
    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = None,
        output_dir: str = "./models",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 3,
        max_length: int = 512
    ):
        """
        Initialize trainer with model and hyperparameters.
        
        Args:
            model_name: Hugging Face model identifier
            device: 'cuda' or 'cpu'
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            num_epochs: Number of training epochs
            max_length: Max sequence length
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model loaded: {model_name}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                labels=labels
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        progress_bar = tqdm(val_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                labels=labels
            )
            loss = outputs.loss
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset = None,
        checkpoint_steps: int = None
    ):
        """
        Full training loop with validation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            checkpoint_steps: Save checkpoint every N steps
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['epoch'].append(epoch + 1)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Validation
            if val_dataset:
                val_loss = self.evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                logger.info(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch)
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_model(self, model_name: str = "final_model"):
        """Save final fine-tuned model."""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save training history
        history_path = model_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Model saved to {model_dir}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text
            max_length: Maximum generated length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated texts
        """
        self.model.eval()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts


def main():
    """Main execution function with example usage."""
    
    # Configuration
    CONFIG = {
        'model_name': 'gpt2',  # or 'distilgpt2' for smaller model
        'data_file': 'my_data.txt',
        'batch_size': 4,
        'num_epochs': 2,
        'learning_rate': 5e-5,
        'max_length': 256,
        'output_dir': './models',
        'val_split': 0.1  # 10% for validation
    }
    
    # Check if data file exists
    if not Path(CONFIG['data_file']).exists():
        logger.warning(f"Data file '{CONFIG['data_file']}' not found!")
        logger.info("Creating sample data file...")
        create_sample_data(CONFIG['data_file'])
    
    # Initialize trainer
    trainer = LLMTrainer(
        model_name=CONFIG['model_name'],
        batch_size=CONFIG['batch_size'],
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        max_length=CONFIG['max_length'],
        output_dir=CONFIG['output_dir']
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = TextDataset(
        file_path=CONFIG['data_file'],
        tokenizer=trainer.tokenizer,
        max_length=CONFIG['max_length']
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * CONFIG['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    # Train model
    trainer.train(train_dataset, val_dataset)
    
    # Save final model
    trainer.save_model()
    
    # Generate sample text
    logger.info("\n--- Sample Generation ---")
    prompts = [
        "The future of artificial intelligence",
        "Once upon a time",
        "Machine learning is"
    ]
    
    for prompt in prompts:
        logger.info(f"\nPrompt: {prompt}")
        generated = trainer.generate(prompt, max_length=50)
        for i, text in enumerate(generated, 1):
            logger.info(f"Generated {i}: {text}")
               

def create_sample_data(filename: str):
    """Create sample training data for demonstration."""
    sample_text = """
    Artificial Intelligence is transforming the world. Machine learning algorithms enable computers
    to learn from data without being explicitly programmed. Deep learning models can recognize patterns,
    generate text, and make predictions with remarkable accuracy.
    
    Transformers have revolutionized natural language processing. BERT, GPT, and other transformer models
    have achieved state-of-the-art results on numerous tasks. These models learn contextual representations
    of words and sentences through self-attention mechanisms.
    
    Fine-tuning is a powerful technique in transfer learning. By training a pre-trained model on task-specific
    data, we can achieve excellent results with limited computational resources. This approach has democratized
    access to powerful language models.
    
    PyTorch provides a flexible framework for implementing machine learning models. Its dynamic computation graphs
    allow for intuitive model design and debugging. The PyTorch community continues to grow with excellent documentation
    and tools.
    
    The future of AI involves larger models, more efficient training methods, and better alignment with human values.
    Research in areas like few-shot learning, prompt engineering, and retrieval-augmented generation continues to advance
    the capabilities of language models.
    """ * 5  # Repeat for more data
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    logger.info(f"Sample data created at {filename}")


if __name__ == "__main__":
    main()
