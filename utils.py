"""
Utility functions for LLM fine-tuning project.
Includes evaluation metrics, visualization, and model serving.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluation metrics for language models."""
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """
        Calculate perplexity from loss.
        Perplexity = exp(loss)
        Lower is better.
        """
        return np.exp(loss)
    
    @staticmethod
    def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
        """
        Calculate BLEU score (simplified version).
        Requires: pip install nltk
        """
        try:
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.tokenize import word_tokenize
            
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            
            weights = [1/max_n] * max_n
            score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights)
            return score
        except ImportError:
            logger.warning("NLTK not installed. Install with: pip install nltk")
            return 0.0
    
    @staticmethod
    def rouge_score(reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores.
        Requires: pip install rouge-score
        """
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, hypothesis)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except ImportError:
            logger.warning("rouge-score not installed. Install with: pip install rouge-score")
            return {}


class TrainingVisualizer:
    """Visualization utilities for training metrics."""
    
    @staticmethod
    def plot_training_history(history_file: str, output_file: str = None):
        """
        Plot training and validation loss.
        
        Args:
            history_file: Path to training_history.json
            output_file: Optional path to save plot
        """
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs = history['epoch']
        train_loss = history['train_loss']
        val_loss = history.get('val_loss', [])
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, marker='o', label='Train Loss')
        if val_loss:
            plt.plot(epochs, val_loss, marker='s', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity plot
        plt.subplot(1, 2, 2)
        train_ppl = [np.exp(loss) for loss in train_loss]
        plt.plot(epochs, train_ppl, marker='o', label='Train Perplexity')
        if val_loss:
            val_ppl = [np.exp(loss) for loss in val_loss]
            plt.plot(epochs, val_ppl, marker='s', label='Val Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Model Perplexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_learning_curve(history_file: str, output_file: str = None):
        """Plot learning rate schedule effect."""
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs = history['epoch']
        train_loss = history['train_loss']
        
        # Calculate gradient (rate of change)
        gradients = np.diff(train_loss)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, train_loss, marker='o', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Over Time')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(range(len(gradients)), gradients, alpha=0.7)
        ax2.set_xlabel('Epoch Transition')
        ax2.set_ylabel('Loss Gradient')
        ax2.set_title('Rate of Loss Change')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {output_file}")
        else:
            plt.show()
        
        plt.close()


class DataAnalyzer:
    """Analyze training data properties."""
    
    @staticmethod
    def analyze_text_file(file_path: str) -> Dict:
        """
        Analyze text file statistics.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with statistics
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        lines = text.split('\n')
        words = text.split()
        sentences = text.split('.')
        
        stats = {
            'total_characters': len(text),
            'total_words': len(words),
            'total_lines': len(lines),
            'total_sentences': len(sentences),
            'avg_line_length': len(text) / len(lines) if lines else 0,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words': len(set(w.lower() for w in words)),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0,
        }
        
        logger.info("Text File Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return stats
    
    @staticmethod
    def analyze_tokenized_data(tokenizer, text: str) -> Dict:
        """
        Analyze tokenized data properties.
        
        Args:
            tokenizer: Hugging Face tokenizer
            text: Input text
            
        Returns:
            Dictionary with tokenization statistics
        """
        tokens = tokenizer.encode(text)
        token_ids = torch.tensor(tokens)
        
        stats = {
            'total_tokens': len(tokens),
            'vocab_size': tokenizer.vocab_size,
            'unique_tokens': len(set(tokens)),
            'avg_tokens_per_word': len(tokens) / len(text.split()),
            'token_distribution': {
                'min': int(token_ids.min().item()),
                'max': int(token_ids.max().item()),
                'mean': float(token_ids.float().mean().item()),
                'std': float(token_ids.float().std().item()),
            }
        }
        
        return stats


class TextGenerationDemo:
    """Interactive text generation utilities."""
    
    @staticmethod
    def interactive_generation(model, tokenizer, device: str = 'cpu'):
        """
        Interactive mode for text generation.
        
        Args:
            model: Fine-tuned language model
            tokenizer: Model tokenizer
            device: 'cuda' or 'cpu'
        """
        print("\n" + "="*60)
        print("Interactive Text Generation")
        print("="*60)
        print("Enter prompts to generate text. Type 'quit' to exit.\n")
        
        model.eval()
        
        while True:
            prompt = input(">> Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                print("Please enter a prompt.\n")
                continue
            
            try:
                input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_length=100,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"\n<< Generated: {generated_text}\n")
                
            except Exception as e:
                print(f"Error: {e}\n")
    
    @staticmethod
    def batch_generation(
        model,
        tokenizer,
        prompts: List[str],
        device: str = 'cpu',
        max_length: int = 100
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            model: Fine-tuned language model
            tokenizer: Model tokenizer
            prompts: List of input prompts
            device: 'cuda' or 'cpu'
            max_length: Maximum generation length
            
        Returns:
            List of generated texts
        """
        model.eval()
        results = []
        
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append(generated_text)
        
        return results


class ModelDeployer:
    """Utilities for model deployment."""
    
    @staticmethod
    def create_inference_script(model_path: str, output_file: str = "inference.py"):
        """
        Create a standalone inference script.
        
        Args:
            model_path: Path to fine-tuned model
            output_file: Output script filename
        """
        script_content = f'''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMInference:
    def __init__(self, model_path="{model_path}"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.7):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage
    llm = LLMInference()
    
    prompt = "Artificial intelligence is"
    result = llm.generate(prompt)
    print(f"Prompt: {{prompt}}")
    print(f"Generated: {{result}}")
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Inference script created: {output_file}")
    
    @staticmethod
    def create_flask_app(model_path: str, output_file: str = "app.py"):
        """
        Create a Flask web app for model serving.
        
        Args:
            model_path: Path to fine-tuned model
            output_file: Output app filename
        """
        app_content = f'''
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model on startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("{model_path}")
model = AutoModelForCausalLM.from_pretrained("{model_path}").to(DEVICE)
model.eval()


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate text from prompt.
    
    Request JSON:
    {{
        "prompt": "The future of AI",
        "max_length": 100,
        "temperature": 0.7
    }}
    """
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    
    if not prompt:
        return jsonify({{"error": "Prompt required"}}), 400
    
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({{
            "prompt": prompt,
            "generated": generated_text,
            "device": DEVICE
        }})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({{"status": "healthy", "device": DEVICE}})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
'''
        
        with open(output_file, 'w') as f:
            f.write(app_content)
        
        logger.info(f"Flask app created: {output_file}")
        logger.info("Run with: flask run")


# Example usage functions
def example_evaluation():
    """Example: Evaluate generated text."""
    evaluator = ModelEvaluator()
    
    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "The fast brown fox jumps over a lazy dog"
    
    bleu = evaluator.bleu_score(reference, hypothesis)
    rouge = evaluator.rouge_score(reference, hypothesis)
    
    print(f"BLEU Score: {bleu:.4f}")
    print(f"ROUGE Scores: {rouge}")


def example_visualization():
    """Example: Visualize training history."""
    visualizer = TrainingVisualizer()
    
    # Assuming training_history.json exists
    visualizer.plot_training_history(
        'models/final_model/training_history.json',
        output_file='training_plot.png'
    )


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("\nAvailable classes:")
    print("- ModelEvaluator: Evaluation metrics")
    print("- TrainingVisualizer: Visualization tools")
    print("- DataAnalyzer: Data statistics")
    print("- TextGenerationDemo: Interactive generation")
    print("- ModelDeployer: Deployment utilities")
