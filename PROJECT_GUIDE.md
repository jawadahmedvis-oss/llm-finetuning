# LLM Fine-tuning with PyTorch - Mini Project Guide

## 📋 Project Overview

This mini project demonstrates **end-to-end fine-tuning of a pre-trained language model** using PyTorch and Hugging Face Transformers. You'll learn how to:

- Load and preprocess text data
- Create a custom PyTorch dataset
- Fine-tune a transformer model (GPT-2 or DistilGPT-2)
- Implement training and validation loops
- Save and load checkpoints
- Generate text from the fine-tuned model

### Key Concepts Covered

1. **Transfer Learning**: Using pre-trained models and adapting them to new tasks
2. **Language Modeling**: Next-token prediction for text generation
3. **Tokenization**: Converting text to numerical tokens
4. **Attention Mechanisms**: Understanding how transformers process sequences
5. **Optimization**: AdamW optimizer with learning rate scheduling
6. **Gradient Clipping**: Preventing exploding gradients
7. **Checkpointing**: Saving model states during training

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

```bash
python llm_finetuning_project.py
```

This will:
- Create sample training data (if not present)
- Initialize GPT-2 model
- Train for 2 epochs
- Save checkpoints and final model
- Generate sample text

### 3. Expected Output

```
Using device: cuda
Loading model: gpt2
Model loaded: gpt2
Model parameters: 124,439,808

Loading dataset...
Loaded text: X characters
Tokenized: Y tokens
Created Z training samples

Starting training for 2 epochs

Epoch 1/2
Training: 100%|████| 25/25 [00:45<00:00]
Train Loss: 2.3456
Val Loss: 2.2891

Epoch 2/2
Training: 100%|████| 25/25 [00:42<00:00]
Train Loss: 1.8234
Val Loss: 1.9123

Training completed!
Model saved to ./models/final_model
```

---

## 📊 Project Architecture

### File Structure

```
.
├── llm_finetuning_project.py    # Main training script
├── requirements.txt              # Dependencies
├── sample_data.txt              # Training data (auto-generated)
├── models/                      # Output directory
│   ├── checkpoint-epoch-1/      # Checkpoint after epoch 1
│   ├── checkpoint-epoch-2/      # Checkpoint after epoch 2
│   └── final_model/             # Final fine-tuned model
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── training_history.json
└── PROJECT_GUIDE.md             # This file
```

### Component Breakdown

#### 1. TextDataset Class
Handles data loading and preprocessing:
- Reads text file
- Tokenizes entire document
- Creates overlapping samples with stride
- Returns PyTorch tensors for training

```python
dataset = TextDataset(
    file_path='data.txt',
    tokenizer=tokenizer,
    max_length=512,
    stride=256
)
```

#### 2. LLMTrainer Class
Main training orchestrator:
- Model initialization
- Optimizer setup
- Training and validation loops
- Checkpointing
- Text generation

#### 3. Training Pipeline
```
Data Loading → Tokenization → Batching → 
Forward Pass → Loss Computation → 
Backward Pass → Gradient Update → Validation
```

---

## 🎯 Configuration & Hyperparameters

### Key Parameters to Tune

```python
CONFIG = {
    'model_name': 'gpt2',           # 'gpt2', 'distilgpt2', 'gpt2-medium'
    'batch_size': 4,                # Larger = faster but more memory
    'num_epochs': 2,                # More epochs = longer training
    'learning_rate': 5e-5,          # Lower = slower but more stable
    'max_length': 256,              # Token sequence length
    'val_split': 0.1,               # 10% for validation
}
```

### Tuning Tips

| Parameter | Increase if... | Decrease if... |
|-----------|----------------|----------------|
| Learning Rate | Loss decreases slowly | Loss oscillates wildly |
| Batch Size | GPU memory available | Out of memory errors |
| Max Length | Have long sequences | Too slow or OOM |
| Num Epochs | Validation loss still decreasing | Overfitting observed |

---

## 💻 Understanding the Code

### Forward Pass
```python
outputs = self.model(input_ids=input_ids, labels=labels)
loss = outputs.loss  # Language modeling loss (cross-entropy)
```

### Backward Pass
```python
loss.backward()  # Compute gradients
torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Clip to prevent exploding gradients
self.optimizer.step()  # Update weights
```

### Text Generation
```python
outputs = self.model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,      # Lower = more deterministic
    top_p=0.9,           # Nucleus sampling
    do_sample=True
)
```

---

## 📈 Monitoring Training

### Training History

The trainer automatically saves `training_history.json`:

```json
{
  "train_loss": [2.3456, 1.8234],
  "val_loss": [2.2891, 1.9123],
  "epoch": [1, 2]
}
```

### Visualization Script

```python
import json
import matplotlib.pyplot as plt

with open('models/final_model/training_history.json') as f:
    history = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)
plt.show()
```

---

## 🔧 Advanced Usage

### Using Your Own Data

```python
# Create a text file with your training data
with open('my_data.txt', 'w') as f:
    f.write(your_text_data)

# Update config
CONFIG['data_file'] = 'my_data.txt'

# Run training
python llm_finetuning_project.py
```

### Using Different Models

```python
# Smaller model (faster, less memory)
CONFIG['model_name'] = 'distilgpt2'

# Larger model (better quality, more resources)
CONFIG['model_name'] = 'gpt2-medium'

# Custom Hugging Face model
CONFIG['model_name'] = 'openai-community/gpt2'
```

### Loading and Using Fine-tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained('./models/final_model')
model = AutoModelForCausalLM.from_pretrained('./models/final_model')

# Generate text
input_ids = tokenizer.encode("AI is", return_tensors='pt')
outputs = model.generate(input_ids, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Resuming Training from Checkpoint

```python
# Load from checkpoint
trainer = LLMTrainer(model_name='./models/checkpoint-epoch-1')

# Continue training
trainer.train(train_dataset, val_dataset)
```

---

## 📚 Key Learning Resources

### Understanding Transformers
- Self-attention mechanism
- Position embeddings
- Multi-head attention
- Feed-forward networks

### PyTorch Concepts
- Automatic differentiation (autograd)
- Dynamic computation graphs
- DataLoader and Dataset
- Optimizer and scheduler

### Hugging Face API
- Model hub access
- Tokenizer types (BPE, WordPiece)
- Configuration files
- Model checkpointing

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Error
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size` (e.g., 4 → 2)
- Reduce `max_length` (e.g., 512 → 256)
- Use smaller model (`distilgpt2` instead of `gpt2-medium`)

### Loss Not Decreasing
**Solutions:**
- Increase learning rate (try 1e-4)
- Check data quality and quantity
- Increase number of epochs
- Verify GPU is being used (check with `torch.cuda.is_available()`)

### Slow Training
**Solutions:**
- Use smaller model
- Increase batch size (if memory allows)
- Use mixed precision training (add `from torch.cuda.amp import autocast`)
- Use multiple GPUs (DistributedDataParallel)

---

## 🎓 Extension Ideas

### 1. Multi-Task Learning
Train on multiple datasets simultaneously:
```python
# Load multiple datasets
wiki_dataset = TextDataset('wikipedia.txt', tokenizer)
arxiv_dataset = TextDataset('arxiv.txt', tokenizer)

# Combine datasets
combined = ConcatDataset([wiki_dataset, arxiv_dataset])
```

### 2. Instruction Fine-tuning
Format data as instructions:
```
<instruction>Summarize the following text:\n<text>...</text>\n<summary>...</summary>
```

### 3. Prompt-based Learning
Implement few-shot prompting:
```python
def few_shot_prompt(task_name, examples, query):
    prompt = f"Task: {task_name}\n\n"
    for example in examples:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt
```

### 4. Evaluation Metrics
Add BLEU, ROUGE, or Perplexity:
```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

def evaluate_generation(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'])
    # Compute metrics
```

### 5. Quantization & Compression
Reduce model size for deployment:
```python
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)
```

---

## 📖 Model Checkpointing Explained

During training, checkpoints save:

1. **Model weights** (`pytorch_model.bin`)
   - All trainable parameters
   - Current model state

2. **Configuration** (`config.json`)
   - Model architecture
   - Hyperparameters
   - Vocabulary size

3. **Tokenizer** (`tokenizer.json`)
   - Vocabulary
   - Merging rules (for BPE)
   - Special tokens

4. **Training metadata** (`training_history.json`)
   - Loss curves
   - Validation metrics
   - Epoch information

---

## 🎯 Project Evaluation Criteria

For college submission, focus on:

1. **Code Quality** (30%)
   - Clean, readable code
   - Proper documentation
   - Error handling

2. **Functionality** (30%)
   - Model trains and converges
   - Checkpointing works
   - Text generation produces coherent output

3. **Documentation** (20%)
   - Clear README
   - Inline comments
   - Architecture explanation

4. **Experimentation** (20%)
   - Multiple model sizes tested
   - Different hyperparameters
   - Loss curves and metrics

---

## 🚀 Next Steps

1. Run the project with sample data
2. Prepare your own dataset
3. Experiment with different hyperparameters
4. Implement evaluation metrics
5. Deploy the model (Flask/FastAPI)
6. Write comprehensive project report

---

## 📝 Example Project Report Structure

```
1. Introduction
   - Problem statement
   - Literature review

2. Methodology
   - Dataset description
   - Model architecture
   - Training procedure

3. Experiments
   - Hyperparameter settings
   - Training curves
   - Inference examples

4. Results & Analysis
   - Performance metrics
   - Generated text samples
   - Comparison with baselines

5. Conclusion & Future Work
   - Key findings
   - Limitations
   - Improvements
```

---

## 🔗 Useful Links

- **Hugging Face Docs**: https://huggingface.co/docs
- **PyTorch Docs**: https://pytorch.org/docs
- **Transformers Paper**: https://arxiv.org/abs/1706.03762
- **GPT-2 Paper**: https://openai.com/research/language-models-are-unsupervised-multitask-learers
- **Transfer Learning**: https://stanford.edu/~shervine/blog/transfer-learning

---

## 📄 License & Attribution

This project demonstrates concepts from:
- Hugging Face Transformers library
- PyTorch documentation
- Academic papers on transfer learning

---

**Happy Learning! 🎉**

Feel free to experiment, modify, and extend this project. The best way to learn is by doing!
