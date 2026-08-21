# LLM Fine-tuning Quick Start Guide

## ⚡ 5-Minute Setup

```bash
# 1. Clone or download the project files

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run training
python llm_finetuning_project.py
```

**That's it!** The script will:
- Download GPT-2 model
- Create sample data
- Train for 2 epochs
- Save checkpoints
- Generate sample text

---

## 📚 Core Components

### 1. Dataset Preparation

**Format**: Plain text file (one or multiple lines)

```python
# Prepare your data
with open('my_data.txt', 'w') as f:
    f.write(your_text_here)

# Update config
CONFIG['data_file'] = 'my_data.txt'
```

### 2. Model Selection

```python
# Small (fast, 82M params)
CONFIG['model_name'] = 'distilgpt2'

# Medium (balanced, 124M params)
CONFIG['model_name'] = 'gpt2'

# Large (better, 355M params)
CONFIG['model_name'] = 'gpt2-medium'
```

### 3. Hyperparameter Tuning

```python
CONFIG = {
    'batch_size': 4,        # ↓ OOM? Lower this
    'learning_rate': 5e-5,  # ↑ Loss plateaus? Increase
    'num_epochs': 2,        # ↑ Val loss still down? Increase
    'max_length': 256,      # ↑ Have long sequences? Increase
}
```

### 4. Training

```python
from llm_finetuning_project import LLMTrainer, TextDataset
import torch

# Initialize
trainer = LLMTrainer(model_name='gpt2')

# Load data
dataset = TextDataset('data.txt', trainer.tokenizer, max_length=256)

# Split train/val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Train
trainer.train(train_ds, val_ds)

# Save
trainer.save_model()
```

### 5. Generate Text

```python
generated = trainer.generate(
    prompt="AI is",
    max_length=50,
    temperature=0.7  # 0.0=deterministic, 1.0=random
)
print(generated[0])
```

### 6. Load Saved Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('./models/final_model')
model = AutoModelForCausalLM.from_pretrained('./models/final_model')

# Use for inference
input_ids = tokenizer.encode("Hello", return_tensors='pt')
outputs = model.generate(input_ids, max_length=50)
print(tokenizer.decode(outputs[0]))
```

---

## 🎯 Typical Workflow

### Step 1: Prepare Data
```bash
# Use your own text file or create sample
python llm_finetuning_project.py
# This creates sample_data.txt if not present
```

### Step 2: Configure Training
Edit `llm_finetuning_project.py`:
```python
CONFIG = {
    'model_name': 'gpt2',
    'data_file': 'your_data.txt',
    'batch_size': 4,
    'num_epochs': 3,
    'learning_rate': 5e-5,
}
```

### Step 3: Train Model
```bash
python llm_finetuning_project.py
```

### Step 4: Evaluate Results
```python
# Check training history
import json
with open('models/final_model/training_history.json') as f:
    history = json.load(f)
print(history)

# Generate text
from llm_finetuning_project import LLMTrainer
trainer = LLMTrainer()
trainer.model = # Load your model
result = trainer.generate("Your prompt here")
```

### Step 5: Deploy (Optional)
```python
from utils import ModelDeployer
ModelDeployer.create_flask_app('./models/final_model')
# Then run: python app.py
```

---

## 🔧 Common Modifications

### Use Different Model

```python
# Before training
CONFIG['model_name'] = 'gpt2-medium'  # Larger
# or
CONFIG['model_name'] = 'distilgpt2'   # Smaller
```

### Custom Dataset with Multiple Files

```python
from pathlib import Path

# Combine multiple files
all_text = ""
for txt_file in Path('.').glob('*.txt'):
    with open(txt_file) as f:
        all_text += f.read() + "\n"

with open('combined.txt', 'w') as f:
    f.write(all_text)

CONFIG['data_file'] = 'combined.txt'
```

### Adjust Sequence Length

```python
# Longer sequences = more context but slower & more memory
CONFIG['max_length'] = 256   # Short sequences
CONFIG['max_length'] = 512   # Medium sequences
CONFIG['max_length'] = 1024  # Long sequences (requires more GPU)
```

### Change Learning Rate Schedule

Add to `LLMTrainer.__init__`:
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

self.scheduler = CosineAnnealingLR(
    self.optimizer,
    T_max=self.num_epochs
)
```

Then in `train_epoch`, add after backward:
```python
self.scheduler.step()
```

---

## 📊 Monitoring Progress

### Real-time Monitoring
```bash
# Watch GPU usage (terminal)
watch -n 1 nvidia-smi

# Or use tensorboard
pip install tensorboard
tensorboard --logdir ./models
```

### Check Loss Curves
```python
from utils import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_training_history(
    'models/final_model/training_history.json',
    'plot.png'
)
```

### Print Training Stats
```python
import json
with open('models/final_model/training_history.json') as f:
    h = json.load(f)

for i, (train, val) in enumerate(zip(h['train_loss'], h['val_loss']), 1):
    ppl_train = round(2.71828 ** train, 2)
    ppl_val = round(2.71828 ** val, 2)
    print(f"Epoch {i}: Loss {train:.4f} → {val:.4f} | PPL {ppl_train} → {ppl_val}")
```

---

## 🚨 Troubleshooting Cheat Sheet

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `batch_size` or `max_length` |
| Loss not decreasing | Increase `learning_rate` or `num_epochs` |
| Training very slow | Use smaller model or increase `batch_size` |
| Generated text gibberish | Train longer or use more data |
| `No module named torch` | Run `pip install -r requirements.txt` |
| Model won't load | Check path in `AutoModelForCausalLM.from_pretrained()` |

---

## 💾 File Organization

```
project/
├── llm_finetuning_project.py  ← Main training script (run this)
├── utils.py                    ← Helper functions
├── requirements.txt            ← Dependencies
├── sample_data.txt            ← Auto-generated demo data
└── models/                    ← Output directory
    ├── checkpoint-epoch-1/
    ├── checkpoint-epoch-2/
    └── final_model/
        ├── config.json
        ├── pytorch_model.bin  ← The model weights
        ├── tokenizer.json
        └── training_history.json
```

---

## 📈 Expected Training Times (on 1 GPU)

| Model | Batch Size | Seq Length | Time/Epoch | Total (3 epochs) |
|-------|-----------|-----------|-----------|-----------------|
| DistilGPT-2 | 8 | 256 | ~30s | ~1.5 min |
| GPT-2 | 4 | 256 | ~1 min | ~3 min |
| GPT-2 | 2 | 512 | ~3 min | ~9 min |
| GPT-2-Medium | 2 | 256 | ~2 min | ~6 min |

---

## 🎓 Learning Path

### Beginner
1. Run with sample data
2. Understand the CONFIG
3. Try different models

### Intermediate
1. Use your own data
2. Modify hyperparameters
3. Check training curves
4. Generate from fine-tuned model

### Advanced
1. Implement custom evaluation metrics
2. Create deployment script
3. Experiment with prompt engineering
4. Try other architectures (T5, LLAMA, etc.)

---

## 🔗 Advanced Usage Examples

### Example 1: Transfer Learning Chain
```python
# Train on dataset A, then fine-tune on dataset B
trainer1 = LLMTrainer(model_name='gpt2')
trainer1.train(dataset_a)
trainer1.save_model('model_v1')

# Load and continue training
trainer2 = LLMTrainer(model_name='./models/model_v1')
trainer2.train(dataset_b)
trainer2.save_model('model_v2')
```

### Example 2: Few-shot Prompt Engineering
```python
def create_few_shot_prompt(task, examples, query):
    prompt = f"Task: {task}\n\n"
    for ex in examples:
        prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"
    prompt += f"Input: {query}\nOutput:"
    return prompt

examples = [
    {"input": "Summarize: AI is...", "output": "Artificial intelligence..."},
]
prompt = create_few_shot_prompt("Summarization", examples, "Summarize: ML is...")
result = trainer.generate(prompt)
```

### Example 3: Batch Inference
```python
from utils import TextGenerationDemo

prompts = [
    "The future of AI",
    "Machine learning is",
    "Deep learning enables"
]

results = TextGenerationDemo.batch_generation(
    trainer.model,
    trainer.tokenizer,
    prompts,
    device=trainer.device
)

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

### Example 4: Model Deployment
```python
from utils import ModelDeployer

# Create standalone inference script
ModelDeployer.create_inference_script(
    './models/final_model',
    'inference.py'
)

# Create Flask web app
ModelDeployer.create_flask_app(
    './models/final_model',
    'app.py'
)
```

---

## 📝 For College Project Submission

### Checklist
- [ ] Code runs without errors
- [ ] Training completes successfully
- [ ] Saves model and checkpoints
- [ ] Generates coherent text
- [ ] Has training history/plots
- [ ] Well-documented code
- [ ] README/guide included

### Report Structure
```
1. Introduction (problem, motivation)
2. Literature Review (transformers, fine-tuning)
3. Methodology (architecture, data, training)
4. Experiments (hyperparameters, results)
5. Results (loss curves, generation samples)
6. Conclusion (findings, limitations, future work)
```

---

## 🎉 You're Ready!

Run your first training:
```bash
python llm_finetuning_project.py
```

Good luck! 🚀
