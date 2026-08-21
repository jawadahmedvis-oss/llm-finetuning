
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load your trained model
tokenizer = AutoTokenizer.from_pretrained('./models/final_model')
model = AutoModelForCausalLM.from_pretrained('./models/final_model')

# Generate text
prompt = "Artificial Intelligence"
input_ids = tokenizer.encode(prompt, return_tensors='pt')
outputs = model.generate(input_ids, max_length=100)
result = tokenizer.decode(outputs[0])

print(f"Prompt: {prompt}")
print(f"Generated: {result}")
```