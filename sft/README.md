# Banking-QA Llama 3.2-3B-Instruct `(BankLlama-3B)`

[![Open In Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/yuvraj17/BankLlama-3B)
[![Open In WandB](https://img.shields.io/badge/Weights%20&%20Biases-Report-blue)](https://api.wandb.ai/links/my-sft-team/8vvmzr4y)

A specialized QLoRA fine-tuned version of Meta's `Llama-3.2-3B-Instruct` model optimized for banking and financial customer service question answering.

## Model Overview

This model has been fine-tuned on the [Banking QA Dataset](https://huggingface.co/datasets/yuvraj17/banking-qa-dataset) to provide accurate, helpful responses to common banking queries. The model maintains the conversational capabilities of the base Llama 3.2 while enhancing its domain-specific knowledge in financial services.

### Key Features

- **Enhanced Banking Domain Knowledge**: Fine-tuned to understand and respond to a wide range of banking and financial queries
- **Optimized for Customer Service**: Provides clear, concise, and helpful responses to common banking questions
- **Efficient Deployment**: Using QLoRA for parameter-efficient fine-tuning, making it deployable on consumer hardware

## LASER Fine-Tuning Approach

This model utilizes the **LASER (Layer-Selective Rank Reduction)** technique for efficient fine-tuning. LASER identifies and targets the 50% of layers with the highest Signal-to-Noise Ratio (SNR) values, focusing computational resources where they'll have the most impact.

Key aspects of our LASER implementation:

- Selected the top 50% of layers with highest SNR values
- Targeted specific projection matrices within transformer blocks
- Achieved superior performance while maintaining model efficiency
- Reduced training time and computational requirements

The layer selection in the configuration file reflects this targeted approach, focusing on layers with the highest information density.

## Using the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ID from Hugging Face (Currently the Model is Private)
model_id = "yuvraj17/BankLlama-3B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Format your input as an instruction
instruction = "What are the different types of bank accounts available in NUST?"
prompt = f"<|begin_of_text|><|user|>\n{instruction}<|end_of_turn|>\n<|assistant|>\n"

# Generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Details

This model was fine-tuned using [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), a powerful framework for training and fine-tuning language models.

### Training Configuration

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit
- **Training Dataset**: [Banking QA Dataset](https://huggingface.co/datasets/yuvraj17/banking-qa-dataset)
- **Epochs**: 4
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (micro) / 8 (effective with gradient accumulation)
- **Sequence Length**: 4000 tokens
- **LoRA Configuration**:
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
  - Target: Selected layers with highest SNR values

### Performance

Full training metrics and evaluation results are available in the [Weights & Biases report](https://api.wandb.ai/links/my-sft-team/8vvmzr4y).

## License

This model is subject to the Meta Llama 3 license. Please refer to [Meta's licensing terms](https://llama.meta.com/llama3/license/) for usage restrictions and permissions.

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for the base Llama 3.2 model
- [OpenAccess-AI-Collective](https://github.com/OpenAccess-AI-Collective) for the Axolotl framework
