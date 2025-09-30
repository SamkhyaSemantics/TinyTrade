# TinyTrade
Trade Finance Transformers based on GPT2

## Trade Finance Domain Fine-Tuning GPT-2 LoRA Model
This project contains code to fine-tune a GPT-2 small language model using LoRA adapters (PEFT) on a finance-focused instruction dataset from Hugging Face.

## Key Features
Model: GPT-2 base model fine-tuned with parameter-efficient low-rank adapters (LoRA).

## Dataset: 

Uses sujet-ai/Sujet-Finance-Instruct-177k consisting of 177k financial instructions and outputs.

## Tokenization: 

Custom tokenization merges instruction and response sequences with end-of-sequence tokens, ensuring model knows when prompts end.

## Training: 

Configured with Hugging Face Trainer and TrainingArguments supporting evaluation, saving best model, and mixed precision if GPU is available.

## Padding fix: 

Sets tokenizer’s padding token to EOS token, required for GPT-2 since it lacks an explicit pad token.

## PEFT integration: 

Efficient training by injecting LoRA modules into GPT-2 attention layers only (attn.c_attn), saving memory and speeding up fine-tuning.

Easy extensibility: Can be adapted for other small open-source LLMs or domain-specific datasets.

## Requirements
- Python 3.9+
- PyTorch
- Transformers >= 4.45.0
- Datasets library
- PEFT library (pip install peft)
- Access to Hugging Face's datasets

## Usage
Clone this repo or copy the example script.

## Install dependencies:
pip install torch transformers datasets peft

## Run the training script:

python tradefinance_finetune.py

Trained model and tokenizer will be saved in ./finance-gpt2-lora.

## Important Notes

- The dataset columns used are instruction and output from the Hugging Face dataset.
- tokenizer.pad_token is explicitly set to tokenizer.eos_token due to GPT-2's tokenizer limitations.
- The training arguments use eval_strategy instead of deprecated evaluation_strategy.
- LoRA targets only the attn.c_attn modules specific to GPT-2 architecture. Modify for other models.
- Adjust hyperparameters like epochs, batch size, or learning rate for your compute setup.
