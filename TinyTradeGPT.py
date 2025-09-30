import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# 1. Load GPT-2 model and tokenizer
model_name = "gpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding token error

model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. LoRA config for GPT-2 attention layers
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attn.c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 3. Load finance instruction dataset from Hugging Face
dataset = load_dataset("sujet-ai/Sujet-Finance-Instruct-177k")

# 4. Tokenize dataset with map, handling columns properly
def tokenize_fn(examples):
    inputs = [inp + tokenizer.eos_token + ans for inp, ans in zip(examples["inputs"], examples["answer"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return {**examples, **model_inputs}

# Remove original text columns that are replaced
columns_to_remove = ["instruction", "output"]

tokenized_dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["Unnamed: 0", "inputs", "answer", "system_prompt", "user_prompt", "task_type", "dataset", "index_level", "conversation_id"]
)

# 5. Training args
training_args = TrainingArguments(
    output_dir="./finance-gpt2-lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy="steps",        # Changed here
    eval_steps=500,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
)

# 6. Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"].select(range(1000)),  # Use a subset for evaluation
    tokenizer=tokenizer
)

trainer.train()

# 7. Save model and tokenizer
model.save_pretrained("./finance-gpt2-lora")
tokenizer.save_pretrained("./finance-gpt2-lora")
