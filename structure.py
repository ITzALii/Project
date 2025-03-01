import json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch
# Set your parameters
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
CONTACT_NAME = "max"  # Your dataset file is assumed to be "datasets/max.json"

# Load the base model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32  # For Mac M3 (MPS)
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load your dataset (which currently has a "train" column with structured dicts)
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}.json")
print(dataset)

# Function to convert each example to a single text string
def combine_example(example):
    # Expecting each example is a dict under the key "train"
    # e.g., example = {"train": {"context": [...], "response": {...}}}
    if isinstance(example["train"], dict):
        # Convert each context message to a string
        context_lines = [
            f"[{msg['time']}] {msg['sender']}: {msg['message']}"
            for msg in example["train"].get("context", [])
        ]
        context_str = "\n".join(context_lines)
        # Convert response to a string
        resp = example["train"].get("response", {})
        response_str = f"[{resp.get('time', '')}] {resp.get('sender', '')}: {resp.get('message', '')}"
        # Prepend the instruction
        instruction = 'Instruction: "Fine tune on the responses by Ali Günes to the context messages."'
        full_text = f"{instruction}\nContext:\n{context_str}\nResponse (Ali Günes):\n{response_str}"
        return {"text": full_text}
    else:
        # Fallback: if "train" isn't a dict, just use its value as text
        return {"text": example["train"]}

# Map the conversion function to create a new "text" field.
dataset = dataset.map(combine_example)
# Optional: remove the old "train" column if desired:
dataset = dataset.remove_columns(["train"])
print(dataset)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False  # Ensure no columns are dropped.
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned LoRA adapter
model.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")
tokenizer.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")

print(f"✅ Fine-tuning completed for {CONTACT_NAME}! Weights saved in './fine_tuned/{CONTACT_NAME}'")