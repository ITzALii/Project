# Load model directly
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def tokenize_function(examples):

    # Tokenize the "text" field; adjust max_length as needed.
    tokenizer.pad_token = tokenizer.eos_token

    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=250)
    print("x")

    # For causal language modeling, labels can be the same as input_ids.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load base model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float32  # Required for Mac M3 (MPS)
)

# Contact-specific dataset (replace with each contact’s dataset)
CONTACT_NAME = "max"  # Change this per contact
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset)
print(tokenized_dataset["train"][0])  # Inspect the first example
print(tokenized_dataset["train"][1])  # Inspect the second example
# Configure LoRA to train only small weight updates
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training settings
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    remove_unused_columns=False,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
)

# Train model
trainer.train()

# Save LoRA adapter for this contact
model.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")
tokenizer.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")

print(f"✅ Fine-tuning completed for {CONTACT_NAME}! Weights saved in './fine_tuned/{CONTACT_NAME}'")