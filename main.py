import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# For an RTX 3090, we use CUDA and half-precision (fp16)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16  # RTX 3090 supports fp16

def tokenize_function(examples):

    # Tokenize the "text" field; adjust max_length as needed.
    tokenizer.pad_token = tokenizer.eos_token
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=250)
    # For causal language modeling, labels can be the same as input_ids.
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load base model and tokenizer from your local cache or HF Hub path
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",           # Ensures that the model is loaded onto the GPU.
    torch_dtype=torch_dtype      # Use fp16 for speed and memory efficiency.
)

# Load your dataset and tokenize it.
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True)


# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training settings optimized for an RTX 3090:
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=2,        # Increase batch size if memory allows
    gradient_accumulation_steps=4,          # Fewer accumulation steps due to larger batch size
    num_train_epochs=15,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,                            # Enable mixed precision for CUDA GPUs
    dataloader_num_workers=4              # Use multiple workers to speed up data loading
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Start training
trainer.train()

# Save the fine-tuned LoRA adapter and tokenizer
model.save_pretrained(f"./fine_tuned2/{CONTACT_NAME}")
tokenizer.save_pretrained(f"./fine_tuned2/{CONTACT_NAME}")

print(f"✅ Fine-tuning completed for {CONTACT_NAME}! Weights saved in './fine_tuned2/{CONTACT_NAME}'")
