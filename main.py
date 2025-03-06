import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Multi-GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16  # RTX 3090 unterstützt fp16

# Modellname
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Padding-Token setzen

# Tokenizer-Funktion
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=250)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels für Causal Language Modeling setzen
    return tokenized

# Datensatz laden und tokenisieren
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA-Konfiguration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Modell laden OHNE `device_map="auto"`
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype
).to(device)  # Manuelle Zuweisung an GPU

model = get_peft_model(model, lora_config)  # LoRA-Modell erstellen

# Trainingsargumente für Multi-GPU mit DDP
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=8,  
    num_train_epochs=50,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,  
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,  # Wichtig für DDP + LoRA
    report_to="none",
    label_names=["labels"]  # Fix für fehlende `label_names`
)

# Trainer mit DistributedDataParallel (DDP)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Training starten
trainer.train()

# Modell und Tokenizer speichern
model.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")
tokenizer.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")

print(f"✅ Fine-Tuning abgeschlossen für {CONTACT_NAME}! Weights gespeichert unter './fine_tuned/{CONTACT_NAME}'")
