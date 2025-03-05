import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Multi-GPU Setup mit Torch 2.4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16  # RTX 3090 unterstützt fp16 für schnellere Berechnungen

# Modellname
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # Sicherstellen, dass Padding-Token korrekt gesetzt ist

# Funktion zur Tokenisierung des Datensatzes
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=250)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels für Causal Language Modeling setzen
    return tokenized

# Datensatz laden und tokenisieren
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# LoRA-Konfiguration für effizientes Fine-Tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Modell laden und LoRA anwenden
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Verteilt das Modell automatisch auf mehrere GPUs
    torch_dtype=torch_dtype
)
model = get_peft_model(model, lora_config)  # LoRA-Modell erstellen

# Trainingsargumente für Multi-GPU DDP
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=4,  # Kann angepasst werden, falls VRAM reicht
    gradient_accumulation_steps=8,  # Reduziert Speicherverbrauch durch Akkumulation
    num_train_epochs=50,
    learning_rate=2e-4,
    save_steps=100,
    save_total_limit=2,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,  # Mixed Precision aktivieren
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,  # Wichtig für DDP + LoRA
    report_to="none"
)

# Trainer mit DistributedDataParallel (DDP) starten
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
