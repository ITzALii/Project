import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split
import torch.distributed as dist

# Prüfe Multi-GPU Verfügbarkeit
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
else:
    num_gpus = 0
    print("No GPUs found, using CPU")

# Modell und Tokenizer laden
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Wichtig für korrekte Padding-Position

# Dataset laden
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}_transformed.json")

# Split in Train und Validation
train_val_dict = dataset["train"].train_test_split(test_size=0.1, seed=42)
print(f"Training examples: {len(train_val_dict['train'])}")
print(f"Validation examples: {len(train_val_dict['test'])}")

def format_instruction(example):
    """
    Formatiert die Nachrichten im Chat-Format für Llama 3.2 Instruct
    """
    messages = example["messages"]
    
    formatted_text = ""
    for msg in messages:
        if msg["role"] == "system":
            formatted_text += f"<|system|>\n{msg['content']}\n"
        elif msg["role"] == "user":
            formatted_text += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            formatted_text += f"<|assistant|>\n{msg['content']}\n"
    
    # Füge den Assistenten-Prompt für die Generierung hinzu
    formatted_text += "<|assistant|>\n"
    
    return {"text": formatted_text}

# Anwenden der Formatierung
formatted_dataset = train_val_dict.map(format_instruction)

def tokenize_function(examples):
    """Tokenisiert die Daten mit passenden Einstellungen"""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Labels für Causal LM
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Setze Label für Padding-Token auf -100 (werden im Loss ignoriert)
    padding_mask = tokenized["attention_mask"] == 0
    tokenized["labels"][padding_mask] = -100
    
    return tokenized

# Tokenisiere das Dataset
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "messages"]
)

# 4-bit Quantisierung für effizienteres Multi-GPU Training
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Multi-GPU Device-Mapping
if num_gpus > 1:
    # Automatisches Device-Mapping für Multi-GPU
    device_map = "auto"
else:
    # Einfaches Mapping für Single-GPU
    device_map = {"": 0} if torch.cuda.is_available() else "cpu"

# Modell mit Quantisierung laden
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16
)

# Modell für Quantisiertes Training vorbereiten
model = prepare_model_for_kbit_training(model)

# LoRA-Konfiguration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Ziele alle wichtigen Module für bessere Anpassung
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, lora_config)

# Optimierte Trainingsparameter für Multi-GPU
# Berechne Batch-Größe und Gradient-Akkumulationsschritte basierend auf verfügbaren GPUs
base_batch_size = 2
if num_gpus > 1:
    per_device_batch_size = base_batch_size
    gradient_accumulation_steps = 4
else:
    per_device_batch_size = base_batch_size
    gradient_accumulation_steps = 8

training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=5,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_total_limit=3,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    # Multi-GPU Konfiguration
    local_rank=-1,  # Wird automatisch gesetzt, wenn mit torch.distributed.launch gestartet
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=50,
    dataloader_num_workers=4 if num_gpus > 1 else 2,
    gradient_checkpointing=True  # Hilfreich bei Multi-GPU für Speichereffizienz
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Training starten
trainer.train()

# Fintuned Modell und Tokenizer speichern
model.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")
tokenizer.save_pretrained(f"./fine_tuned/{CONTACT_NAME}")

print(f"✅ Fine-tuning abgeschlossen für {CONTACT_NAME}! Weights in './fine_tuned/{CONTACT_NAME}' gespeichert")

# Testfunktion für Inferenz (nur auf einer GPU)
def test_model(model, tokenizer, test_prompts):
    """Testet das Modell mit Beispiel-Prompts"""
    model.eval()
    # Setze das Modell auf die erste GPU für Inferenz
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt}")
        # Formatiere den Prompt im korrekten Format für Llama 3.2
        formatted_prompt = f"<|system|>\nRespond as Ali Günes would.\n<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extrahiere nur die Antwort des Assistenten
            response_parts = generated_text.split("<|assistant|>")
            if len(response_parts) > 1:
                response = response_parts[-1].strip()
            else:
                response = generated_text
                
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during generation: {e}")
        print("-" * 50)

# Test prompts
test_prompts = [
    "Willst du morgen ins gym gehen?",
    "Ich habe neue Bücher gekauft.",
    "Wann treffen wir uns?"
]

# Modell nur auf dem primären Prozess testen, wenn Multi-GPU verwendet wird
is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
if is_main_process:
    print("Testing the model...")
    test_model(model, tokenizer, test_prompts)
