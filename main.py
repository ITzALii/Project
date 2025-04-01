import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os
from sklearn.model_selection import train_test_split

# Für RTX 3090 nutzen wir CUDA und Quantisierung
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modell und Tokenizer laden
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Wichtig für korrekte Padding-Position

# Dataset laden (mit dem neuen Format)
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}_transformed.json")

# Split in Train und Validation
train_val_dict = dataset["train"].train_test_split(test_size=0.1, seed=42)


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

# 4-bit Quantisierung konfigurieren
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16  # bfloat16 ist effizienter als float32
)

# Modell mit Quantisierung laden
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
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

# Optimierte Trainingsparameter
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=2,  # Erhöhe wenn möglich für bessere Effizienz
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-5,  # Etwas niedrigere Learning Rate für Stabilität
    warmup_ratio=0.03,  # Warmup als Verhältnis statt fester Schritte
    save_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_total_limit=3,
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,
    optim="adamw_torch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
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


# Testfunktion für Inferenz
def test_model(model, tokenizer, test_prompts):
    """Testet das Modell mit Beispiel-Prompts"""
    model.eval()
    for prompt in test_prompts:
        # Formatiere den Prompt im korrekten Format für Llama 3.2
        formatted_prompt = f"<|system|>\nRespond as Ali Günes would.\n<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

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
        response_start = formatted_prompt.strip()
        if generated_text.startswith(response_start):
            response = generated_text[len(response_start):].strip()
        else:
            response = generated_text

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)


# Test prompts
test_prompts = [
    "Willst du morgen ins gym gehen?",
    "Ich habe neue Bücher gekauft.",
    "Wann treffen wir uns?"
]

# Modell testen
test_model(model, tokenizer, test_prompts)