import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import os

# Für M3 Mac nutzen wir MPS oder CPU
try:
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
except:
    device = "cpu"

print(f"Using device: {device}")

# Modell und Tokenizer laden
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Dataset laden mit limitierter Größe
CONTACT_NAME = "max"
dataset = load_dataset("json", data_files=f"datasets/{CONTACT_NAME}_transformed.json")

# Reduziere das Dataset stark für den Test
small_dataset = dataset["train"].select(range(min(50, len(dataset["train"]))))
train_val_dict = small_dataset.train_test_split(test_size=0.2, seed=42)
print(f"Reduced training examples: {len(train_val_dict['train'])}")
print(f"Reduced validation examples: {len(train_val_dict['test'])}")


def format_instruction(example):
    """Formatiert die Nachrichten im Chat-Format für Llama 3.2 Instruct"""
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
    """Tokenisiert die Daten mit sehr kleiner Sequenzlänge"""
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,  # Extrem reduzierte Sequenzlänge
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

# Prüfe, ob LoRA-Only Modus möglich ist
try:
    # Im LoRA-Only Modus laden wir nur die Adapter-Gewichte
    from peft import PeftModel, PeftConfig

    print("Trying to load model in LoRA-only mode...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"Error in LoRA-only approach: {e}")
    print("Trying alternative lightweight approach...")

    # Alternativmethode falls LoRA-only nicht funktioniert
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )

# Extrem kleine LoRA-Konfiguration
lora_config = LoraConfig(
    r=2,  # Minimaler Rang
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    # Absolutes Minimum an Modulen
    target_modules=["q_proj"]  # Nur ein Modul
)

print("Preparing LoRA model...")
model = get_peft_model(model, lora_config)
print(f"Model trainable parameters: {model.print_trainable_parameters()}")

# Minimalste Trainingsparameter
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}_mac_test",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    max_steps=10,  # Nur 10 Trainingsschritte zum Testen
    learning_rate=1e-4,
    save_strategy="no",
    evaluation_strategy="no",  # Keine Evaluierung
    logging_steps=1,
    remove_unused_columns=False,
    fp16=False,
    dataloader_num_workers=0,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Training starten
print("Starting minimal training...")
try:
    trainer.train()
    # Fintuned Modell und Tokenizer speichern
    model.save_pretrained(f"./fine_tuned/{CONTACT_NAME}_mac_test")
    tokenizer.save_pretrained(f"./fine_tuned/{CONTACT_NAME}_mac_test")
    print(f"✅ Test fine-tuning completed! Weights saved in './fine_tuned/{CONTACT_NAME}_mac_test'")
except Exception as e:
    print(f"Training error: {e}")
    print("Your Mac might not have enough resources for even this minimal training.")
    print("Consider using a cloud service like Google Colab for the actual training.")


# Vereinfachte Testfunktion für Inferenz
def test_model(model, tokenizer, test_prompts):
    """Testet das Modell mit einem Beispiel-Prompt"""
    model.eval()
    # Nur einen Prompt testen
    prompt = test_prompts[0]
    print(f"\nTesting prompt: {prompt}")
    formatted_prompt = f"<|system|>\nRespond as Ali Günes would.\n<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,  # Sehr kurze Ausgabe
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {generated_text}")
    except Exception as e:
        print(f"Error during generation: {e}")


# Test prompt
test_prompts = ["Willst du morgen ins gym gehen?"]

# Modell testen
print("Testing the model with a simple prompt...")
try:
    test_model(model, tokenizer, test_prompts)
except Exception as e:
    print(f"Testing error: {e}")