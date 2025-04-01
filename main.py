import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import Accelerator  # Wichtig für Multi-GPU

# Initialisiere Accelerator frühzeitig
accelerator = Accelerator()

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
tokenizer.padding_side = "right"

# Dataset laden und vorbereiten (wie zuvor)

# Modell-Loading angepasst für DDP
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": accelerator.device} if num_gpus > 1 else "auto",  # Kritische Änderung
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    ) if num_gpus <= 1 else None  # Quantisierung nur im Single-GPU Modus
)

# Vorbereitung für Multi-GPU
if num_gpus > 1:
    model = accelerator.prepare_model(model)  # Wichtig für DDP

# LoRA-Konfiguration (wie zuvor)
model = get_peft_model(model, lora_config)

# Training Arguments angepasst
training_args = TrainingArguments(
    output_dir=f"./fine_tuned/{CONTACT_NAME}",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    learning_rate=2e-5,
    # ... restliche Parameter
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    fsdp="full_shard offload" if num_gpus > 1 else None,  # FSDP für bessere Skalierung
    dataloader_pin_memory=True,
    report_to="none"
)

# Trainer mit Accelerator vorbereiten
trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
))

# Training starten
if accelerator.is_main_process:
    print("Starting training...")
trainer.train()

# ... restlicher Code
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
