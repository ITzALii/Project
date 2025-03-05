import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the path to your fine-tuned model (adjust if necessary)
MODEL_PATH = "./fine_tuned/max"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Move the model to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Interactive Prompt Mode. Type 'quit' or 'exit' to end.")
while True:
    # Get user prompt from the terminal
    prompt="respond to this message like Ali:"
    rprompt= input("\nPrompt: ")
    prompt = prompt+ rprompt
    if prompt.lower() in ["quit", "exit"]:
        break

    # Tokenize the prompt and move tensors to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate a response from the model
    outputs = model.generate(
        **inputs,
        max_length=250,      # Adjust the maximum length as needed
        do_sample=True,      # Enable sampling for more creative outputs
        top_k=50,            # Consider only the top_k tokens for sampling
        top_p=0.95           # Use nucleus sampling
    )

    # Decode the generated tokens into text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nResponse:", response)
