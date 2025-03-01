import json

# Define the instruction to be prepended to every training example.
instruction = 'Instruction: "Fine tune on the responses by Ali Günes to the context messages."'

# Load the structured JSON dataset.
with open("structured_trainable.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)

# Filter entries: Keep only entries where the response sender is "Ali Günes".
filtered_entries = []
for entry in raw_data["max"]:
    if entry["response"]["sender"] == "Ali Günes":
        filtered_entries.append(entry)

# Create final training examples.
final_examples = []
for entry in filtered_entries:
    # Format context: each message becomes "[time] sender: message"
    context_lines = []
    for msg in entry["context"]:
        context_lines.append(f"[{msg['time']}] {msg['sender']}: {msg['message']}")
    context_str = "\n".join(context_lines)

    # Format the response.
    resp = entry["response"]
    response_str = f"[{resp['time']}] {resp['sender']}: {resp['message']}"

    # Combine instruction, context, and response into one text block.
    full_text = (
        f"{instruction}\n"
        f"Context:\n{context_str}\n"
        f"Response (Ali Günes):\n{response_str}"
    )

    final_examples.append( full_text)

# Create final dataset structure with a "train" key.
final_dataset = { "text":final_examples}

# Save the final training dataset.
with open("datasets/max.json", "w", encoding="utf-8") as file:
    json.dump(final_dataset, file, indent=4, ensure_ascii=False)

print("✅ Final training dataset saved as 'max.json'")