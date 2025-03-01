import json

# Load the structured JSON
with open("structured_trainable.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Define your name (so we keep only your messages)
your_name = "Ali Günes"

# Create a new dataset where only your responses are kept
filtered_data = {"max": []}

for entry in data["max"]:
    response = entry["response"]

    # Check if your name is the sender of the response
    if response["sender"] == your_name:
        filtered_data["max"].append(entry)  # Keep the entire entry

# Save the filtered JSON
with open("datasets/max.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, indent=4, ensure_ascii=False)

print("✅ Filtered dataset saved as 'max.json' (only your responses).")