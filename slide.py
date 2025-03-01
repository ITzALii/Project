import json
import re

# Load the JSON file
with open("Max_chat.txt.json", "r", encoding="utf-8") as file:
    raw_data = json.load(file)

# Extract messages and remove unnecessary nesting
messages = [entry[0] for entry in raw_data["max"]]

# Convert messages into structured context-response pairs
formatted_data = {"max": []}
window_size = 3  # Adjust how many previous messages are used as context

for i in range(window_size, len(messages)):
    context_messages = messages[i - window_size:i]  # Get last 3 messages as context
    response_message = messages[i]  # Current message as response

    formatted_data["max"].append({
        "context": [
            {
                "time": msg["time"],
                "sender": msg["sender"],
                "message": msg["message"]
            } for msg in context_messages
        ],
        "response": {
            "time": response_message["time"],
            "sender": response_message["sender"],
            "message": response_message["message"]
        }
    })

# Save the corrected JSON format
with open("structured_trainable.json", "w", encoding="utf-8") as file:
    json.dump(formatted_data, file, indent=4, ensure_ascii=False)

print("✅ JSON reformatted correctly and saved as 'structured_trainable.json'")