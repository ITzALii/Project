import json
import re

# Input file: update with the path to your current dataset JSON file.
input_file = "datasets/max.json"
output_file = "datasets/max_converted_dataset.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = {"train": []}

# Process each training example in your dataset.
for example in data["text"]:
    text = example

    # Remove everything before "Context:" (i.e. drop the instruction)
    context_index = text.find("Context:")
    if context_index == -1:
        continue  # Skip if "Context:" is not found
    text = text[context_index:]

    # Split the text into context and response parts.
    # This regex splits on "Response" followed by an optional parenthesized string and a colon.
    parts = re.split(r"Response\s*\(.*?\):", text, flags=re.IGNORECASE)
    if len(parts) < 2:
        continue  # Skip if no response part is found.

    # The first part is the context (which starts with "Context:"), the second is the response.
    context_part = parts[0].strip()
    response_part = parts[1].strip()

    # Remove the "Context:" header from the context part.
    if context_part.lower().startswith("context:"):
        context_part = context_part[len("Context:"):].strip()

    # Create a new dictionary for the example.
    new_example = {
        "context": context_part,
        "response": response_part
    }
    new_data["train"].append(new_example)

# Save the converted dataset.
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"Dataset conversion complete. Saved as {output_file}")