import json
import re
import os

# Pfad zu deinem originalen Datensatz
input_file = "datasets/max.json"
output_file = "datasets/max_transformed.json"


# Funktion zum Extrahieren des Namens und der Nachricht aus einer Zeile
def extract_name_and_message(line):
    # Pattern für "[Datum, Zeit] Name: Nachricht"
    pattern = r'\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\] ([^:]+): (.*)'
    match = re.match(pattern, line)
    if match:
        name = match.group(1).strip()
        message = match.group(2).strip()
        # Ignoriere "Bild weggelassen"
        if message == "Bild weggelassen":
            return name, None
        return name, message
    return None, None


# Lade den originalen Datensatz
with open(input_file, 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# Neues Format für Llama 3.2
transformed_data = []

for entry in original_data["text"]:
    # Extrahiere Kontext und Antwort
    parts = entry.split("Response (Ali Günes):")
    if len(parts) != 2:
        continue

    context_part = parts[0].strip()
    response_part = parts[1].strip()

    # Extrahiere die letzten Nachrichten aus dem Kontext (ohne Instruction)
    context_lines = context_part.split("\n")
    instruction = context_lines[0] if context_lines[0].startswith("Instruction:") else ""

    # Finde die letzte Nachricht von Max im Kontext
    user_message = None
    for line in reversed(context_lines):
        name, message = extract_name_and_message(line)
        if name == "Max" and message is not None:
            user_message = message
            break

    if not user_message:
        continue  # Überspringe Einträge ohne Benutzernachricht

    # Extrahiere die Antwort von Ali
    response_message = None
    response_lines = response_part.split("\n")
    for line in response_lines:
        name, message = extract_name_and_message(line)
        if name == "Ali Günes" and message is not None:
            response_message = message
            break

    if not response_message:
        continue  # Überspringe Einträge ohne Antwort

    # Erstelle den Eintrag im neuen Format
    new_entry = {
        "messages": [
            {
                "role": "system",
                "content": "Respond as Ali Günes would."
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": response_message
            }
        ]
    }

    transformed_data.append(new_entry)

# Speichere den transformierten Datensatz
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=2)

print(f"Transformation abgeschlossen. {len(transformed_data)} Konversationspaare wurden erstellt.")
print(f"Ausgabe gespeichert in: {output_file}")