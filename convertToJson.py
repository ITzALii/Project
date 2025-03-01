import json
import re
chat_dir="/Users/ali/Documents/chats_whatsapp/"
chatName=input("Enter the name of the chat: ")
directory=chat_dir+chatName
#f=open(directory,"r")
#x=open(chatName+".json","a")
#j=json.dumps(f.read(),ensure_ascii=False)
#x.write(j)

with open(chatName+".json", "r", encoding="utf-8") as file:
    text = file.read()

# Regex pattern to find timestamps
pattern = r"(\[\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2}\])"

# Replace: Add a `"` before each timestamp
#text = re.sub(pattern, r'"],["\1', text)

# Save the modified file
#with open(chatName+".json", "w", encoding="utf-8") as file:
    #file.write(text)

print("Timestamps modified and saved ")

pattern = r'^\[(\d{2}\.\d{2}\.\d{2}, \d{2}:\d{2}:\d{2})\] ([^:]+): (.+)'

finalProduct= []
# Match regex pattern
with open(chatName+".json", "r", encoding="utf-8") as file:
    text2 = json.load(file)
text1= [entry for entry in text2['max']]
i=0
for entry in text1:
    match = re.match(pattern, entry[0].strip('"'))

    if match:
        time, sender, message = match.groups()

    # Create JSON object
        structured_data = [{

        "time": time,
        "sender": sender,
        "message": message

        },
            ]


    # Convert to JSON format

        json_output = json.dumps(structured_data, indent=4, ensure_ascii=False)
        with open(chatName+".json", "a", encoding="utf-8") as file:

            file.write(json_output)
            file.write(",\n")

    else:
        print("no match")
    i=i+1


