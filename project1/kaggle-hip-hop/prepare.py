import os
import json

data = {}

for root, dirs, files in os.walk("data"):
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(root, file), 'rb') as f:
                data[file] = f.read().decode('utf-8', errors='ignore')

with open('data.txt', 'w') as f:
    f.write("\n".join(data.values()))

with open('data.json', 'w') as f:
    json.dump(data, f)