import json
import os 

path = "data/Mochi-Synthetic-Data/videos2caption.json"

with open(path, 'r') as f:
    data = json.load(f)
    
# save with indent
with open(path, 'w') as f:
    json.dump(data, f, indent=4)