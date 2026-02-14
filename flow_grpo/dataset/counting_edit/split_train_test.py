import json
import random

# Load data from the JSONL file
data = []
with open('output.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Shuffle the data using random
random.seed(42)
random.shuffle(data)

# Split into test set (128 samples) and training set (remaining)
test_set = data[:112]
train_set = data[112:]

# Save the test set
with open('test_metadata.jsonl', 'w') as f:
    for item in test_set:
        json.dump(item, f)
        f.write('\n')

# Save the training set
with open('train_metadata.jsonl', 'w') as f:
    for item in train_set:
        json.dump(item, f)
        f.write('\n')

print(f"Test set size: {len(test_set)}")
print(f"Training set size: {len(train_set)}")
