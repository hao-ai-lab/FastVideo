import json

initial_path = "/workspace/codefolder/FastVideo/assets/mask_strategy_hunyuan.json"

with open(initial_path, "r") as f:
    data = json.load(f)
    for key in data.keys():
        t, h, w = data[key]
        data[key] = [min(2,t), h, w]

save_path = "/workspace/codefolder/FastVideo/assets/test_mask_strategy_hunyuan.json"


# print unique values for new data
unique_values = set()
for key in data.keys():
    unique_values.add(tuple(data[key]))
print(unique_values)

with open(save_path, "w") as f:
    json.dump(data, f, indent=4)