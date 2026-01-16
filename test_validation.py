import json

path = "examples/training/finetune/MatrixGame2.0/validation_phase2.json"

with open(path, "r") as f:
    data = json.load(f)

action_patterns = [
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
]

action_map = {
    tuple(action_patterns[0]): "Left  (0)",
    tuple(action_patterns[1]): "Stop  (1)",
    tuple(action_patterns[2]): "Right (2)",
}

def get_action_name(action_list):
    key = tuple(action_list)
    return action_map.get(key, f"Unknown {action_list}")

# sequence_indices = [0, 2, 2, 0, 2, 2, 0] 
sequence_indices = [0, 0, 2, 0, 0, 2, 0]
frames_per_action = 12
max_length = 77

for index, entry in enumerate(data["data"]):
    print(index, entry["caption"])

    total_len = len(entry["keyboard_cond"])
    for i in range(0, total_len, 12):
        chunk = entry["keyboard_cond"][i : i + 12]
        first_action = tuple(chunk[0])
        is_consistent = all(tuple(x) == first_action for x in chunk)
        
        start_frame = i
        end_frame = min(i + 12, total_len)
        chunk_len = len(chunk)
        
        if is_consistent:
            action_name = get_action_name(chunk[0])
            print(f"  [frame {start_frame:02d}-{end_frame:02d}]: {action_name} * {chunk_len}")
        else:
            print(f"  [frame {start_frame:02d}-{end_frame:02d}]: Mixed - {chunk_len}")

    if index == 7:
        full_sequence = []
        for action_idx in sequence_indices:
            action = action_patterns[action_idx]
            full_sequence.extend([action] * frames_per_action)
    
        current_len = len(full_sequence)
        if current_len > max_length:
            full_sequence = full_sequence[:max_length]
        else:
            full_sequence.extend([action_patterns[1]] * (max_length - current_len))
        entry["keyboard_cond"] = full_sequence

with open(path, "w") as f:
    json.dump(data, f)
