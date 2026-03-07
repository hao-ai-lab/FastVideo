import json
import os

# Paths for two image directories
image_dir_val = "/mnt/weka/home/hao.zhang/kaiqin/traindata_0205_1330/data/0_static_plus_w_only/first_frame"
image_dir_train = "/mnt/weka/home/hao.zhang/kaiqin/traindata_0205_1330/data/0_same_1st_frame_static_plus_w_only/first_frame"

# Action paths (used for both)
action_still = "/mnt/weka/home/hao.zhang/kaiqin/traindata_0205_1330/data/0_static_plus_w_only/videos/000000_action.npy"
action_w = "/mnt/weka/home/hao.zhang/kaiqin/traindata_0205_1330/data/0_static_plus_w_only/videos/001050_action.npy"

# Output path
output_path = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/validation_static_w.json"

# Fixed fields
fixed_fields = {
    "video_path": None,
    "num_inference_steps": 40,
    "height": 352,
    "width": 640,
    "num_frames": 77
}

data = []

# 16 images from each directory, alternating: val (0,1), train (2,3), val (4,5), train (6,7), ...
for i in range(16):
    # Val images: indices 0,1, 4,5, 8,9, ... (pair index 0, 2, 4, ...)
    image_path_val = os.path.join(image_dir_val, f"{i:06d}.png")
    
    # Still action for val
    data.append({
        "caption": f"val {i:02d} - Still",
        "image_path": image_path_val,
        "action_path": action_still,
        **fixed_fields
    })
    
    # W action for val
    data.append({
        "caption": f"val {i:02d} - W",
        "image_path": image_path_val,
        "action_path": action_w,
        **fixed_fields
    })
    
    # Train images: indices 2,3, 6,7, 10,11, ... (pair index 1, 3, 5, ...)
    image_path_train = os.path.join(image_dir_train, f"{i:06d}.png")
    
    # Still action for train
    data.append({
        "caption": f"train {i:02d} - Still",
        "image_path": image_path_train,
        "action_path": action_still,
        **fixed_fields
    })
    
    # W action for train
    data.append({
        "caption": f"train {i:02d} - W",
        "image_path": image_path_train,
        "action_path": action_w,
        **fixed_fields
    })

# Write to file
output = {"data": data}
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(data)} entries to {output_path}")
