import json
import os

# Paths
image_dir = "/mnt/weka/home/hao.zhang/kaiqin/traindata_0205_1330/data/0_static_plus_w_only/first_frame"
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

# 32 images, each with 2 actions (Still and W)
for i in range(32):
    image_path = os.path.join(image_dir, f"{i:06d}.png")
    
    # Still action
    data.append({
        "caption": f"{i:02d} - Still",
        "image_path": image_path,
        "action_path": action_still,
        **fixed_fields
    })
    
    # W action
    data.append({
        "caption": f"{i:02d} - W",
        "image_path": image_path,
        "action_path": action_w,
        **fixed_fields
    })

# Write to file
output = {"data": data}
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(data)} entries to {output_path}")
