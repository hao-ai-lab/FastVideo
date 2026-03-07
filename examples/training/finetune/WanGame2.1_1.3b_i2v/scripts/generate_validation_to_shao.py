#!/usr/bin/env python3
"""
Generate a validation JSON where all image_path and action_path entries come from
examples/training/finetune/WanGame2.1_1.3b_i2v/to_shao.

Scans each subfolder of to_shao for pairs (NN.jpg, NN_action.npy) and emits one
validation entry per pair. Uses the same fixed_fields as generate_validation.py.
"""
import json
import os

# Paths relative to this script / repo
FINETUNE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TO_SHAO_DIR = os.path.join(FINETUNE_DIR, "to_shao")
OUTPUT_PATH = os.path.join(FINETUNE_DIR, "validation_to_shao.json")

# Same fixed fields as generate_validation.py
FIXED_FIELDS = {
    "video_path": None,
    "num_inference_steps": 40,
    "height": 352,
    "width": 640,
    "num_frames": 77,
}


def collect_samples_from_to_shao():
    """Yield (caption, image_path, action_path) for each sample under to_shao."""
    if not os.path.isdir(TO_SHAO_DIR):
        raise FileNotFoundError(f"to_shao directory not found: {TO_SHAO_DIR}")

    for subdir_name in sorted(os.listdir(TO_SHAO_DIR)):
        subdir = os.path.join(TO_SHAO_DIR, subdir_name)
        if not os.path.isdir(subdir):
            continue
        # Find all NN.jpg (or NN.jpeg) and matching NN_action.npy
        for f in sorted(os.listdir(subdir)):
            if f.endswith(".jpg") or f.endswith(".jpeg"):
                base = f[: -4] if f.endswith(".jpg") else f[:-5]
                action_name = f"{base}_action.npy"
                action_path = os.path.join(subdir, action_name)
                if not os.path.isfile(action_path):
                    continue
                image_path = os.path.join(subdir, f)
                caption = f"to_shao/{subdir_name}/{base}"
                yield caption, image_path, action_path


def main():
    data = []
    for caption, image_path, action_path in collect_samples_from_to_shao():
        data.append(
            {
                "caption": caption,
                "image_path": image_path,
                "action_path": action_path,
                **FIXED_FIELDS,
            }
        )

    output = {"data": data}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Generated {len(data)} entries to {OUTPUT_PATH}")

    # Check all paths exist
    missing = []
    with open(OUTPUT_PATH) as f:
        loaded = json.load(f)
    for i, item in enumerate(loaded["data"]):
        for key in ("image_path", "action_path"):
            path = item.get(key)
            if path and not os.path.isfile(path):
                missing.append((i, key, path))
    if missing:
        print("Missing paths:")
        for idx, key, path in missing:
            print(f"  [{idx}] {key}: {path}")
    else:
        print("All paths exist.")


if __name__ == "__main__":
    main()
