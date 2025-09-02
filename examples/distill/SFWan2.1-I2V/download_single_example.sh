#!/bin/bash

# Download the full dataset first
python scripts/huggingface/download_hf.py --repo_id "wlsaidhi/crush-smol-merged" --local_dir "data/crush-smol" --repo_type "dataset"

# Create a single-example dataset for debugging
SINGLE_EXAMPLE_DIR="data/crush-smol-single"
mkdir -p "$SINGLE_EXAMPLE_DIR/videos"

# Copy the specific video that matches the validation.json style (macaron crushing)
cp "data/crush-smol/videos/7P02AihYkCU-Scene-005.mp4" "$SINGLE_EXAMPLE_DIR/videos/"

# Create a single-line videos.txt
echo "videos/7P02AihYkCU-Scene-005.mp4" > "$SINGLE_EXAMPLE_DIR/videos.txt"

# Create a single-line prompt.txt with the macaron crushing prompt
echo "PIKA_CRUSH A large metal press is shown compressing a pile of colorful macarons, flattening them as if they were under a hydraulic press. The press moves down, crushing the macarons into a pile of crumbs and squishing the colorful filling out." > "$SINGLE_EXAMPLE_DIR/prompt.txt"

# Generate the JSON file and merge.txt for the single example
python scripts/dataset_preparation/prepare_json_file.py --data_folder "$SINGLE_EXAMPLE_DIR" --output "videos2caption.json"

# Create a validation.json that uses the same example for consistency
cat > "$SINGLE_EXAMPLE_DIR/validation.json" << 'EOF'
{
    "data": [
      {
        "caption": "A large metal press is shown compressing a pile of colorful macarons, flattening them as if they were under a hydraulic press. The press moves down, crushing the macarons into a pile of crumbs and squishing the colorful filling out.",
        "image_path": null,
        "video_path": null,
        "num_inference_steps": 50,
        "height": 480,
        "width": 832,
        "num_frames": 81
      }
    ]
  }
EOF

echo "Single example dataset created at $SINGLE_EXAMPLE_DIR"
echo "Contains:"
echo "- 1 video: $(cat $SINGLE_EXAMPLE_DIR/videos.txt)"
echo "- 1 prompt: $(cat $SINGLE_EXAMPLE_DIR/prompt.txt)"
echo "- Validation file created with the same example for consistency" 