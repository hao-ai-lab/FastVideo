#!/bin/bash

num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=
# For longcat, we must first convert the official weights to FastVideo native format
# conversion method: python scripts/checkpoint_conversion/longcat_to_fastvideo.py
# --source /path/to/LongCat-Video/weights/LongCat-Video
# --output weights/longcat-native
export MODEL_BASE=weights/longcat-native

fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --dit-cpu-offload False \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload False \
    --pin-cpu-memory False \
    --enable-bsa False \
    --lora-path "$MODEL_BASE/lora/distilled" \
    --lora-nickname "distilled" \
    --height 480 \
    --width 832 \
    --num-frames 93 \
    --num-inference-steps 16 \
    --fps 15 \
    --guidance-scale 1.0 \
    --prompt "In a realistic photography style, an asian boy around seven or eight years old sits on a park bench, wearing a light yellow T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene." \
    --seed 42 \
    --output-path outputs_video/longcat_distill
