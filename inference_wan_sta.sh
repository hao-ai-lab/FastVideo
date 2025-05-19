#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29506
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=/home/bcds/model/Wan-AI/Wan2.1-T2V-14B-Diffusers

height=768
width=1280
num_frames=69
prompt="A man is running."
output_path="inference_results/sta"

python example_wan.py \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --prompt "$prompt" \
    --output_path $output_path

