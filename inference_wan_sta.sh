#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export MASTER_PORT=29506
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=/home/bcds/model/Wan-AI/Wan2.1-T2V-14B-Diffusers

height=768
width=1280
num_frames=69
prompt_path="assets/prompt.txt"
output_path="inference_results/sta/mask_search"
STA_mode="STA_searching"
num_gpus=1

python example_wan.py \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --prompt_path $prompt_path \
    --output_path $output_path \
    --STA_mode $STA_mode \
    --num_gpus $num_gpus

STA_mode="STA_tuning_cfg"
python example_wan.py \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --prompt_path $prompt_path \
    --output_path $output_path \
    --STA_mode $STA_mode \
    --num_gpus $num_gpus

