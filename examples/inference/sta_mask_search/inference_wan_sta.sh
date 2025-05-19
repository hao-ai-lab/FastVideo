#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export MASTER_PORT=29506
# export FASTVIDEO_ATTENTION_CONFIG=output/mask_search_strategy_1280x768/mask_strategy_s12.json
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers

height=768
width=1280
num_frames=69
prompt_path="assets/prompt.txt"
output_path="inference_results/sta/mask_search_sparse"
num_gpus=1

# STA_mode="STA_searching"
# python examples/inference/sta_mask_search/wan_example.py \
#     --height $height \
#     --width $width \
#     --num_frames $num_frames \
#     --prompt_path $prompt_path \
#     --output_path $output_path \
#     --STA_mode $STA_mode \
#     --num_gpus $num_gpus

# STA_mode="STA_tuning_cfg"
# python examples/inference/sta_mask_search/wan_example.py \
#     --height $height \
#     --width $width \
#     --num_frames $num_frames \
#     --prompt_path $prompt_path \
#     --output_path $output_path \
#     --STA_mode $STA_mode \
#     --num_gpus $num_gpus

STA_mode="STA_inference"
python examples/inference/sta_mask_search/wan_example.py \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --prompt_path $prompt_path \
    --output_path $output_path \
    --STA_mode $STA_mode \
    --num_gpus $num_gpus
