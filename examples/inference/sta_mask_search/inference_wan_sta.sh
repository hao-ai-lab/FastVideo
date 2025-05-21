#!/bin/bash

# export FASTVIDEO_ATTENTION_CONFIG=output/mask_search_strategy_1280x768/mask_strategy_s12.json
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers

height=768
width=1280
num_frames=69
output_path="inference_results/sta/mask_search_sparse"
base_port=29503
num_gpu=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
gpu_ids=$(seq 0 $((num_gpu-1)))

for i in $gpu_ids; do
    port=$((base_port+i))
    STA_mode="STA_searching"
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --height $height \
        --width $width \
        --num_frames $num_frames \
        --prompt_path ./assets/prompt_extend_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode &
    sleep 1
done
wait
echo "STA searching completed"

STA_mode="STA_tuning_cfg"
CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$base_port python examples/inference/sta_mask_search/wan_example.py \
    --height $height \
    --width $width \
    --num_frames $num_frames \
    --prompt_path ./assets/prompt_extend.txt \
    --output_path $output_path \
    --STA_mode $STA_mode
echo "STA tuning completed"

for i in $gpu_ids; do
    port=$((base_port+i))
    STA_mode="STA_inference"
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --height $height \
        --width $width \
        --num_frames $num_frames \
        --prompt_path ./assets/prompt_extend_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode &
    sleep 1
done
wait
echo "STA inference completed"
echo "All jobs completed"