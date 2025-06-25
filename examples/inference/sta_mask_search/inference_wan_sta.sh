#!/bin/bash

export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers

base_port=29503
num_gpu=1
gpu_ids=$(seq 0 $((num_gpu-1)))
skip_time_steps=12

output_path="inference_results/sta/mask_search_full"
STA_mode="STA_searching"
for i in $gpu_ids; do
    port=$((base_port+i))
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --prompt_path ./assets/prompt_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode &
    sleep 1
done
wait
echo "STA searching completed"

output_path="inference_results/sta/mask_search_sparse"
STA_mode="STA_tuning"
for i in $gpu_ids; do
    port=$((base_port+i))
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --prompt_path ./assets/prompt_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode \
        --skip_time_steps $skip_time_steps &
    sleep 1
done
wait
echo "STA tuning completed"

echo "All jobs completed"