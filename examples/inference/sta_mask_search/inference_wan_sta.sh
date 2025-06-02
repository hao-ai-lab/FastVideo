#!/bin/bash

export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers

output_path="inference_results/sta/mask_search_sparse"
base_port=29503
num_gpu=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
gpu_ids=$(seq 0 $((num_gpu-1)))
skip_time_steps=12

for i in $gpu_ids; do
    port=$((base_port+i))
    STA_mode="STA_searching"
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --prompt_path ./assets/prompt_extend_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode &
    sleep 1
done
wait
echo "STA searching completed"

STA_mode="STA_tuning"
MASTER_PORT=$base_port python examples/inference/sta_mask_search/wan_example.py \
    --prompt_path ./assets/prompt_extend.txt \
    --output_path $output_path \
    --STA_mode $STA_mode \
    --skip_time_steps $skip_time_steps
echo "STA tuning completed"

# switch to new mask strategy config
for i in $gpu_ids; do
    port=$((base_port+i))
    STA_mode="STA_inference"
    CUDA_VISIBLE_DEVICES=$i MASTER_PORT=$port python examples/inference/sta_mask_search/wan_example.py \
        --prompt_path ./assets/prompt_extend_${i}.txt \
        --output_path $output_path \
        --STA_mode $STA_mode &
    sleep 1
done
wait
echo "STA inference completed"
echo "All jobs completed"