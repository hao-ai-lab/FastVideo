#!/bin/bash

# Inference with STA + Teacache
num_gpus=1
mask_strategy_file_path=assets/mask_strategy_hunyuan.json
export MODEL_BASE=data/hunyuan
rel_l1_thresh=0.15
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29603 \
    fastvideo/sample/sample_t2v_hunyuan_STA.py \
    --height 768 \
    --width 1280 \
    --num_frames 117 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 12345 \
    --output_path outputs_video/hunyuan_STA/ \
    --model_path $MODEL_BASE \
    --mask_strategy_file_path $mask_strategy_file_path \
    --rel_l1_thresh $rel_l1_thresh \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-sp \
    --enable_teacache 

# Inference with STA only
num_gpus=1
mask_strategy_file_path=assets/mask_strategy.json
export MODEL_BASE=data/hunyuan
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29603 \
    fastvideo/sample/sample_t2v_hunyuan_STA.py \
    --height 768 \
    --width 1280 \
    --num_frames 117 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 12345 \
    --output_path outputs_video/hunyuan_STA/ \
    --model_path $MODEL_BASE \
    --mask_strategy_file_path $mask_strategy_file_path \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-sp \
    --enable_torch_compile