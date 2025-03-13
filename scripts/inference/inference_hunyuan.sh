#!/bin/bash

# num_gpus=4
# export MODEL_BASE=data/FastHunyuan
# torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
#     fastvideo/sample/sample_t2v_hunyuan.py \
#     --height 720 \
#     --width 1280 \
#     --num_frames 125 \
#     --num_inference_steps 6 \
#     --guidance_scale 1 \
#     --embedded_cfg_scale 6 \
#     --flow_shift 17 \
#     --flow-reverse \
#     --prompt ./assets/prompt.txt \
#     --seed 1024 \
#     --output_path outputs_video/hunyuan/vae_sp/ \
#     --model_path $MODEL_BASE \
#     --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
#     --vae-sp

# Inference original Hunyuan
num_gpus=1
export MODEL_BASE=data/hunyuan
mask_strategy_file_path=assets/test_mask_strategy_hunyuan.json
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 768 \
    --width 1280 \
    --num_frames 45 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_video/hunyuan/flex_attention_235/ \
    --model_path $MODEL_BASE \
    --mask_strategy_file_path $mask_strategy_file_path \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt \
    --vae-sp \
    --use-fp8 \
    --use-cpu-offload \
    --vae_tiling \
    --enable_teacache \
    --rel_l1_thresh 0.15 \
