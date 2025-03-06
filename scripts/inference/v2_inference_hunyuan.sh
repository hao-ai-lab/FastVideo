#!/bin/bash

export MODEL_BASE=data/FastHunyuan
python fastvideo/sample/v2_fastvideo_inference.py \
    --sp_size 4 \
    --height 720 \
    --width 1280 \
    --num_frames 125 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_video/hunyuan/vae_sp_v1_1/ \
    --model_path $MODEL_BASE \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-sp