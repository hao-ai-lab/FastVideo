#!/bin/bash

num_gpus=4

torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompts "A cute fluffy panda eating Chinese food in a restaurant" \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --output_path outputs_video/debug \
    --seed 12345




num_gpus=4
torchrun --nnodes=1 --nproc_per_node=4 --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi/ \
    --prompt_embed_path data/Encoder_Overfit_Data/prompt_embed/0.pt \
    --encoder_attention_mask_path data/Encoder_Overfit_Data/prompt_attention_mask/0.pt \
    --num_frames 47 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --output_path outputs_video/overfit/debug_47_frames_lora \
    --seed 12345 \
    --lora_path data/outputs/BW_Testrun \
    

# 115 47