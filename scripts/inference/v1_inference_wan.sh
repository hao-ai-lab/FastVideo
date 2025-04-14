#!/bin/bash

num_gpus=2
export FASTVIDEO_ATTENTION_BACKEND=
export MODEL_BASE=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# Note that the tp_size and sp_size should be the same and equal to the number
# of GPUs. They are used for different parallel groups. sp_size is used for
# dit model and tp_size is used for encoder models.
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/v1/sample/v1_fastvideo_inference.py \
    --sp_size $num_gpus \
    --tp_size $num_gpus \
    --height 480 \
    --width 832 \
    --num_frames 77 \
    --num_inference_steps 50 \
    --fps 16 \
    --guidance_scale 3.0 \
    --prompt_path ./assets/prompt.txt \
    --neg_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output_path outputs_video/ \
    --model_path $MODEL_BASE \
    --vae-sp \
    --text-encoder-precision "fp32" \
    --use-cpu-offload