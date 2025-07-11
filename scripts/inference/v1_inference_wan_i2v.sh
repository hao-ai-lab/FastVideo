#!/bin/bash

num_gpus=2
export FASTVIDEO_ATTENTION_BACKEND=
export MODEL_BASE=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size $num_gpus \
    --num-gpus $num_gpus \
    --height 480 \
    --width 832 \
    --num-frames 77 \
    --num-inference-steps 40 \
    --fps 16 \
    --flow-shift 3.0 \
    --guidance-scale 5.0 \
    --image-path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg" \
    --prompt "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_i2v/