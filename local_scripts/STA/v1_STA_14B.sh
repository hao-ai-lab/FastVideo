#!/bin/bash

num_gpus=1
export FASTVIDEO_ATTENTION_CONFIG=assets/mask_strategy_wan.json
export FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN
export MODEL_BASE=Wan-AI/Wan2.1-T2V-14B-Diffusers
# Note that the tp_size and sp_size should be the same and equal to the number
# of GPUs. They are used for different parallel groups. sp_size is used for
# dit model and tp_size is used for encoder models.
gpu_ids=(0 1 2 3 4 5 6 7)
for i in {1..8}; do
    gpu=${gpu_ids[((i-1))]}
    prompt_file="./assets/prompt.txt"
    # Read the i-th line from the prompt file
    prompt=$(sed -n "${i}p" "$prompt_file")
    fastvideo generate \
        --model-path $MODEL_BASE \
        --sp-size $num_gpus \
        --tp-size $num_gpus \
        --num-gpus $num_gpus \
        --height 768 \
        --width 1280 \
        --num-frames 69 \
        --num-inference-steps 50 \
        --fps 16 \
        --guidance-scale 5.0 \
        --prompt "$prompt" \
        --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
        --seed 1024 \
        --output-path outputs_video/STA_1024_SSIM/ &
done

# Wait for first block to complete
wait
