#!/bin/bash

# 480P 1.3B runnable on 4090
num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export MODEL_BASE=FastVideo/FastWan2.1-T2V-1.3B-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 3 \
    --dit-cpu-offload False \
    --vae-cpu-offload False \
    --text-encoder-cpu-offload True \
    --pin-cpu-memory False \
    --fps 16 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_dmd_1.3B/ \
    --VSA-sparsity 0.8 \
    --dmd-denoising-steps "1000,757,522" \
    --enable_torch_compile 



# 480P 14B
num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export MODEL_BASE=FastVideo/FastWan2.1-T2V-14B-480P-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 3 \
    --fps 16 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_dmd_14B/ \
    --VSA-sparsity 0.9 \
    --dmd-denoising-steps "1000,757,522" \
    --enable_torch_compile 



# 720P 14B
num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export MODEL_BASE=FastVideo/FastWan2.1-T2V-14B-480P-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --num-inference-steps 3 \
    --fps 16 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_dmd_14B_720P/ \
    --VSA-sparsity 0.9 \
    --dmd-denoising-steps "1000,757,522" \
    --enable_torch_compile 

# 720P 5B
num_gpus=1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export MODEL_BASE=FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# You can either use --prompt or --prompt-txt, but not both.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size 1 \
    --num-gpus $num_gpus \
    --height 704 \
    --width 1280 \
    --num-frames 104 \
    --num-inference-steps 3 \
    --fps 24 \
    --prompt-txt assets/prompt.txt \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_video_dmd_5B_720P/ \
    --dmd-denoising-steps "1000,757,522" \
    --enable_torch_compile 