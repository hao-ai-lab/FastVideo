#!/bin/bash
export HOME=/mnt/weka/home/hao.zhang/wei
num_gpus=2
export FASTVIDEO_ATTENTION_BACKEND=
export MODEL_BASE=Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
# export MODEL_BASE=hunyuanvideo-community/HunyuanVideo
# Note that the tp_size and sp_size should be the same and equal to the number
# of GPUs. They are used for different parallel groups. sp_size is used for
# dit model and tp_size is used for encoder models.
fastvideo generate \
    --model-path $MODEL_BASE \
    --sp-size $num_gpus \
    --tp-size $num_gpus \
    --num-gpus $num_gpus \
    --height 480 \
    --width 832 \
    --num-frames 29 \
    --num-inference-steps 40 \
    --fps 16 \
    --flow-shift 3.0 \
    --guidance-scale 5.0 \
    --image-path "assets/dog.png" \
    --prompt "A brown dog is shaking its head while sitting on a light-colored sofa in a cozy room. Behind the dog, there is a framed painting on the shelf, surrounded by pink flowers. The soft, warm lighting in the room creates a comfortable atmosphere." \
    --negative-prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
    --seed 1024 \
    --output-path outputs_videox_i2v/