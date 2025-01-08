#!/bin/bash

num_gpus=1
export MODEL_BASE="data/FastHunyuan-diffusers"
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 12345 \
    fastvideo/sample/sample_t2v_diffusers_hunyuan.py \
    --height 720 \
    --width 1280 \
    --num_frames 45 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_video/hunyuan_quant/nf4/ \
    --model_path $MODEL_BASE \
    --quantization "nf4" 