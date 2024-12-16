#!/bin/bash

num_gpus=[Your GPU Count]

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 720 \
    --width 1280 \
    --num_frames 125 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 12345 \
    --output_path outputs_video/hunyuan/ \
    --model_path data/FastHunyuan \
    --dit-weight data/FastHunyuan/hunyuan-video-t2v-720p/transformers/diffusion_pytorch_model.safetensors