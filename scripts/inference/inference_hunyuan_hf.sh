#!/bin/bash

num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ~/data/hunyuan_diffusers/ \
    --prompt_path "assets/prompt.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 50 \
    --guidance_scale 6 \
    --output_path outputs_video/hunyuan_hf_test/ \
    --seed 1024 \




