#!/bin/bash

num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/FastMochi-diffusers \
    --prompt_path "assets/prompt.txt" \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 1.5 \
    --output_path outputs_video/mochi_sp/ \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" \
    --linear_threshold 0.1 \
    --linear_range 0.75

