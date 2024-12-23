#!/bin/bash

num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ~/hunyuan_hf/ \
    --prompt_path "assets/prompt_test.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 50 \
    --guidance_scale 1.5 \
    --output_path outputs_video/hunyuan_hf/ \
    --seed 1024 \
    --linear_threshold 0.1 \
    --flow_shift 17 \
    --flow-reverse \
    --linear_range 0.75 \



