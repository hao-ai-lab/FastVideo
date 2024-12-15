#!/bin/bash

num_gpus=4

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 480 \
    --width 848 \
    --num_frames 93 \
    --num_inference_steps 4 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt_path "data/prompt.txt" \
    --seed 12345 \
    --output_path outputs_video/hunyuan_sp/
