#!/bin/bash

python fastvideo/sample/sample_t2v_mochi_no_sp.py \
    --model_path data/FastMochi-diffusers \
    --prompt_path "assets/prompt.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 8 \
    --guidance_scale 4.5 \
    --output_path outputs_video/mochi_no_sp/ \
    --shift 8 \
    --seed 12345 \
    --scheduler_type "pcm_linear_quadratic" 
