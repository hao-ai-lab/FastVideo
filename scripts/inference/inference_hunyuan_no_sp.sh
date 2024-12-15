#!/bin/bash

python fastvideo/sample/sample_t2v_hunyuan_no_sp.py \
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
    --output_path outputs_video/hunyuan_no_sp/
