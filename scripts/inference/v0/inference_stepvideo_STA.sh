#!/bin/bash
# Inference with STA + Teacache
# You better have two terminal, one for the remote server, and one for DiT
# remove enable_teacache to disable Teacache
CUDA_VISIBLE_DEVICES=7 python fastvideo/sample/call_remote_server_stepvideo.py --model_dir data/stepvideo-t2v/ &
parallel=4
url='127.0.0.1'
model_dir=data/stepvideo-t2v
mask_strategy_file_path=assets/mask_strategy_stepvideo.json
rel_l1_thresh=0.23
torchrun --nproc_per_node $parallel fastvideo/sample/sample_t2v_stepvideo_STA.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url  \
    --prompt "A rocket blasting off from the launch pad, accelerating rapidly into the sky." \
    --infer_steps 50  \
    --width 768 \
    --height 768 \
    --num_frames 204 \
    --cfg_scale 9.0 \
    --save_path outputs/ \
    --time_shift 13.0 \
    --rel_l1_thresh $rel_l1_thresh \
    --enable_teacache \
    --mask_strategy_file_path $mask_strategy_file_path
