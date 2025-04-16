#!/bin/bash
# You better have two terminal, one for the remote server, and one for DiT
CUDA_VISIBLE_DEVICES=0 #python fastvideo/sample/call_remote_server_stepvideo.py --model_dir data/stepvideo-t2v/ &
num_gpus=1
url='127.0.0.1'
model_dir=data/stepvideo-t2v
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/v1/sample/v1_fastvideo_inference.py \
    --sp_size 1 \
    --tp_size 1 \
    --height 256 \
    --width 512 \
    --num_frames 21 \
    --num_inference_steps 6 \
    --embedded_cfg_scale 9.0 \
    --prompt_path ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_stepvideo/ \
    --model_path $model_dir \
    --time_shift 13.0 