#!/bin/bash
# You better have two terminal, one for the remote server, and one for DiT
CUDA_VISIBLE_DEVICES=1 # python fastvideo/sample/v1_call_remote_server_stepvideo.py --model_dir data/stepvideo-t2v/ &
export FASTVIDEO_ATTENTION_BACKEND=
num_gpus=2
url='127.0.0.1'
model_dir=data/stepvideo-t2v
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/v1/sample/v1_fastvideo_inference.py \
    --sp_size ${num_gpus} \
    --tp_size ${num_gpus} \
    --height 256 \
    --width 256 \
    --num_frames 29 \
    --num_inference_steps 50 \
    --embedded_cfg_scale 9.0 \
    --guidance_scale 9.0 \
    --prompt_path ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_stepvideo/ \
    --model_path $model_dir \
    --flow_shift 13.0 \
    --vae_precision bf16