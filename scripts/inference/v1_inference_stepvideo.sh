#!/bin/bash
# You better have two terminal, one for the remote server, and one for DiT
CUDA_VISIBLE_DEVICES=1 # python fastvideo/sample/v1_call_remote_server_stepvideo.py --model_dir data/stepvideo-t2v/ &
export FASTVIDEO_ATTENTION_BACKEND=
num_gpus=2
url='127.0.0.1'
model_dir=data/stepvideo-t2v
fastvideo generate \
    --model-path $model_dir \
    --sp-size ${num_gpus} \
    --tp-size ${num_gpus} \
    --num-gpus ${num_gpus} \
    --height 256 \
    --width 256 \
    --num-frames 29 \
    --num-inference-steps 50 \
    --embedded-cfg-scale 9.0 \
    --guidance-scale 9.0 \
    --prompt "A beautiful woman in a red dress walking down a street" \
    --seed 1024 \
    --output-path outputs_stepvideo/ \
    --flow-shift 13.0 \
    --vae-precision bf16