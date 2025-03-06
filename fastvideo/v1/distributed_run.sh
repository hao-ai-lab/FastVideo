#!/bin/bash

num_gpus=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503 \
    tp_example.py

    
num_gpus=2
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503 \
    test_hunyuanvideo.py