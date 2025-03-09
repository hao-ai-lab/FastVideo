#!/bin/bash

num_gpus=8
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503 \
    tp_example.py

    
num_gpus=2
torchrun --standalone --nnodes=4 --nproc_per_node=$num_gpus \
    --master_port 29503 \
    test_hunyuanvideo.py --sequence_model_parallel_size $num_gpus



num_gpus=1
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    --master_port 29503  \
    test_hunyuanvideo_load.py --sequence_model_parallel_size $num_gpus 