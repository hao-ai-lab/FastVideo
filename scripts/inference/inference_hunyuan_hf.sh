#!/bin/bash

num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ~/data/hunyuan_diffusers/ \
    --prompt_path "assets/prompt_test_3.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 50 \
    --output_path outputs_video/hunyuan_hf/ \
    --seed 1024 \

num_gpus=4
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ~/data/hunyuan_diffusers/ \
    --prompt_path "assets/prompt_test_3.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 50 \
    --output_path outputs_video/hunyuan_hf_new_4/ \
    --seed 1024 \
    --lora_checkpoint_dir data/outputs/HSH-Taylor-Finetune-Hunyuan_8e5_ra32_v40/lora-checkpoint-6000/

num_gpus=4
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29603 \
    fastvideo/sample/sample_t2v_hunyuan_hf.py \
    --model_path ~/data/hunyuan_diffusers/ \
    --prompt_path "assets/prompt_test_3.txt" \
    --num_frames 93 \
    --height 480 \
    --width 848 \
    --num_inference_steps 50 \
    --output_path outputs_video/hunyuan_hf_new_5/ \
    --seed 12345 \
    --lora_checkpoint_dir data/outputs/HSH-Taylor-Finetune-Hunyuan_1e4_ra64_v40/lora-checkpoint-5250/




