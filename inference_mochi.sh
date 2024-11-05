#!/bin/bash

num_gpus=2

torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompts "A hand enters the frame, pulling a sheet of plastic wrap over three balls of dough placed on a wooden surface. The plastic wrap is stretched to cover the dough more securely. The hand adjusts the wrap, ensuring that it is tight and smooth over the dough. The scene focuses on the hand's movements as it secures the edges of the plastic wrap. No new objects appear, and the camera remains stationary, focusing on the action of covering the dough." \
    --num_frames 79 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --transformer_path  data/outputs/debug_synthetic_uniform/checkpoint-200/transformer\
    --output_path outputs_video/overfit_reproduce/uniform_cfg_train_200_cfg_f79_syn \
    --seed 12345




num_gpus=2
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29503 \
    fastvideo/sample/sample_t2v_mochi.py \
    --model_path data/mochi \
    --prompt_embed_path data/Encoder_Overfit_Data/prompt_embed/0.pt \
    --encoder_attention_mask_path data/Encoder_Overfit_Data/prompt_attention_mask/0.pt \
    --num_frames 163 \
    --height 480 \
    --width 848 \
    --num_inference_steps 64 \
    --guidance_scale 4.5 \
    --transformer_path data/outputs/debug_uniform_28/checkpoint-100/transformer \
    --output_path outputs_video/overfit/uniform_cfg_train_100_cfg_f163_sp4 \
    --seed 12345

