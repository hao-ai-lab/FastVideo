#!/bin/bash

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY='73190d8c0de18a14eb3444e222f9432d247d1e30'
export WANDB_API_KEY='8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
DATA_DIR=data/crush-smol_processed_i2v_1_3b_inp/combined_parquet_dataset/
# DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn/latents_i2v/train/
VALIDATION_DIR=examples/training/finetune/wan_i2v_14b_480p/crush_smol/validation.json
# VALIDATION_DIR=/mnt/weka/home/hao.zhang/wl/FastVideo/data/mixkit/validation.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export MASTER_PORT=29501
export TOKENIZERS_PARALLELISM=false

# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/v1/training/wan_i2v_distillation_pipeline.py \
    --model_path weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers  \
    --inference_mode False\
    --pretrained_model_name_or_path weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers  \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 8 \
    --sp_size 1 \
    --tp_size 1 \
    --num_gpus 8 \
    --hsdp_replicate_dim 8 \
    --hsdp-shard-dim 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 6000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 1000 \
    --validation_steps 50 \
    --validation_sampling_steps "50" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd_train_i2v/wan_i2v_finetune_9e6" \
    --tracker_project_name Wan_distillation \
    --wandb_run_name "temp_consistency" \
    --i2v_frame_weighting \
    --i2v_weighting_scheme "first_frame_only" \
    --i2v_temporal_scale_factor 1.0 \
    --i2v_first_frame_weight 0.01 \
    --num_height 480 \
    --num_width 832 \
    --num_frames 61 \
    --flow_shift 3 \
    --validation_guidance_scale "1.0" \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --vae_precision "bf16" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --student_critic_update_ratio 5 \
    --denoising_step_list '1000,757,522' \
    --min_step_ratio 0.02 \
    --max_step_ratio 0.98 \
    --seed 1000 \
    --teacher_guidance_scale 3.5 
