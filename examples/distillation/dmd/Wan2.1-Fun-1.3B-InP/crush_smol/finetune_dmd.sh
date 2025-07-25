#!/bin/bash
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
DATA_DIR=data/crush-smol_processed_i2v_1_3b_inp/combined_parquet_dataset/
VALIDATION_DIR=examples/training/finetune/wan_t2v_1_3b/crush_smol/validation.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false

# If you experience OOM, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/v1/training/wan_distillation_pipeline.py \
    --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
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
    --max_train_steps 30000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 500 \
    --validation_steps 50 \
    --validation_sampling_steps "3" \
    --log_validation \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --training_cfg_rate 0.0 \
    --output_dir "outputs_dmd_train/wan_finetune_1e5" \
    --tracker_project_name Wan_distillation \
    --wandb_run_name "crush_smol_dmd_test" \
    --num_height 448 \
    --num_width 832 \
    --num_frames 61 \
    --flow_shift 8 \
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
    --teacher_guidance_scale 3.5 \
    --enable_gradient_checkpointing_type "full" 