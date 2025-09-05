export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY='50632ebd88ffd970521cec9ab4a1a2d7e85bfc45'
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export TRITON_CACHE_DIR=/tmp/triton_cache
# DATA_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn-upload/latents_i2v/train/
DATA_DIR=/mnt/weka/home/hao.zhang/wei/FastVideo/data/crush-smol_processed_t2v/combined_parquet_dataset
# VALIDATION_DIR=/mnt/sharefs/users/hao.zhang/Vchitect-2M/mixkit/validation_8.json
VALIDATION_DIR=/mnt/weka/home/hao.zhang/wei/FastVideo/data/crush-smol-single_processed_t2v/validation.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]
CHECKPOINT_PATH="outputs_train_test/wan_finetune/checkpoint-10"
# If you do not have 32 GPUs and to fit in memory, you can: 1. increase sp_size. 2. reduce num_latent_t
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_training_pipeline.py \
    --model_path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --inference_mode False\
    --pretrained_model_name_or_path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --cache_dir "/home/ray/.cache" \
    --data_path "$DATA_DIR" \
    --validation_dataset_file  "$VALIDATION_DIR" \
    --train_batch_size 1 \
    --num_latent_t 16\
    --sp_size 4 \
    --tp_size 1 \
    --num_gpus $NUM_GPUS \
    --hsdp_replicate_dim 1 \
    --hsdp-shard-dim 8 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 30000 \
    --learning_rate 1e-5 \
    --mixed_precision "bf16" \
    --checkpointing_steps 1000 \
    --validation_steps 30 \
    --validation_sampling_steps "40" \
    --checkpoints_total_limit 3 \
    --ema_start_step 0 \
    --training_cfg_rate 0.1 \
    --seed 1024 \
    --output_dir "outputs_train_test/wan_finetune_v1" \
    --tracker_project_name VSA_finetune \
    --num_height 448 \
    --num_width 832 \
    --num_frames 61 \
    --flow_shift 5 \
    --validation_guidance_scale "5.0" \
    --num_euler_timesteps 50 \
    --master_weight_type "fp32" \
    --dit_precision "fp32" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0