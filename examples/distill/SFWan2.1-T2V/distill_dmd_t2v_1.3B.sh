# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export WANDB_API_KEY='8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde'

# Configs
NUM_GPUS=8

# Model paths for DMD distillation:
GENERATOR_MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Critic model

DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/SFWan2.1-I2V/validation_better.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name SFwan_t2v_distill_self_forcing_dmd
  --output_dir "/mnt/sharefs/users/hao.zhang/wl/sf_checkpoints/ode0_SFwan_t2v_finetune_sf_${lr}_c${critic_lr}"
  --wandb_run_name "DEBUG${lr}_c${critic_lr}"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81
  --warp_denoising_step
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frame_per_block 3
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
)

# Parallel arguments
parallel_args=(
  --num_gpus 8 # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim 8
)

# Model arguments
model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --generator_model_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
#   --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "3"
  --validation_guidance_scale "6.0" # not used for dmd inference
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 50
  --weight_only_checkpointing_steps 50
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 100
  --init_weights_from_safetensors "/mnt/weka/home/hao.zhang/wl/Self-Forcing/diffusers_ode_init/model.safetensors"
)

# Self-forcing DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
)

# Self-forcing specific arguments
self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks False  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
  --validate_cache_structure False  # Set to True for debugging KV cache issues
)

torchrun \
--nnodes 1 \
--nproc_per_node $NUM_GPUS \
--master_port $MASTER_PORT \
    fastvideo/training/wan_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}"