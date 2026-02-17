

# Basic info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_$$"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NODE_RANK=0
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export WANDB_API_KEY=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

# Configs
NUM_GPUS=4
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=data/Wan-Syn_77x448x832_600k
VALIDATION_DATASET_FILE=examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json

# Core training arguments
core_training_args=(
  --tracker_project_name wan_t2v_VSA
  --output_dir "checkpoints/wan_t2v_finetune_VSA"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --enable_gradient_checkpointing_type "full" # if OOM enable this
)

# Validation generation shape arguments (used during training validation)
validation_generation_shape_args=(
  --num_height 448
  --num_width 832
  --num_frames 77
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_GPUS
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim $NUM_GPUS
  --hsdp_shard_dim 1
)

# Model arguments
model_args=(
  --model_path "$MODEL_PATH"
  --pretrained_model_name_or_path "$MODEL_PATH"
)

# Dataset arguments
dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "5.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 1e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 1000
  --training_state_checkpointing_steps 1000
  --weight_decay 0.01
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.1
  --dit_precision "fp32"
  --ema_start_step 0
  --flow_shift 1
  --seed 1000
)

# VSA arguments
vsa_args=(
  --VSA_decay_rate 0.03
  --VSA_decay_interval_steps 50
  --VSA_sparsity 0.9
)

torchrun \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  --node_rank "$NODE_RANK" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  fastvideo/training/wan_training_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${core_training_args[@]}" \
  "${validation_generation_shape_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}" \
  "${vsa_args[@]}"
