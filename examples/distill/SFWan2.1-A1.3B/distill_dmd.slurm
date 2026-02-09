#!/bin/bash
#SBATCH --job-name=1.3B_high_noise_moe_8n_sf_distill
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=1.3B_moe_8n_sf_distill_output/moe_sf_distill_%j.out
#SBATCH --error=1.3B_moe_8n_sf_distill_output/moe_sf_distill_%j.err
#SBATCH --exclusive
set -e -x


# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate wei-fv
export HOME="/mnt/weka/home/hao.zhang/wei"

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=64

# Model paths for Self-Forcing DMD distillation:
GENERATOR_MODEL_PATH="rand0nmr/SFWan2.1-T2V-A1.3B-Diffusers"
# REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"  # Teacher model
REAL_SCORE_MODEL_PATH="/mnt/sharefs/users/hao.zhang/wei/wan_models/Wan2.1-T2V-14B-Diffusers"
FAKE_SCORE_MODEL_PATH="/mnt/sharefs/users/hao.zhang/wei/wan_models/Wan2.1-T2V-A1.3B-Diffusers"

# DATA_DIR="data/test-text-preprocessing/Node_0_GPU_1_File_1/combined_parquet_dataset/"
# DATA_DIR="/mnt/weka/home/hao.zhang/matthew/FastVideo/data/test-text-preprocessing"
DATA_DIR="/mnt/sharefs/users/hao.zhang/wl/release/vidprom_16k_text_embed"
# DATA_DIR="data/crush-smol_processed_t2v/combined_parquet_dataset/"
# VALIDATION_DATASET_FILE="data/crush-smol-single_processed_t2v/validation.json"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_64.json"
# VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_2.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Training arguments
training_args=(
  --tracker_project_name SFwan_t2v_distill_self_forcing_dmd  # Updated for self-forcing DMD
  --output_dir "/mnt/sharefs/users/hao.zhang/wei/1.3B_moe_x0_CacheAppend_3b_4c_high_x0_ode_init_addnoisehigh_moecritic"
  --wandb_run_name "1.3B_moe_8n_sf_distill_x0_CacheAppend_3b_4c_high_x0_ode_init_addnoisehigh"
  # --use_sf_wan
  # --sf_ode_init_path "checkpoints/ode_init.pt"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --num_frames 81  # Must be divisible by num_frame_per_block (81 % 3 = 0 ✓)
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
)

# Parallel arguments
parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_TOTAL_GPUS
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
  --dataloader_num_workers 8
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0" # not used for dmd inference
  --vae_cpu_offload True
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --fake_score_learning_rate 4e-7
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 200
  --weight_only_checkpointing_steps 200
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 10.0
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
  --ema_start_step 200
  # --init_weights_from_safetensors "/mnt/weka/home/hao.zhang/wei/FastVideo/diffusers_1.3B_SF/model.safetensors"
  # --init_weights_from_safetensors "/mnt/weka/home/hao.zhang/wl/Self-Forcing/diffusers_ode_init/model.saftensors"
  --init_weights_from_safetensors "/mnt/sharefs/users/hao.zhang/wl/release/official_ode_init_8n_high_1_3_x0/checkpoint_model_002000/model.safetensors"
  --init_weights_from_safetensors_2 "/mnt/sharefs/users/hao.zhang/wl/release/official_ode_init_8n_low_1_3/checkpoint_model_002000/model.safetensors"
  # --init_weights_from_safetensors "/mnt/weka/home/hao.zhang/wl/FastVideo2/vidprom_8b16k_1e-5_gn1/checkpoint-3500/transformer/diffusion_pytorch_model.safetensors"
#   --init_weights_from_safetensors "/mnt/weka/home/hao.zhang/wl/sf2/checkpoint_v3_7k/model.safetensors"
  # --resume_from_checkpoint "/mnt/sharefs/users/hao.zhang/wei/1.3B_MoE_SFwan_t2v_finetune/checkpoint-1990"
)

# Self-forcing DMD arguments
dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.04
  --max_timestep_ratio 0.96
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  # --real_score_guidance_scale_2 3.0
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
)

# Self-forcing specific arguments
self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
  --validate_cache_structure False  # Set to True for debugging KV cache issues
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
--nproc_per_node $NUM_GPUS \
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