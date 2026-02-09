#!/bin/bash
#SBATCH --job-name=moe_8n_sf_distill
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=moe_8n_sf_distill_output/moe_sf_distill_%j.out
#SBATCH --error=moe_8n_sf_distill_output/moe_sf_distill_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false
# export WANDB_API_KEY="8d9f4b39abd68eb4e29f6fc010b7ee71a2207cde"
export WANDB_API_KEY="50632ebd88ffd970521cec9ab4a1a2d7e85bfc45"
# export WANDB_API_KEY='your_wandb_api_key_here'
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

source ~/conda/miniconda/bin/activate
conda activate wei-fv
export HOME="/mnt/weka/home/hao.zhang/wei"

# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=64

# Model paths for Self-Forcing DMD distillation with Wan2.2:
# GENERATOR_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
GENERATOR_MODEL_PATH="rand0nmr/SFWan2.2-T2V-A14B-Diffusers"  # Updated to Wan2.2
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers" # Critic model
# GENERATOR_MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers" # Updated to Wan2.2
# REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Teacher model
# FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" # Critic model

# DATA_DIR="data/test-text-preprocessing/Node_0_GPU_1_File_1/combined_parquet_dataset/"
# DATA_DIR="/mnt/weka/home/hao.zhang/matthew/FastVideo/data/test-text-preprocessing"
DATA_DIR="/mnt/weka/home/hao.zhang/wl/sharefs/wl/release/vidprom_16k_text_embed"
# DATA_DIR=data/crush-smol_processed_t2v/combined_parquet_dataset
# DATA_DIR="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn-upload/latents_i2v/train/"
# VALIDATION_DATASET_FILE="data/crush-smol-single_processed_t2v/validation.json"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_64.json"

# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

training_args=(
  --tracker_project_name SFwan2.2_t2v_distill_self_forcing_dmd  # Updated for Wan2.2
  # --output_dir "/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_x0_cache_detach_df3000_cfg2_31_split"
  --output_dir "/mnt/weka/home/hao.zhang/SFwan2.2_x0odeinit_x0CacheAppend_noaddnoisehigh_22cfg_moecritic_4steps"
  # --output_dir "/mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_x0_CacheAppend_df3000_cfg2"
  --wandb_run_name "x0odeinit_x0CacheAppend_noaddnoisehigh_22cfg_moecritic_4steps"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480  # Updated to match Wan2.2 config
  --num_width 832   # Updated to match Wan2.2 config
  --enable_gradient_checkpointing_type "full"
  --simulate_generator_forward
  --log_visualization
  --num_frames 81
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
  # --init_weights_from_safetensors /mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_inference_transformer
  # --init_weights_from_safetensors_2 /mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-246_weight_only/generator_2_inference_transformer
  # --init_weights_from_safetensors /mnt/weka/home/hao.zhang/wl/sharefs/wl/release/wan2.2-moe-ode-init/official_ode_init_wan22_high_ckpt2000_sharded_new
  # --init_weights_from_safetensors_2 /mnt/weka/home/hao.zhang/wl/sharefs/wl/release/wan2.2-moe-ode-init/official_ode_init_wan22_low_ckpt2000_sharded_new
  --init_weights_from_safetensors /mnt/weka/home/hao.zhang/official_ode_init_wan22_high_xbx0_4steps/checkpoint_model_002000/model.safetensors
  --init_weights_from_safetensors_2 /mnt/weka/home/hao.zhang/official_ode_init_wan22_low_4steps/checkpoint_model_002000/model.safetensors
  # --init_weights_from_safetensors /mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-736_weight_only/generator_inference_transformer
  # --init_weights_from_safetensors_2 /mnt/sharefs/users/hao.zhang/wei/SFwan2.2_distill_self_forcing_release_cfg2/checkpoint-736_weight_only/generator_2_inference_transformer
  # --init_weights_from_safetensors /mnt/weka/home/hao.zhang/wl/Self-Forcing/diffusers_ode_init/model.safetensors
  # --init_weights_from_safetensors "/mnt/sharefs/users/hao.zhang/wei/diffusion_init_4000_14b_low_dense_shift5/model.safetensors"
)

parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 10
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0" # not used for dmd inference
  --vae_cpu_offload True
)

optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 49
  --weight_only_checkpointing_steps 99999
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 10.0
)

miscellaneous_args=(
  # --use_add_noise_high
  # --override_boundary_ratio 0.866
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 12
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 5
  # --resume_from_checkpoint "/mnt/weka/home/hao.zhang/SFwan2.2_x0odeinit_x0CacheAppend_noaddnoisehigh_22cfg_moecritic/checkpoint-441"
)

dmd_args=(
  # --dmd_denoising_steps '1000,850,800,750,570,450,350,250'
  # --dmd_denoising_steps '1000,850,700,550,350,275,200,125'
  --dmd_denoising_steps '1000,550,350,125'
  # --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.04
  --max_timestep_ratio 0.96
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 2.0
  --real_score_guidance_scale_2 2.0
  --fake_score_learning_rate 4e-7
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
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