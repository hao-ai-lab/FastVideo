#!/bin/bash
#SBATCH --job-name=distill_dmd_t2v_1.3b
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=distill_dmd_t2v_1.3b_output/distill_dmd_t2v_1.3b_%j.out
#SBATCH --error=distill_dmd_t2v_1.3b_output/distill_dmd_t2v_1.3b_%j.err
#SBATCH --exclusive

# Basic Info
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export NCCL_DEBUG_SUBSYS=INIT,NET
# different cache dir for different processes
# export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_PROCID}
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

export HF_HOME=/mnt/data/huggingface
export HF_DATASETS_CACHE="/tmp/hf_datasets_${SLURM_JOB_ID}_${SLURM_PROCID}"
mkdir -p "${HF_DATASETS_CACHE}"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/shared-bin/local/usr/lib/aarch64-linux-gnu:$CUDA_HOME/lib64"
export CUDACXX="$CUDA_HOME/bin/nvcc"

source ~/miniconda3/bin/activate
conda activate fastvideo

# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=64

# Model paths for Self-Forcing DMD distillation:
GENERATOR_MODEL_PATH="wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
REAL_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-14B-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"  # Critic model

DATA_DIR="/mnt/data/vidprom_filtered_extended_umt5_text_embed"
# DATA_DIR="/mnt/data/mixkit_wan_1.3b_processed_t2v_distributed"
DATA_DIR_2="/mnt/data/mixkit_wan_1.3b_processed_t2v_distributed"
VALIDATION_DATASET_FILE="/mnt/home/zhouw.jerry2017/FastVideo/examples/distill/SFWan2.1-T2V/validation_64.json"
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

training_args=(
  --tracker_project_name SFwan_t2v_distill_self_forcing_dmd
  --wandb_run_name causal_forcing_vidprom_extended
  --output_dir /mnt/data/wan_cf_df_distill_1.3b
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 832
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frames 81
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
)

parallel_args=(
  --num_gpus $NUM_TOTAL_GPUS # 64
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1 # 64
  --hsdp_shard_dim $NUM_TOTAL_GPUS
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  # --data_path_2 "$DATA_DIR_2"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "4"
  --validation_guidance_scale "1.0"
)

optimizer_args=(
  --optimizer-type "adamw"
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

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
#   --init_weights_from_safetensors "/mnt/data/wan_tf_ode_init_3333/checkpoint-1600/transformer/diffusion_pytorch_model.safetensors"
#   --init_weights_from_safetensors "/mnt/data/wan_tf_init_3333/checkpoint-800/transformer/diffusion_pytorch_model.safetensors"
  # --init_weights_from_safetensors "/mnt/data/Causal-Forcing-wan1.3b/causal_ode.safetensors"
  # --init_weights_from_safetensors "/mnt/data/wan_ar_diffusion_3333_mixkit_shift5/checkpoint-7200/transformer/diffusion_pytorch_model.safetensors"
  --init_weights_from_safetensors "/mnt/data/wan_ar_diffusion_3333_mixkit_shift5/checkpoint-4800/transformer/diffusion_pytorch_model.safetensors"
  # --init_weights_from_safetensors "/mnt/data/wan_tf_ode_init_3333_mixkit_shift5/checkpoint-2000/transformer/diffusion_pytorch_model.safetensors"
  # --init_weights_from_safetensors "/mnt/data/wan_ar_df_diffusion_3333_mixkit_shift5/checkpoint-4800/transformer/diffusion_pytorch_model.safetensors"
)

dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 2e-6
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
  # --use-decoupled-dmd True
  # --use-gan-loss True
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
