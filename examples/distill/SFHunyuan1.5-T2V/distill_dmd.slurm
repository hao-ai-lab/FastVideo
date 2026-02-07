#!/bin/bash
#SBATCH --job-name=dmd_3333
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=dmd_3333_output/dmd_3333_%j.out
#SBATCH --error=dmd_3333_output/dmd_3333_%j.err
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
conda activate wei-fv-distill
export HOME="/mnt/weka/home/hao.zhang/wei"

# Configs
NUM_GPUS=8
NUM_TOTAL_GPUS=32

# Model paths for Self-Forcing DMD distillation with Hunyuan1.5:
GENERATOR_MODEL_PATH="weizhou03/SFHunyuanVideo-1.5-Diffusers-480p_t2v"  # Updated to Wan2.2
REAL_SCORE_MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v" # Teacher model
FAKE_SCORE_MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
# REAL_SCORE_MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"
# FAKE_SCORE_MODEL_PATH="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled" # Critic model

# DATA_DIR="data/test-text-preprocessing/Node_0_GPU_1_File_1/combined_parquet_dataset/"
# DATA_DIR="/mnt/weka/home/hao.zhang/matthew/FastVideo/data/test-text-preprocessing"
DATA_DIR="/mnt/weka/home/hao.zhang/wl/sharefs/wl/release/vidprom_16k_text_embed"
DATA_DIR_2="/mnt/weka/home/hao.zhang/wl/sharefs/wl/release/fv-ode-preprocessing-16k-hy15-121"
# DATA_DIR=data/crush-smol_processed_t2v/combined_parquet_dataset
# DATA_DIR="/mnt/sharefs/users/hao.zhang/Vchitect-2M/Wan-Syn-upload/latents_i2v/train/"
# VALIDATION_DATASET_FILE="data/crush-smol-single_processed_t2v/validation.json"
VALIDATION_DATASET_FILE="/mnt/weka/home/hao.zhang/wl/FastVideo/examples/distill/Wan2.2-TI2V-5B-Diffusers/Data-free/validation_64.json"

# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

training_args=(
  --tracker_project_name SFhy1.5_t2v_distill_self_forcing_dmd  # Updated for Wan2.2
  --output_dir "/mnt/weka/home/hao.zhang/wei/SFhy1.5_distill_3333_1e-5_1e-5_cfg6_corrected_scheduler"
  --wandb_run_name "self_forcing_3333_context_forcing"
  --max_train_steps 4000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 480
  --num_width 848
  --enable_gradient_checkpointing_type "full"
  --simulate_generator_forward
  --log_visualization
  --num_frames 81
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  # --resume-from-checkpoint "/mnt/weka/home/hao.zhang/wei/SFhy1.5_distill_1333/checkpoint-300"
  --init_weights_from_safetensors /mnt/weka/home/hao.zhang/wei/hy15_worldplay_tf_init_3333/checkpoint-2800/transformer/diffusion_pytorch_model.safetensors
  # --init_weights_from_safetensors /mnt/weka/home/hao.zhang/wei/hy15_worldplay_df_init_1333/checkpoint-11600/transformer/diffusion_pytorch_model.safetensors
  # --init_weights_from_safetensors /mnt/weka/home/hao.zhang/wei/hy15_distilled_ode_init_3333/checkpoint-2400/transformer/diffusion_pytorch_model.safetensors
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
  # --data_path_2 "$DATA_DIR_2"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 50
  --validation_sampling_steps "8"
  --validation_guidance_scale "1.0" # not used for dmd inference
  --text-encoder-cpu-offload
  # --vae_cpu_offload True
)

optimizer_args=(
  --optimizer-type "muon"
  --learning_rate 1e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 100
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
)

dmd_args=(
  --warp_denoising_step
  # --dmd_denoising_steps '1000,875,750,625,500,375,250,125'
  # --dmd_denoising_steps '1000,760,520,280'
  # --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.5
  --fake_score_learning_rate 5e-6
  --fake_score_betas '0.0,0.999'
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
  # --use-ode-init True
  # --use-context-forcing True
)

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node $NUM_GPUS \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/hy15_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}"