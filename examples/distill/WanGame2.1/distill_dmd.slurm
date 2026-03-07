#!/bin/bash
#SBATCH --job-name=wg-dmd
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=log/dmd_train_output/dmd_%j.out
#SBATCH --error=log/dmd_train_output/dmd_%j.err
#SBATCH --exclusive

set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/mhuo-fv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo:$PYTHONPATH"

# Basic Info
export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

RUN_NAME=$(date +"%m%d_%H%M")
echo "RUN_NAME: $RUN_NAME"

GENERATOR_MODEL_PATH="../wg_models/WanGame-2.1-0223-9000steps"
REAL_SCORE_MODEL_PATH="../wg_models/WanGame-2.1-0223-9000steps"  # Teacher model
FAKE_SCORE_MODEL_PATH="../wg_models/WanGame-2.1-0223-9000steps"  # Critic model

DATA_DIR="../traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed"
VALIDATION_DATASET_FILE="examples/distill/WanGame2.1/validation.json"

# Training arguments
training_args=(
  --tracker_project_name "wangame_dmd"
  --output_dir "checkpoints/wangame_dmd_${RUN_NAME}"
  --wandb_run_name "${RUN_NAME}_dmd"
  --max_train_steps 3000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 20
  --num_height 352
  --num_width 640
  --num_frames 77
  --enable_gradient_checkpointing_type "full"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
)

# Parallel arguments
parallel_args=(
  --num_gpus 32
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 32
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

# Validation arguments
validation_args=(
  --log_validation
  --log_visualization
  --visualization-steps 200
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 2e-6
  --mixed_precision "bf16"
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
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
)

# DMD-specific arguments
dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
)

mkdir -p log/dmd_train_output

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 8 \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wangame_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}"
