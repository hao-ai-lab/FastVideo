#!/bin/bash
#SBATCH --job-name=matrixgame_sf_dmd
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=logs/slurm/matrixgame_sf_dmd_%j.out
#SBATCH --error=logs/slurm/matrixgame_sf_dmd_%j.err
#SBATCH --exclusive

set -euo pipefail
set -x

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/hal-kaiqin/FastVideo_mg3}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
CONDA_BIN="${CONDA_ROOT}/bin/conda"
ENV_NAME="${ENV_NAME:-FastVideo_kaiqin}"
MASTER_PORT="${MASTER_PORT:-29500}"
WANDB_API_KEY="${WANDB_API_KEY:-7ff8b6e8356924f7a6dd51a0342dd1a422ea9352}"
WANDB_MODE="${WANDB_MODE:-online}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
TOTAL_GPUS="$(( SLURM_JOB_NUM_NODES * GPUS_PER_NODE ))"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_PROCID}"
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY
export WANDB_MODE

if [ -x "${CONDA_BIN}" ]; then
    eval "$("${CONDA_BIN}" shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "conda not found at ${CONDA_BIN}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/slurm"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST}") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT
export NODE_RANK="${SLURM_PROCID}"

echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NODE_RANK: ${NODE_RANK}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "TOTAL_GPUS: ${TOTAL_GPUS}"

RUN_NAME="$(date +"%m%d_%H%M")"
echo "RUN_NAME: ${RUN_NAME}"

GENERATOR_MODEL_PATH="./mg_causal_Solaris"
REAL_SCORE_MODEL_PATH="./mg_bidirectional_Solaris"
FAKE_SCORE_MODEL_PATH="./mg_causal_Solaris"

DATA_DIR="/home/hal-kaiqin/solaris_dataset_unzip/solaris"
VALIDATION_DATASET_FILE="examples/distill/MatrixGame2.0/validation.json"

training_args=(
  --tracker_project_name "matrixgame_sf"
  --output_dir "checkpoints/matrixgame_sf_${RUN_NAME}"
  --wandb_run_name "${RUN_NAME}_test"
  --max_train_steps 1200
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 2
  --num_latent_t 21
  --num_height 352
  --num_width 640
  --simulate_generator_forward
  --num_frames 81
  --num_frame_per_block 3
  --init_weights_from_safetensors "checkpoints/matrixgame_sf_0413_1109/checkpoint-1200_weight_only/ema/generator_ema.safetensors"
)

parallel_args=(
  --num_gpus "${TOTAL_GPUS}"
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim "${TOTAL_GPUS}"
)

model_args=(
  --model_path "${GENERATOR_MODEL_PATH}"
  --pretrained_model_name_or_path "${GENERATOR_MODEL_PATH}"
  --real_score_model_path "${REAL_SCORE_MODEL_PATH}"
  --fake_score_model_path "${FAKE_SCORE_MODEL_PATH}"
)

dataset_args=(
  --data_path "${DATA_DIR}"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --log_visualization
  --visualization-steps 100
  --validation_dataset_file "${VALIDATION_DATASET_FILE}"
  --validation_steps 100
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

optimizer_args=(
  --learning_rate 3e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 400
  --training_state_checkpointing_steps 400
  --weight_decay 0
  --betas "0.9,0.95"
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
  --dmd_denoising_steps "1000,750,500,250"
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 3e-7
  --fake_score_betas "0.9,0.95"
  --warp_denoising_step
)

self_forcing_args=(
  --independent_first_frame False
  --same_step_across_blocks True
  --last_step_only False
  --context_noise 0
)

srun python -m torch.distributed.run \
  --nnodes "${SLURM_JOB_NUM_NODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${SLURM_PROCID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  fastvideo/training/matrixgame_self_forcing_distillation_pipeline.py \
  "${parallel_args[@]}" \
  "${model_args[@]}" \
  "${dataset_args[@]}" \
  "${training_args[@]}" \
  "${optimizer_args[@]}" \
  "${validation_args[@]}" \
  "${miscellaneous_args[@]}" \
  "${dmd_args[@]}" \
  "${self_forcing_args[@]}"
