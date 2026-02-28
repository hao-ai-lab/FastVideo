#!/bin/bash
#SBATCH --job-name=wg-ar-diff
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=log/ar_diff_output/ar_diff_%j.out
#SBATCH --error=log/ar_diff_output/ar_diff_%j.err
#SBATCH --exclusive

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/mhuo-fv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo:$PYTHONPATH"

export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export MASTER_PORT=29503
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=online
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN

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

MODEL_PATH="../wg_models/WanGame-2.1-0223-9000steps"

DATA_DIR="../traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1/preprocessed"
VALIDATION_DATASET_FILE="examples/training/consistency_finetune/causal_wangame_ode_init/validation_same.json"

training_args=(
  --tracker_project_name wangame_ar_diffusion
  --output_dir "checkpoints/wangame_ar_diffusion_${RUN_NAME}"
  --wandb_run_name "${RUN_NAME}_df_bs32"
  --override_transformer_cls_name "CausalWanGameActionTransformer3DModel"
  --max_train_steps 5000
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 15
  --num_height 352
  --num_width 640
  --enable_gradient_checkpointing_type "full"
  --num_frames 57
  --num_frame_per_block 3
)

parallel_args=(
  --num_gpus 32
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 32
)

model_args=(
  --model_path $MODEL_PATH
  --pretrained_model_name_or_path $MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 200
  --validation_sampling_steps "50"
  --validation_guidance_scale "1.0"
)

optimizer_args=(
  --learning_rate 2e-5
  --mixed_precision "bf16"
  --training_state_checkpointing_steps 500
  --weight_only_checkpointing_steps 500
  --weight_decay 0.01
  --betas '0.9,0.999'
  --max_grad_norm 1.0
  --lr_scheduler cosine
  --lr_warmup_steps 100
)

miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 8
  --seed 42
)

mkdir -p log/ar_diff_output

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 8 \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/wangame_ar_diffusion_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
