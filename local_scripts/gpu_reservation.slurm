#!/bin/bash
#SBATCH --job-name=hyhf-lora-5e6-r32
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --output=slurm_log/slurm.out
#SBATCH --error=slurm_log/slurm.err
#SBATCH --exclusive
#SBATCH --time=72:00:00

PYTHON_VIRTUAL_ENVIRONMENT=fastvideo
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

echo " "
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

cd ~/yongqi/FastVideo
export MASTER_PORT=29803
torchrun --nnodes 1 --nproc_per_node 8 --master_port $MASTER_PORT \
    fastvideo/train.py \
    --seed 1024 \
    --pretrained_model_name_or_path ~/data/hunyuan_diffusers \
    --model_type hunyuan_hf \
    --cache_dir data/.cache \
    --data_json_path ~/data/HD-MixKit-Hunyuan-Distill-125x768x1280/videos2caption.json \
    --validation_prompt_dir ~/data/HD-MixKit-Hunyuan-Distill-125x768x1280/validation \
    --gradient_checkpointing \
    --train_batch_size 4 \
    --num_latent_t 32 \
    --sp_size 8 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 8000 \
    --learning_rate 5e-6 \
    --mixed_precision bf16 \
    --checkpointing_steps 200 \
    --validation_steps 100 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir data/outputs/SBA_lora_5e6_r32 \
    --tracker_project_name SBA \
    --num_frames 125 \
    --num_width 1280 \
    --num_height 768 \
    --validation_guidance_scale "1.0" \
    --use_lora \
    --lora_rank 32 \
    --lora_alpha 64

echo "Run completed at:- "
date