#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --time=144:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=gpumid
#SBATCH --account=llm360-1
#SBATCH --reservation=data

# Load necessary modules
module load cuda/12.1

# Initialize Conda
eval "$(/lustre/scratch/users/hao.zhang/dacheng/anaconda3/condabin/conda shell.bash hook)"
conda activate fastvideo  # Replace with your actual environment name
cd /home/hao.zhang/peiyuan/FastVideo-OSP

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
# Run the training command
export WANDB_API_KEY
torchrun --nnodes 4 --nproc_per_node 4 \
    fastvideo/train.py \
    --rdzv_id=456 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    --seed 42 \
    --pretrained_model_name_or_path data/mochi \
    --cache_dir "data/.cache" \
    --data_json_path "data/T_J-Finetune-Synthetic-Data/videos2caption.json" \
    --validation_prompt_dir "data/T_J-Finetune-Synthetic-Data/validation_prompt_embed_mask" \
    --uncond_prompt_dir "data/T_J-Finetune-Synthetic-Data/uncond_prompt_embed_mask" \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 14 \
    --sp_size 2 \
    --train_sp_batch_size 2 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=500 \
    --learning_rate=1e-5 \
    --mixed_precision="bf16" \
    --checkpointing_steps=200 \
    --validation_steps 100 \
    --validation_sampling_steps 64 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.1 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir="data/outputs/T_J_FT"
