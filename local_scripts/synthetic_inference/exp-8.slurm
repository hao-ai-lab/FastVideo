#!/bin/bash
#SBATCH --job-name=syn-8
#SBATCH --partition=mbzuai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=960G
#SBATCH --output=slurm_log_last_dance/syn-8.out
#SBATCH --error=slurm_log_last_dance/syn-8.err
#SBATCH --exclusive
#SBATCH --time=72:00:00

conda init
source ~/conda/miniconda/bin/activate
PYTHON_VIRTUAL_ENVIRONMENT=fastvideo
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
cd /mbz/users/hao.zhang/yongqi/FastVideo

echo " "
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

# Calculate prompts per GPU
prompt_start_idx_all=1680
num_promtps_all=240
num_gpus=8
prompts_per_gpu=$((num_promtps_all / num_gpus))
export MODEL_BASE=data/hunyuan

for gpu_id in $(seq 0 $((num_gpus-1))); do
    start_idx=$((prompt_start_idx_all + gpu_id * prompts_per_gpu))
    end_idx=$((start_idx + prompts_per_gpu))
    prompt_start_end_idx_single_gpu="${start_idx},${end_idx}"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python fastvideo/sample/generate_synthetic_hunyuan.py \
        --height 768 \
        --width 1280 \
        --num_frames 117 \
        --num_inference_steps 50 \
        --guidance_scale 1 \
        --embedded_cfg_scale 6 \
        --flow_shift 7 \
        --flow-reverse \
        --prompt_json ~/data/HD-MixKit-Hunyuan-Distill-125x768x1280/videos2caption.json \
        --seed 1024 \
        --output_path outputs_latents/ \
        --model_path $MODEL_BASE \
        --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
        --only_save_latent True \
        --prompt_start_end_idx ${prompt_start_end_idx_single_gpu} &
done

# Wait for all background jobs to complete
wait