#!/bin/bash
#SBATCH --job-name=gen_vid
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=logs/gen_%j.out
#SBATCH --error=logs/gen_%j.err
#SBATCH --exclusive

set -e

# Environment Setup
source ~/miniconda3/bin/activate
export PATH=~/miniconda3/envs/fastvideo/bin:$PATH

PROMPT_DIR="/home/hal-jundas/codes/FastVideo-demo/data/Davids048/ltx2-data/video_prompts.gpt-5-mini-2025-08-07.0-1000"
OUTPUT_DIR="/home/hal-jundas/codes/FastVideo-demo/data/ltx2-base-videos/"

# Specify Start and End Index Here
START_IDX=600
END_IDX=700

echo "Job ID: $SLURM_JOB_ID"
echo "Processing range: $START_IDX to $END_IDX"

conda activate fastvideo && python data_preperation/generate_videos_from_json_prompts.py \
  --prompt-dir "$PROMPT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --start-idx "$START_IDX" \
  --end-idx "$END_IDX" \
  --num-gpus 4
