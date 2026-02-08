#!/bin/bash
#SBATCH --job-name=hy15_ode_trajectory
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --output=hy15_ode_trajectory/%j.out
#SBATCH --error=hy15_ode_trajectory/%j.err

# conda init
source ~/.venv/bin/activate
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

set -euo pipefail

echo " "
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

# Accept parameters from launch script
START_FILE=${1:-1}  # Starting file number for this node
NODE_ID=${2:-0}  # Node identifier (0-7)

export MODEL_BASE=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v
# Start port number - we'll increment for each job
base_port=$((29603 + NODE_ID * 100))  # Different port range per node

# Create an array of CUDA device IDs
gpu_ids=(0 1 2 3)

GPU_NUM=1

echo "NODE_ID: $NODE_ID"
echo "START_FILE: $START_FILE"
echo "Base port for this node: $base_port"

# Run 8 parallel preprocessing jobs on this node (in 2 batches of 4)
for i in {1..4}; do
    # Calculate port for this job
    port=$((base_port + i))
    
    # Get GPU ID using modulo to cycle through available GPUs
    gpu=${gpu_ids[((i-1))]}
    
    # Calculate which file this GPU should process
    file_num=$((START_FILE + i - 1))
    DATA_MERGE_PATH="/home/hal-weiz/hy15_tf_init_3333/prompts_16k_32_files/v2m_${file_num}.txt"
    
    # Create unique output directory based on node and GPU
    OUTPUT_DIR="/home/hal-weiz/fv-ode-preprocessing-16k-hy15-121/Node_${NODE_ID}_GPU_${i}_File_${file_num}"
    
    # Distribute 16 CPUs across 4 GPUs (4 CPUs per GPU)
    start_cpu=$(( (i-1) * 4 ))
    end_cpu=$(( start_cpu + 3 ))
    
    echo "Starting GPU $gpu processing file v2m_${file_num}.txt on port $port, output: $OUTPUT_DIR"
    
    # Run the preprocessing command in background
    CUDA_VISIBLE_DEVICES=$gpu taskset -c ${start_cpu}-${end_cpu} torchrun --nnodes=1 --nproc_per_node=$GPU_NUM --master_port $port \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_BASE \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 480 \
        --max_width 848 \
        --num_frames 121 \
        --flow_shift 5.0 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --train_fps 24 \
        --samples_per_file 8 \
        --flush_frequency 8 \
        --video_length_tolerance_range 5 \
        --preprocess_task "hy15_ode_trajectory" &

done

# Wait for all jobs on this node to complete
wait

echo "All processing blocks completed!"
