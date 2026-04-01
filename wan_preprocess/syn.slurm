#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=960G
#SBATCH --exclusive
#SBATCH --time=72:00:00

# conda init
# source ~/conda/miniconda/bin/activate
source ~/miniconda3/bin/activate
PYTHON_VIRTUAL_ENVIRONMENT=fastvideo
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

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

num_gpus=1
export MODEL_BASE=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
export HF_HOME=/mnt/data/huggingface
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/shared-bin/local/usr/lib/aarch64-linux-gnu:$CUDA_HOME/lib64"
export CUDACXX="$CUDA_HOME/bin/nvcc"
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

# Start port number - we'll increment for each job
base_port=$((29603 + NODE_ID * 100))  # Different port range per node

# Create an array of CUDA device IDs
gpu_ids=(0 1 2 3 4 5 6 7)

GPU_NUM=1
MODEL_TYPE="wan"

echo "NODE_ID: $NODE_ID"
echo "START_FILE: $START_FILE"
echo "Base port for this node: $base_port"

# Run 8 parallel preprocessing jobs on this node
for i in {1..8}; do
    # Calculate port for this job
    port=$((base_port + i))
    
    # Get GPU ID using modulo to cycle through available GPUs
    gpu=${gpu_ids[((i-1))]}
    
    # Calculate which file this GPU should process
    file_num=$((START_FILE + i - 1))
    DATA_MERGE_PATH="/mnt/home/zhouw.jerry2017/mixkit_prompts/mixkit_prompts_${file_num}.txt"
    
    # Create unique output directory based on node and GPU
    OUTPUT_DIR="/mnt/data/fv-ode-preprocessing-6k-mixkit-wan-1.3b/Node_${NODE_ID}_GPU_${i}_File_${file_num}"
    
    start_cpu=$(( (i-1)*2 ))  # Reduced CPU allocation for 8 nodes
    end_cpu=$(( start_cpu+1 ))
    
    echo "Starting GPU $gpu processing file mixkit_prompts_${file_num}.txt on port $port, output: $OUTPUT_DIR"
    
    # Run the preprocessing command in background
    CUDA_VISIBLE_DEVICES=$gpu taskset -c ${start_cpu}-${end_cpu} torchrun --nnodes=1 --nproc_per_node=$GPU_NUM --master_port $port \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_BASE \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 480 \
        --max_width 832 \
        --num_frames 81 \
        --flow_shift 5.0 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --train_fps 16 \
        --samples_per_file 8 \
        --flush_frequency 8 \
        --video_length_tolerance_range 5 \
        --preprocess_task "ode_trajectory" &
done

# Wait for all jobs on this node to complete
wait

echo "All processing blocks completed!"
