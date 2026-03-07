#!/bin/bash
#SBATCH --partition=main
#SBATCH --qos=hao
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=960G
#SBATCH --exclusive
#SBATCH --time=72:00:00

# conda init
source ~/conda/miniconda/bin/activate
conda activate fastvideo_kaiqin

# Accept parameters from launch script
START_FILE=${1:-0}  # Starting file number for this node
NODE_ID=${2:-0}     # Node identifier (0-7)

MODEL_PATH="../Matrix-Game-2.0-Base-Diffusers"
OUTPUT_BASE="../FastvideoWorldModel-MC/preprocessed"

# Port range calculation
base_port=$((29700 + NODE_ID * 100)) # Using a different port range to avoid collision with other tasks
gpu_ids=(0 1 2 3 4 5 6 7)

for i in {1..8}; do
    port=$((base_port + i))
    gpu=${gpu_ids[((i-1))]}
    file_num=$((START_FILE + i - 1))
    
    DATA_MERGE_PATH="../FastvideoWorldModel-MC/gen/merge_${file_num}.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/gpu_${gpu}_file_${file_num}"
    
    # CPU binding
    start_cpu=$(( (i-1)*2 ))
    end_cpu=$(( start_cpu+1 ))
    
    echo "Starting GPU $gpu processing file merge_${file_num}.txt on port $port"

    CUDA_VISIBLE_DEVICES=$gpu taskset -c ${start_cpu}-${end_cpu} torchrun --nnodes=1 --nproc_per_node=1 --master_port $port \
        FastVideo/fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_PATH \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 352 \
        --max_width 640 \
        --num_frames 81 \
        --flow_shift 5.0 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --train_fps 25 \
        --samples_per_file 8 \
        --flush_frequency 8 \
        --video_length_tolerance_range 5 \
        --preprocess_task "matrixgame_ode_trajectory" &
done

wait
echo "Node $NODE_ID ODE preprocessing blocks completed!"
