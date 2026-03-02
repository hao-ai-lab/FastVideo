#!/bin/bash
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=960G
#SBATCH --exclusive
#SBATCH --time=72:00:00

# conda init
source ~/conda/miniconda/bin/activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/alexfv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo_clean:$PYTHONPATH"

# Accept parameters from launch script
START_FILE=${1:-1}  # Starting file number for this node
NODE_ID=${2:-0}     # Node identifier (0-7)

MODEL_PATH="../mg_models/Matrix-Game-2.0-Foundation-Diffusers"
OUTPUT_BASE="../solaris/datasets/train/preprocessed_0228_81"

# Port range calculation
base_port=$((29500 + NODE_ID * 100))
gpu_ids=(0 1 2 3 4 5 6 7)

for i in {1..8}; do
    port=$((base_port + i))
    gpu=${gpu_ids[((i-1))]}
    file_num=$((START_FILE + i - 1))
    
    DATA_MERGE_PATH="../solaris/datasets/train/merge_${file_num}.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/gpu_${gpu}_file_${file_num}"
    echo "DATA_MERGE_PATH: $DATA_MERGE_PATH"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    
    # CPU binding (optional, kept from syn.slurm logic)
    start_cpu=$(( (i-1)*2 ))
    end_cpu=$(( start_cpu+1 ))
    
    echo "Starting GPU $gpu processing file merge_${file_num}.txt on port $port"

    CUDA_VISIBLE_DEVICES=$gpu taskset -c ${start_cpu}-${end_cpu} torchrun --nnodes=1 --nproc_per_node=1 --master_port $port \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_PATH \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 352 \
        --max_width 640 \
        --num_frames 81 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --samples_per_file 8 \
        --train_fps 20 \
        --flush_frequency 8 \
        --preprocess_task matrixgame &
done

wait
echo "Node $NODE_ID processing blocks completed!"
