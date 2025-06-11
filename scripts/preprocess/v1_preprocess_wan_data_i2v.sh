export WANDB_MODE="offline"
export HOME="/mnt/weka/home/hao.zhang/wei"
GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="$HOME/FastVideo/data/wei-i2v-dataset/crush-smol_raw/merge.txt"
OUTPUT_DIR="$HOME/FastVideo/data/wei-i2v-dataset/crush-smol_preprocessed"
VALIDATION_PATH="assets/prompt.txt"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 8 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --validation_prompt_txt $VALIDATION_PATH \
    --samples_per_file 16 \
    --flush_frequency 32 \
    --preprocess_task "i2v" 