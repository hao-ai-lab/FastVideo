export WANDB_MODE="offline"
export HOME="/mnt/user_storage/src/"
GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="$HOME/FastVideo/data/crush-smol/merge.txt"
OUTPUT_DIR="$HOME/FastVideo/data/crush-smol_parq_i2v"
VALIDATION_PATH="examples/training/finetune/wan_i2v_14b_480p/crush_smol/validation.json"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 4 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 77 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --validation_dataset_file $VALIDATION_PATH \
    --samples_per_file 16 \
    --flush_frequency 32 \
    --preprocess_task "i2v" 