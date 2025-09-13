# export WANDB_MODE="offline"
GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_TYPE="wan"
DATA_MERGE_PATH="data/mini_i2v_dataset/crush-smol_raw/merge.txt"
OUTPUT_DIR="data/mini_i2v_dataset/crush-smol_preprocessed"
VALIDATION_PATH="examples/training/finetune/wan_t2v_1_3b/crush_smol/validation.json"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --max_height 480 \
    --max_width 832 \
    --num_frames 81 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 16 \
    --validation_dataset_file $VALIDATION_PATH \
    --samples_per_file 1 \
    --flush_frequency 1 \
    --video_length_tolerance_range 5 \
    --preprocess_task "t2v"
