# export WANDB_MODE="offline"
GPU_NUM=1 # 2,4,8
MODEL_PATH="/workspace/data/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
TEXT_ENCODER_PATH="/workspace/data/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tokenizer"
MODEL_TYPE="wan"
DATA_MERGE_PATH="/workspace/data/Mixkit-Src/merge.txt"
OUTPUT_DIR="/workspace/data/HD-Mixkit-Finetune-Wan"
VALIDATION_PATH="assets/prompt.txt"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size=4 \
    --preprocess_text_batch_size=4 \
    --max_height=480 \
    --max_width=832 \
    --num_frames=81 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --text_encoder_name $TEXT_ENCODER_PATH \
    --train_fps 16 \
    --validation_prompt_txt $VALIDATION_PATH