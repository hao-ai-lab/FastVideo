# export WANDB_MODE="offline"
GPU_NUM=8
MODEL_PATH="data/hunyuan"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="data/Mixkit-All-Clips/merge.txt"
OUTPUT_DIR="data/Hunyuan-Mixkit-Data"

# torchrun --nproc_per_node=$GPU_NUM \
#     ./fastvideo/utils/data_preprocess/finetune_data_VAE.py \
#     --model_path $MODEL_PATH \
#     --data_merge_path $DATA_MERGE_PATH \
#     --train_batch_size=1 \
#     --max_height=480 \
#     --max_width=848 \
#     --num_frames=93 \
#     --dataloader_num_workers 1 \
#     --output_dir=$OUTPUT_DIR \
#     --model_type $MODEL_TYPE \
#     --train_fps 24 



torchrun --nproc_per_node=$GPU_NUM \
    ./fastvideo/utils/data_preprocess/finetune_data_text_encoder.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR 