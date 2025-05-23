# export WANDB_MODE="offline"
GPU_NUM=1 # 2,4,8
# MODEL_PATH="/home/ray/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_MERGE_PATH="data/cats_480/merge.txt"
OUTPUT_DIR="data/cats_480_latents/"
VALIDATION_PATH="assets/prompt.txt"

# torchrun --nproc_per_node=$GPU_NUM \
#     fastvideo/data_preprocess/preprocess_vae_latents_v1.py \
#     --model_path $MODEL_PATH \
#     --data_merge_path $DATA_MERGE_PATH \
#     --train_batch_size=1 \
#     --max_height=480 \
#     --max_width=832 \
#     --num_frames=81 \
#     --dataloader_num_workers 1 \
#     --output_dir=$OUTPUT_DIR \
#     --train_fps 16

# torchrun --nproc_per_node=$GPU_NUM \
#     fastvideo/data_preprocess/preprocess_text_embeddings_v1.py \
#     --model_path $MODEL_PATH \
#     --output_dir=$OUTPUT_DIR 

torchrun --nproc_per_node=1 \
    fastvideo/data_preprocess/preprocess_validation_text_embeddings_v1.py \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt_txt $VALIDATION_PATH