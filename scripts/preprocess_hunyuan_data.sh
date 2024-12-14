# export WANDB_MODE="offline"
GPU_NUM=8
SHARD_NUM=8
SHARD_IDX=0
MODEL_PATH="data/hunyuan"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="data/Distill-30K-Src/merge.txt"
OUTPUT_DIR="data/HD-Hunyuan-30K-Distill-Data_Shard${SHARD_IDX}"
VALIDATION_PATH="assets/prompt.txt"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=720 \
    --max_width=1280 \
    --num_frames=129 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 \
    --shard_num=$SHARD_NUM \
    --shard_idx=$SHARD_IDX



torchrun --nproc_per_node=$GPU_NUM  \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR 



torchrun --nproc_per_node=1 \
    fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt_txt $VALIDATION_PATH