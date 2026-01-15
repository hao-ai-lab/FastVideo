# export WANDB_MODE="offline"
GPU_NUM=8 # 2,4,8
MODEL_PATH="/data/yanguo.sun/hunyuan-video/HunyuanVideo/ckpts"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="/data/yanguo.sun/hunyuan-video/OpenVidHD/merge.txt"
OUTPUT_DIR="/data/yanguo.sun/hunyuan-video/datasets"
VALIDATION_PATH="/data/yanguo.sun/hunyuan-video/OpenVidHD/validation_dataset/prompt.txt"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=720 \
    --max_width=1280 \
    --num_frames=125 \
    --dataloader_num_workers 4 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 24 

torchrun --nproc_per_node=$GPU_NUM \
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
