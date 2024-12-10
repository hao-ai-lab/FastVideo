GPU_NUM=8
MODEL_PATH="./data/mochi"
DATA_MERGE_PATH="./data/test/merge.txt"
OUTPUT_DIR="./data/Test-Data"

torchrun --nproc_per_node=$GPU_NUM \
    ./fastvideo/utils/data_preprocess/finetune_data_VAE.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=163 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR



torchrun --nproc_per_node=$GPU_NUM \
    ./fastvideo/utils/data_preprocess/finetune_data_T5.py \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR