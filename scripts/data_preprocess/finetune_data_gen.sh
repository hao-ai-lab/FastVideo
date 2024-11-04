# export WANDB_MODE="offline"
MODEL_PATH="data/mochi"
MOCHI_DIR="/ephemeral/hao.zhang/resourcefolder/mochi/mochi-1-preview"
DATA_MERGE_PATH="./data/dummyVid/merge.txt"
OUTPUT_DIR="./data/Encoder-Synthetic-Data"

python \
    ./fastvideo/utils/data_preprocess/finetune_data_VAE.py \
    --mochi_dir $MOCHI_DIR \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --target_length=163 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR

python \
    ./fastvideo/utils/data_preprocess/finetune_data_T5.py \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR