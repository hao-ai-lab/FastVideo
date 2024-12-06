# export WANDB_MODE="offline"
GPU_NUM=8
MODEL_PATH="/ephemeral/hao.zhang/outputfolder/ckptfolder/mochi_diffuser"
DATA_MERGE_PATH="/ephemeral/hao.zhang/resourcefolder/Mochi-Synthetic-Data-BW-Finetune/merge.txt"
OUTPUT_DIR="./data/BW-Finetune-Synthetic-Data_test"

rchrun --nproc_per_node=$GPU_NUM \
    ./fastvideo/utils/data_preprocess/finetune_data_VAE.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=163 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR

to



torchrun --nproc_per_node=$GPU_NUM \
    ./fastvideo/utils/data_preprocess/finetune_data_T5.py \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR