# export WANDB_MODE="offline"
GPU_NUM=1 # 2,4,8
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_MERGE_PATH="data/mini_i2v_dataset/crush-smol_raw/merge.txt"
OUTPUT_DIR="data/mini_i2v_dataset/crush-smol_preprocessed"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --config scripts/preprocess/v1_preprocess_wan_data_t2v.yaml \
    --model_path $MODEL_PATH \
    --preprocess.data_merge_path $DATA_MERGE_PATH \
    --preprocess.dataset_output_dir=$OUTPUT_DIR
