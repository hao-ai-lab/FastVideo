export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export TOKENIZERS_PARALLELISM=false
# export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA

DATA_DIR=[your data dir]
VALIDATION_DATASET_FILE=[your validation dataset file]
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=4,5
# IP=[MASTER NODE IP]

# Make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS\
    fastvideo/training/wan_training_pipeline.py\
    --config scripts/finetune/finetune_v1.yaml \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DATASET_FILE" \
    --engine.num_gpus "$NUM_GPUS" \
    --training.checkpoint.output_dir "$DATA_DIR/outputs/wan_finetune"