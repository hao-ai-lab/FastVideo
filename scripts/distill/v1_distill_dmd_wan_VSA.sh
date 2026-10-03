export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=offline
export WANDB_API_KEY=
export TRITON_CACHE_DIR=/tmp/triton_cache
DATA_DIR=mini_i2v_dataset/crush-smol_preprocessed/combined_parquet_dataset/
VALIDATION_DIR=mini_i2v_dataset/crush-smol_raw/validation.json
NUM_GPUS=8
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export TOKENIZERS_PARALLELISM=false

# Train generator with VSA
# Make sure that num_latent_t is a multiple of sp_size
torchrun --nnodes 1 --nproc_per_node $NUM_GPUS \
    fastvideo/training/wan_distillation_pipeline.py \
    --config scripts/distill/v1_distill_dmd_wan_VSA.yaml \
    --training.data.data_path "$DATA_DIR" \
    --training.validation.dataset_file "$VALIDATION_DIR" \
    --engine.num_gpus "$NUM_GPUS" \
    --engine.parallelism.hsdp_replicate_dim "$NUM_GPUS"