

# Basic info
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_$$"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NODE_RANK=0
export TOKENIZERS_PARALLELISM=false
export WANDB_BASE_URL="https://api.wandb.ai"
export FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN
export WANDB_API_KEY="your_wandb_api_key_here" # TODO: Replace with your actual key or load from a secure location

# Configs
NUM_GPUS=4
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DATA_DIR=data/Wan-Syn_77x448x832_600k
VALIDATION_DATASET_FILE=examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json

torchrun \
  --nnodes 1 \
  --nproc_per_node "$NUM_GPUS" \
  --node_rank "$NODE_RANK" \
  --rdzv_backend c10d \
  --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
  fastvideo/training/wan_training_pipeline.py \
  --config examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/T2V-1.3B-VSA_4_gpu_debug.yaml \
  --model_path "$MODEL_PATH" \
  --engine.num_gpus "$NUM_GPUS" \
  --engine.parallelism.hsdp_replicate_dim "$NUM_GPUS" \
  --training.data.data_path "$DATA_DIR" \
  --training.validation.dataset_file "$VALIDATION_DATASET_FILE"
