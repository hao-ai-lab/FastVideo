#!/usr/bin/env bash
set -euo pipefail

cd /FastVideo

export CUDA_VISIBLE_DEVICES=0,1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "===== GPU information ====="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo "===== FastVideo source ====="
python -c 'import fastvideo; print(fastvideo.__file__)'

echo "===== FlashAttention ====="
python -c 'import flash_attn; print("flash_attn:", flash_attn.__version__)'

echo "===== Start 2-GPU pure Ring inference ====="
echo "num_gpus=2, sp_size=2, ring_size=2"

mkdir -p video_samples logs

python examples/inference/basic/basic.py \
    2>&1 | tee logs/ring_2gpu_pipeline.log

echo "===== Finished ====="
echo "Videos: /FastVideo/video_samples"
echo "Log: /FastVideo/logs/ring_2gpu_pipeline.log"