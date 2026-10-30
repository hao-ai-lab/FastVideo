# Pika H3 Flex QAD environment (OmniRef QAD v2 env with this project code). Usage: env.sh <cmd...>
set -a
Q=/mnt/nfs/vlm-aryan/h3-omniref-qad
source /mnt/nfs/vlm-aryan/fasth3-33b-20260806/secrets.env >/dev/null 2>&1
EXP=/mnt/lustre/vlm-wlsaidhi/fastvideo/experiments/main-hf-step3500-gb200-20261005T203006Z
KERNEL=$EXP/kernel/install
PY=/mnt/nfs/vlm-aryan/h3-nvfp4-decoder/venv/bin/python
CODE=/mnt/nfs/vlm-aryan/pika-flex-qad/code
PYTHONPATH=$KERNEL:$CODE
HF_HOME=/mnt/nfs/vlm-aryan/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WANDB_MODE=${WANDB_MODE:-online} WANDB_DIR=/mnt/nfs/vlm-aryan/pika-flex-qad/wandb WANDB_PROJECT=pika-h3-flex-nvfp4 WANDB_ENTITY=aryan5v-san-jose-state-university
PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
SCRATCH=/tmp/pika-flex-${SLURM_JOB_ID:-nojob}
HOME=$SCRATCH/home XDG_CACHE_HOME=$SCRATCH/cache XDG_CONFIG_HOME=$SCRATCH/config TMPDIR=$SCRATCH/tmp
TRITON_CACHE_DIR=/tmp/triton-pika-flex TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-pika-flex TORCH_EXTENSIONS_DIR=/tmp/torchext-pika-flex
NUMBA_CACHE_DIR=$SCRATCH/numba FLASHINFER_WORKSPACE_BASE=/mnt/nfs/vlm-aryan/h3-omniref-qad/cache/flashinfer
mkdir -p $HOME $XDG_CACHE_HOME $XDG_CONFIG_HOME $TMPDIR $NUMBA_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TORCH_EXTENSIONS_DIR $WANDB_DIR
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_NCCL_ENABLE_MONITORING=0 NCCL_CUMEM_ENABLE=0 NCCL_MNNVL_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_NET=IB NCCL_IB_HCA==mlx5_0,mlx5_1,mlx5_3,mlx5_4 NCCL_SOCKET_IFNAME=eth0 NCCL_SOCKET_FAMILY=AF_INET NCCL_DEBUG=WARN
FASTVIDEO_MINIMAX_H3_FUSIONS=0 FASTVIDEO_H3_VSA_FP4=0 FASTVIDEO_VSA_SM100A=0
set +a
cd $CODE
exec "$@"
