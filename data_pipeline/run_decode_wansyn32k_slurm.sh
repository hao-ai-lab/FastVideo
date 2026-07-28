#!/usr/bin/env bash
# Wan 2.2 5B VAE decode of the 32k HF latent dataset -> 480x832 mp4s.
# Sharded across NODES * GPUS * PROCS_PER_GPU workers via SLURM_PROCID.
set -euo pipefail
WORK=/mnt/lustre/vlm-s4duan
: "${PARQUET_DIR:?}"; : "${OUT_DIR:?}"; : "${VAE_DIR:?}"
NODES=${NODES:-8}; GPUS=${GPUS:-4}; PROCS_PER_GPU=${PROCS_PER_GPU:-2}
TASKS_PER_NODE=$(( GPUS * PROCS_PER_GPU ))
CPUS_PER_TASK=$(( 128 / TASKS_PER_NODE )); [ "$CPUS_PER_TASK" -lt 1 ] && CPUS_PER_TASK=1
NUM_SHARDS=$(( NODES * TASKS_PER_NODE ))

mkdir -p "$OUT_DIR" "$WORK/logs"
echo "nodes=$NODES gpus/node=$GPUS procs/gpu=$PROCS_PER_GPU -> $NUM_SHARDS workers"

sbatch -N "$NODES" --gres=gpu:$GPUS --ntasks-per-node=$TASKS_PER_NODE --exclusive \
  --cpus-per-task=$CPUS_PER_TASK --mem=0 -t 12:00:00 -J decode_wansyn32k \
  --chdir="$WORK/FastVideo" -o "$WORK/logs/decode_wansyn32k_%j.out" -e "$WORK/logs/decode_wansyn32k_%j.out" \
  --wrap "srun --chdir=$WORK/FastVideo \
    bash -lc 'source .venv/bin/activate && \
      export HOME=$WORK HF_HOME=$WORK/.hf TORCH_HOME=$WORK/.torch \
        MPLCONFIGDIR=$WORK/.mpl TOKENIZERS_PARALLELISM=false && \
      export CUDA_VISIBLE_DEVICES=\$(( SLURM_LOCALID % $GPUS )) && \
      python data_pipeline/decode_wansyn32k.py \
        --parquet-dir $PARQUET_DIR --vae-dir $VAE_DIR --out-dir $OUT_DIR \
        --num-shards $NUM_SHARDS --shard \$SLURM_PROCID'"
