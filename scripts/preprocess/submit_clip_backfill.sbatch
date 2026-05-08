#!/bin/bash
#SBATCH --job-name=clip_backfill
#SBATCH --account=AIFAC_P02_082
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/AIFAC_P02_082/.slurm/%x_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_082/.slurm/%x_%j.err

# CLIP feature backfill (TECH-118). Submit with:
#   sbatch --export=ALL,SRC=/leonardo_scratch/large/.../latents_wan_v1/W21_..._clip,\
#                   VIDEO_ROOT=/leonardo_work/.../clips_wan_v1/master_77f_25fps_720p/clips,\
#                   SHARDS="shard_00",MAX_FILES=15 \
#       scripts/preprocess/submit_clip_backfill.sbatch
#
# Variables:
#   SRC                       - destination cache root containing shard_NN/ subdirs (REQUIRED).
#                               Must be /leonardo_scratch/large (cache rewrite footprint ~3 TB
#                               across both datasets does not fit on /leonardo_work).
#   VIDEO_ROOT                - source mp4 root (REQUIRED). Reads directly from /leonardo_work;
#                               Phase 0b (staging mp4s to scratch) was abandoned — the N=8
#                               prefetch pool amortizes Lustre stat/open latency.
#   SHARDS                    - space-separated shard names to process (default: all under SRC)
#   MAX_FILES                 - cap on parquet files (smoke run); empty = no cap
#   MICRO_BATCH               - CLIP forward micro-batch size (default 256, was 64 in Smoke #1)
#   VIDEO_PREFETCH_WORKERS    - CPU decode threads (default 8; 0 disables, runs serial)
#   MODEL_PATH                - Wan-Fun I2V checkpoint path (default the cached one)

set -euo pipefail
set +u
source /etc/profile || true
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
set -u

module purge
module load profile/deeplrn cuda/12.1

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/scripts/preprocess" ]]; then
    echo "ERROR: REPO_ROOT not resolvable. Submit from the FastVideo repo root or set REPO_ROOT." >&2
    exit 1
fi

source "$REPO_ROOT/.venv-overlay/bin/activate"
export PYTHONPATH="$REPO_ROOT:/leonardo_work/AIFAC_P02_082/code/FastVideo/.venv-fa/lib/python3.12/site-packages"

export HF_HOME=/leonardo_work/AIFAC_P02_082/.cache/huggingface
export HF_HUB_OFFLINE=1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}

cd "$REPO_ROOT"

: "${SRC:?SRC not set}"
: "${VIDEO_ROOT:?VIDEO_ROOT not set}"
MODEL_PATH="${MODEL_PATH:-/leonardo_work/AIFAC_P02_082/.cache/Wan2.1-I2V-14B-480P-Diffusers}"
MICRO_BATCH="${MICRO_BATCH:-256}"
VIDEO_PREFETCH_WORKERS="${VIDEO_PREFETCH_WORKERS:-8}"

echo "=========================================="
echo " job: ${SLURM_JOB_ID:-?} on $(hostname)"
echo " gpus: ${SLURM_GPUS_ON_NODE:-?}"
echo " src:                     $SRC"
echo " video_root:              $VIDEO_ROOT"
echo " shards:                  ${SHARDS:-<all>}"
echo " max_files:               ${MAX_FILES:-<none>}"
echo " micro_batch:             $MICRO_BATCH"
echo " video_prefetch_workers:  $VIDEO_PREFETCH_WORKERS"
echo " model_path:              $MODEL_PATH"
echo "=========================================="

# Launch via `python -m torch.distributed.run` (the overlay venv doesn't ship
# the `torchrun` console script). FastVideo's pipeline init calls
# maybe_init_distributed_environment_and_model_parallel(1,1) which expects
# MASTER_ADDR/PORT/RANK/WORLD_SIZE - torch.distributed.run sets these even
# for nproc=1.
# Derive a unique master port per job so two jobs co-located on the same
# node (which torch.distributed.run uses even at nproc=1) do not collide
# on EADDRINUSE. Range 29500-39499.
MASTER_PORT=$(( 29500 + SLURM_JOB_ID % 10000 ))

CMD=(python -m torch.distributed.run --nproc_per_node=1 --master_port="$MASTER_PORT"
     scripts/preprocess/backfill_clip_features.py
     --src "$SRC"
     --video-root "$VIDEO_ROOT"
     --model-path "$MODEL_PATH"
     --micro-batch "$MICRO_BATCH"
     --video-prefetch-workers "$VIDEO_PREFETCH_WORKERS")

if [[ -n "${SHARDS:-}" ]]; then
    # SHARDS is space-separated; expand into argv
    CMD+=(--shards $SHARDS)
fi
if [[ -n "${MAX_FILES:-}" ]]; then
    CMD+=(--max-files "$MAX_FILES")
fi

"${CMD[@]}"

# Post-run validator intentionally skipped: it scans every parquet file in
# $SRC (Lustre stat-bound, ~6-12 min per shard) and the answer is the same
# until the full cache is filled. Run it once at the end of Stage B and C:
#   python scripts/preprocess/validate_clip_features.py --src "$SRC"
