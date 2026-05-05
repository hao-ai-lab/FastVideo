#!/bin/bash
#SBATCH --job-name=wan22_smoke_5frames
#SBATCH --account=AIFAC_P02_082
#SBATCH --partition=boost_usr_prod
# NOTE 2026-05-03: when extending this to multi-shard arrays or multi-node
# training, switch to --qos=normal -- lprod caps at 8 nodes per account
# while normal has no visible per-account cap (verified by 32-shard parallel
# encode array on 2026-05-03). bprod is unusable: it requires >=65 nodes
# per job. See reference_leonardo_slurm.md memory.
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_P02_082/.slurm/%x_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_082/.slurm/%x_%j.err

set -euo pipefail

set +u
source /etc/profile || true
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
set -u

module purge
module load profile/deeplrn cuda/12.1

# Resolve the FastVideo checkout to run from. Priority:
#   1) REPO_ROOT explicit override (sbatch --export=ALL,REPO_ROOT=/path ...)
#   2) SLURM_SUBMIT_DIR -- the dir sbatch was called from. Works when you
#      submit from the repo root (the common case).
# We do NOT fall back to BASH_SOURCE because under sbatch the script is
# copied to a spool dir, so it would resolve to /var/spool/slurmd/... .
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$REPO_ROOT" || ! -d "$REPO_ROOT/scripts/soccernet_smoke" ]]; then
    echo "ERROR: REPO_ROOT not resolvable. Submit from the FastVideo repo root, or set REPO_ROOT explicitly." >&2
    exit 1
fi

# Layered Python environment:
#  - python 3.12 + huggingface_hub + diffusers etc. come from the conda env
#  - torch 2.9.1+cu128, flash-attn, fastvideo deps come from the proven .venv-fa
#  - the local FastVideo source comes first on PYTHONPATH so the editable
#    checkout at $REPO_ROOT wins.
source "$REPO_ROOT/.venv-overlay/bin/activate"

export PYTHONPATH="$REPO_ROOT:/leonardo_work/AIFAC_P02_082/code/FastVideo/.venv-fa/lib/python3.12/site-packages"

export HF_HOME=/leonardo_work/AIFAC_P02_082/.cache/huggingface
export HF_HUB_OFFLINE=1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}

cd "$REPO_ROOT"

SEED_DIR="$REPO_ROOT/outputs/wan22_soccernet_smoke/seeds"

echo "=========================================="
echo " job: ${SLURM_JOB_ID:-?} on $(hostname)"
echo " gpus: ${SLURM_GPUS_ON_NODE:-?}"
echo " repo: $REPO_ROOT"
echo " python: $(which python)"
nvidia-smi -L 2>&1 | head -2 || true
echo "=========================================="

# Sample seed frames if not already present (idempotent: skip if seeds.json exists).
if [[ ! -f "$SEED_DIR/seeds.json" ]]; then
    python scripts/soccernet_smoke/sample_seed_frames.py --out-dir "$SEED_DIR"
else
    echo "[seeds] $SEED_DIR/seeds.json exists, skipping sampler"
fi

python scripts/soccernet_smoke/run_wan22_i2v.py
