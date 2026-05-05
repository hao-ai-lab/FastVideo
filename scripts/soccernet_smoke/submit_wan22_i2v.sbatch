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

# Layered Python environment:
#  - python 3.12 + huggingface_hub + diffusers etc. come from the conda env
#  - torch 2.9.1+cu128, flash-attn, fastvideo deps come from the proven .venv-fa
#  - the user's local FastVideo source comes first on PYTHONPATH so the editable
#    code in /leonardo_work/AIFAC_P02_082/code/moein/FastVideo wins.
source /leonardo_work/AIFAC_P02_082/code/moein/FastVideo/.venv-overlay/bin/activate

export PYTHONPATH=/leonardo_work/AIFAC_P02_082/code/moein/FastVideo:/leonardo_work/AIFAC_P02_082/code/FastVideo/.venv-fa/lib/python3.12/site-packages

export HF_HOME=/leonardo_work/AIFAC_P02_082/.cache/huggingface
export HF_HUB_OFFLINE=1
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-$$}

cd /leonardo_work/AIFAC_P02_082/code/moein/FastVideo

echo "=========================================="
echo " job: ${SLURM_JOB_ID:-?} on $(hostname)"
echo " gpus: ${SLURM_GPUS_ON_NODE:-?}"
echo " python: $(which python)"
nvidia-smi -L 2>&1 | head -2 || true
echo "=========================================="

python scripts/soccernet_smoke/run_wan22_i2v.py
