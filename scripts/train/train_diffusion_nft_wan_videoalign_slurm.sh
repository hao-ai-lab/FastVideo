#!/usr/bin/env bash
set -euo pipefail

# Submit/run the Modal DiffusionNFT Wan video RL equivalent on a Slurm cluster.
#
# From the login node:
#   PARTITION=all NUM_GPUS=4 bash scripts/train/train_diffusion_nft_wan_videoalign_slurm.sh
#
# Useful overrides:
#   NUM_FRAMES=17 MAX_TRAIN_STEPS=10 CHECK_REWARDS=0 ...
#   bash scripts/train/train_diffusion_nft_wan_videoalign_slurm.sh --method.validation.num_prompts 4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
fi

CONFIG="${CONFIG:-examples/train/configs/rl/wan/diffusion_nft_videoalign.yaml}"
PARTITION="${PARTITION:-all}"
NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS="${NUM_GPUS:-4}"
GRES="${GRES:-gpu:nvidia_g:${NUM_GPUS}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-0}"
TIME="${TIME:-08:00:00}"
JOB_NAME="${JOB_NAME:-diffusion_nft_wan_videoalign}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-logs/slurm}"
MASTER_PORT="${MASTER_PORT:-29531}"
ACCOUNT="${ACCOUNT:-}"
QOS="${QOS:-}"
EXCLUDE="${EXCLUDE:-}"
DRY_RUN="${DRY_RUN:-0}"
NUM_FRAMES="${NUM_FRAMES:-29}"
NUM_LATENT_T="${NUM_LATENT_T:-0}"
if (( (NUM_FRAMES - 1) % 4 != 0 )); then
    echo "NUM_FRAMES must satisfy Wan's 4n + 1 rule; got ${NUM_FRAMES}." >&2
    echo "29 is the closest supported value to 30." >&2
    exit 1
fi
DERIVED_NUM_LATENT_T="$(( (NUM_FRAMES - 1) / 4 + 1 ))"
if (( NUM_LATENT_T > 0 && NUM_LATENT_T != DERIVED_NUM_LATENT_T )); then
    echo "NUM_FRAMES=${NUM_FRAMES} implies NUM_LATENT_T=${DERIVED_NUM_LATENT_T}; got ${NUM_LATENT_T}." >&2
    exit 1
fi

if [[ "${_FASTVIDEO_DIFFUSION_NFT_SLURM_WORKER:-0}" != "1" ]]; then
    mkdir -p "${REPO_ROOT}/${SLURM_LOG_DIR}"
    sbatch_args=(
        --job-name="${JOB_NAME}"
        --partition="${PARTITION}"
        --nodes="${NUM_NODES}"
        --ntasks="${NUM_NODES}"
        --ntasks-per-node=1
        --gres="${GRES}"
        --cpus-per-task="${CPUS_PER_TASK}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${REPO_ROOT}/${SLURM_LOG_DIR}/${JOB_NAME}_%j.out"
        --error="${REPO_ROOT}/${SLURM_LOG_DIR}/${JOB_NAME}_%j.err"
    )
    if [[ -n "${ACCOUNT}" ]]; then
        sbatch_args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${QOS}" ]]; then
        sbatch_args+=(--qos="${QOS}")
    fi
    if [[ -n "${EXCLUDE}" ]]; then
        sbatch_args+=(--exclude="${EXCLUDE}")
    fi

    echo "=== DiffusionNFT Wan Slurm submission ==="
    echo "repo:       ${REPO_ROOT}"
    echo "config:     ${CONFIG}"
    echo "partition:  ${PARTITION}"
    echo "nodes:      ${NUM_NODES}"
    echo "gpus/node:  ${NUM_GPUS}"
    echo "gres:       ${GRES}"
    echo "job name:   ${JOB_NAME}"
    echo "extra args: $*"
    echo "========================================="

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'sbatch'
        printf ' %q' "${sbatch_args[@]}"
        printf ' --export=ALL,_FASTVIDEO_DIFFUSION_NFT_SLURM_WORKER=1 %q' "$0"
        if (($# > 0)); then
            printf ' %q' "$@"
        fi
        echo
        exit 0
    fi

    sbatch "${sbatch_args[@]}" \
        --export=ALL,_FASTVIDEO_DIFFUSION_NFT_SLURM_WORKER=1 \
        "$0" "$@"
    exit 0
fi

cd "${REPO_ROOT}"

if [[ "${CONFIG}" = /* ]]; then
    CONFIG_PATH="${CONFIG}"
else
    CONFIG_PATH="${REPO_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Training config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
    if [[ ! -x "${CONDA_ENV_PATH}/bin/python" ]]; then
        echo "CONDA_ENV_PATH was set, but no python was found at ${CONDA_ENV_PATH}/bin/python" >&2
        exit 1
    fi
    export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
elif [[ -n "${CONDA_ROOT:-}" ]]; then
    if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        echo "CONDA_ROOT was set, but conda.sh was not found under ${CONDA_ROOT}" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-fastvideo}"
fi

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
    uv pip install --prerelease=allow -e .
    uv pip install --prerelease=allow -r examples/train/requirements-diffusion-nft.txt
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diffusion_nft_wan}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}_${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/.slurm_data}"
CACHE_PARENT="${SCRATCH:-${REPO_ROOT}/.slurm_cache}"
CACHE_ROOT="${CACHE_ROOT:-${CACHE_PARENT}/diffusion_nft}"
RUN_CONFIG_DIR="${RUN_CONFIG_DIR:-${REPO_ROOT}/outputs/diffusion_nft_run_configs/${RUN_ID}}"
DIFFUSION_NFT_ROOT="${DIFFUSION_NFT_ROOT:-${CACHE_ROOT}/DiffusionNFT}"
VIDEOALIGN_CHECKPOINT_PATH="${VIDEOALIGN_CHECKPOINT_PATH:-${CACHE_ROOT}/VideoReward}"

DATASET="${DATASET:-world-r1-enhanced-dynamic}"
REWARD="${REWARD:-videoalign}"
MAX_PROMPTS="${MAX_PROMPTS:-512}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"
NUM_SAMPLES_PER_PROMPT="${NUM_SAMPLES_PER_PROMPT:-4}"
NUM_BATCHES_PER_EPOCH="${NUM_BATCHES_PER_EPOCH:-1}"
COLLECTION_BATCH_SIZE="${COLLECTION_BATCH_SIZE:-4}"
INNER_EPOCHS="${INNER_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:--1}"
SAMPLE_NUM_STEPS="${SAMPLE_NUM_STEPS:-50}"
SAMPLE_FLOW_SHIFT="${SAMPLE_FLOW_SHIFT:--1}"
SAMPLE_GUIDANCE_SCALE="${SAMPLE_GUIDANCE_SCALE:--1}"
VALIDATION_NUM_STEPS="${VALIDATION_NUM_STEPS:-50}"
VALIDATION_NUM_PROMPTS="${VALIDATION_NUM_PROMPTS:-16}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4}"
LOG_SAMPLE_MAX_VIDEOS="${LOG_SAMPLE_MAX_VIDEOS:-16}"
PREPROCESS_BATCH_SIZE="${PREPROCESS_BATCH_SIZE:-128}"
PREPROCESS_NUM_GPUS="${PREPROCESS_NUM_GPUS:-1}"
PREPROCESS_MASTER_PORT="${PREPROCESS_MASTER_PORT:-29541}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SP_SIZE="${SP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
HSDP_REPLICATE_DIM="${HSDP_REPLICATE_DIM:-1}"
HSDP_SHARD_DIM="${HSDP_SHARD_DIM:-$((NUM_NODES * NUM_GPUS))}"
PROJECT_NAME="${PROJECT_NAME:-diffusion_nft_wan}"
RUN_NAME="${RUN_NAME:-wan2.1_diffusion_nft_videoalign_${RUN_ID}}"
CHECK_REWARDS="${CHECK_REWARDS:-1}"
REWARD_DEVICE="${REWARD_DEVICE:-auto}"

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))
if (( HSDP_REPLICATE_DIM * HSDP_SHARD_DIM != TOTAL_GPUS )); then
    echo "Invalid HSDP mesh: HSDP_REPLICATE_DIM * HSDP_SHARD_DIM must equal total GPUs." >&2
    echo "Got ${HSDP_REPLICATE_DIM} * ${HSDP_SHARD_DIM} != ${TOTAL_GPUS}" >&2
    exit 1
fi
if (( (TOTAL_GPUS * COLLECTION_BATCH_SIZE) % NUM_SAMPLES_PER_PROMPT != 0 )); then
    echo "K-repeat sampling requires TOTAL_GPUS * COLLECTION_BATCH_SIZE divisible by NUM_SAMPLES_PER_PROMPT." >&2
    exit 1
fi
if (( (TOTAL_GPUS * TRAIN_BATCH_SIZE) % NUM_SAMPLES_PER_PROMPT != 0 )); then
    echo "Training batches should keep full prompt groups: TOTAL_GPUS * TRAIN_BATCH_SIZE must be divisible by NUM_SAMPLES_PER_PROMPT." >&2
    exit 1
fi

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_diffusion_nft_wan_${SLURM_JOB_ID:-manual}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export DIFFUSION_NFT_ROOT
export VIDEOALIGN_CHECKPOINT_PATH

mkdir -p "${OUTPUT_DIR}" "${DATA_ROOT}" "${CACHE_ROOT}" "${RUN_CONFIG_DIR}" logs/train

echo "=== DiffusionNFT Wan Slurm job ==="
echo "host:                    $(hostname)"
echo "slurm job:               ${SLURM_JOB_ID:-manual}"
echo "nodes / gpus per node:   ${NUM_NODES} / ${NUM_GPUS}"
echo "total gpus:              ${TOTAL_GPUS}"
echo "output dir:              ${OUTPUT_DIR}"
echo "data root:               ${DATA_ROOT}"
echo "cache root:              ${CACHE_ROOT}"
echo "dataset / reward:        ${DATASET} / ${REWARD}"
echo "frames / latent T:       ${NUM_FRAMES} / ${DERIVED_NUM_LATENT_T} (29 frames is the closest Wan-supported value to 30)"
echo "validation steps/prompts/batch: ${VALIDATION_NUM_STEPS} / ${VALIDATION_NUM_PROMPTS} / ${VALIDATION_BATCH_SIZE}"
echo "=================================="
nvidia-smi || true

prep_cmd=(
    "${PYTHON_BIN}" examples/train/prepare_diffusion_nft_assets.py
    --repo-root "${REPO_ROOT}"
    --config "${CONFIG_PATH}"
    --data-root "${DATA_ROOT}"
    --cache-root "${CACHE_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --run-config-dir "${RUN_CONFIG_DIR}"
    --diffusion-nft-root "${DIFFUSION_NFT_ROOT}"
    --videoalign-checkpoint-path "${VIDEOALIGN_CHECKPOINT_PATH}"
    --dataset "${DATASET}"
    --reward "${REWARD}"
    --max-prompts "${MAX_PROMPTS}"
    --num-frames "${NUM_FRAMES}"
    --num-latent-t "${DERIVED_NUM_LATENT_T}"
    --num-gpus "${TOTAL_GPUS}"
    --sp-size "${SP_SIZE}"
    --tp-size "${TP_SIZE}"
    --hsdp-replicate-dim "${HSDP_REPLICATE_DIM}"
    --hsdp-shard-dim "${HSDP_SHARD_DIM}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
    --num-samples-per-prompt "${NUM_SAMPLES_PER_PROMPT}"
    --collection-batch-size "${COLLECTION_BATCH_SIZE}"
    --inner-epochs "${INNER_EPOCHS}"
    --train-batch-size "${TRAIN_BATCH_SIZE}"
    --log-sample-max-videos "${LOG_SAMPLE_MAX_VIDEOS}"
    --preprocess-batch-size "${PREPROCESS_BATCH_SIZE}"
    --preprocess-num-gpus "${PREPROCESS_NUM_GPUS}"
    --preprocess-master-port "${PREPROCESS_MASTER_PORT}"
    --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
    --project-name "${PROJECT_NAME}"
    --run-name "${RUN_NAME}"
    --json
)
if [[ "${CHECK_REWARDS}" == "1" ]]; then
    prep_cmd+=(--check-rewards --reward-device "${REWARD_DEVICE}")
fi
if awk "BEGIN {exit !(${LEARNING_RATE} >= 0)}"; then
    prep_cmd+=(--learning-rate "${LEARNING_RATE}")
fi
if (( SAMPLE_NUM_STEPS > 0 )); then
    prep_cmd+=(--sample-num-steps "${SAMPLE_NUM_STEPS}")
fi
if awk "BEGIN {exit !(${SAMPLE_FLOW_SHIFT} >= 0)}"; then
    prep_cmd+=(--sample-flow-shift "${SAMPLE_FLOW_SHIFT}")
fi
if awk "BEGIN {exit !(${SAMPLE_GUIDANCE_SCALE} >= 0)}"; then
    prep_cmd+=(--sample-guidance-scale "${SAMPLE_GUIDANCE_SCALE}")
fi

echo "Preparing DiffusionNFT assets:"
printf '  %q' "${prep_cmd[@]}"
echo
"${prep_cmd[@]}"

RUN_CONFIG_PATH="${RUN_CONFIG_DIR}/diffusion_nft_wan_run.yaml"
if [[ ! -f "${RUN_CONFIG_PATH}" ]]; then
    echo "Prepared run config not found: ${RUN_CONFIG_PATH}" >&2
    exit 1
fi

nodes=( $(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}") )
MASTER_ADDR="${MASTER_ADDR:-${nodes[0]}}"
export MASTER_ADDR MASTER_PORT

train_args=(
    --config "${RUN_CONFIG_PATH}"
    --training.checkpoint.output_dir "${OUTPUT_DIR}"
    --training.tracker.run_name "${RUN_NAME}"
    --training.loop.max_train_steps "${MAX_TRAIN_STEPS}"
    --training.distributed.num_gpus "${TOTAL_GPUS}"
    --training.distributed.sp_size "${SP_SIZE}"
    --training.distributed.tp_size "${TP_SIZE}"
    --training.distributed.hsdp_replicate_dim "${HSDP_REPLICATE_DIM}"
    --training.distributed.hsdp_shard_dim "${HSDP_SHARD_DIM}"
    --training.loop.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --training.data.num_frames "${NUM_FRAMES}"
    --training.data.num_latent_t "${DERIVED_NUM_LATENT_T}"
    --method.num_video_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
    --method.sample_train_batch_size "${COLLECTION_BATCH_SIZE}"
    --method.num_inner_epochs "${INNER_EPOCHS}"
    --method.train_batch_size "${TRAIN_BATCH_SIZE}"
)
if (( VALIDATION_NUM_STEPS > 0 )); then
    train_args+=(--method.validation.num_steps "${VALIDATION_NUM_STEPS}")
fi
if (( VALIDATION_NUM_PROMPTS > 0 )); then
    train_args+=(--method.validation.num_prompts "${VALIDATION_NUM_PROMPTS}")
fi
if (( VALIDATION_BATCH_SIZE > 0 )); then
    train_args+=(--method.validation.batch_size "${VALIDATION_BATCH_SIZE}")
fi
if (( NUM_BATCHES_PER_EPOCH > 0 )); then
    train_args+=(--method.num_batches_per_epoch "${NUM_BATCHES_PER_EPOCH}")
fi

train_cmd=(
    srun
    --ntasks "${NUM_NODES}"
    --ntasks-per-node 1
    bash -c
    'torchrun --nnodes "$1" --nproc_per_node "$2" --node_rank "$SLURM_PROCID" --rdzv_backend c10d --rdzv_endpoint "$3" -m fastvideo.train.entrypoint.train "${@:4}"'
    bash
    "${NUM_NODES}"
    "${NUM_GPUS}"
    "${MASTER_ADDR}:${MASTER_PORT}"
    "${train_args[@]}"
)

echo "Launching training:"
printf '  %q' "${train_cmd[@]}" "$@"
echo

"${train_cmd[@]}" "$@"
