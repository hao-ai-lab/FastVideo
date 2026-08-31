#!/usr/bin/env bash
set -euo pipefail

RVM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RVM_SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export RVM_ENV_NAME="${RVM_ENV_NAME:-fasth3-rvm}"
export RVM_ARTIFACT_ROOT="${RVM_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/rvm_h3}"
export FASTH3_MODEL_DIR="${FASTH3_MODEL_DIR:-${RVM_ARTIFACT_ROOT}/models/fasth3}"
export RVM_PROMPT_DIR="${RVM_PROMPT_DIR:-${RVM_ARTIFACT_ROOT}/prompts}"
export RVM_TRAIN_DATA="${RVM_TRAIN_DATA:-${RVM_ARTIFACT_ROOT}/data/train}"
export RVM_EVAL_DATA="${RVM_EVAL_DATA:-${RVM_ARTIFACT_ROOT}/data/eval}"
export RVM_SMOKE_DATA="${RVM_SMOKE_DATA:-${RVM_ARTIFACT_ROOT}/data/train_smoke}"
export RVM_REWARD_ROOT="${RVM_REWARD_ROOT:-${RVM_ARTIFACT_ROOT}/rewards}"
export VIDEOALIGN_RUNTIME_PATH="${VIDEOALIGN_RUNTIME_PATH:-${RVM_REWARD_ROOT}/VideoAlign}"
export VIDEOALIGN_CHECKPOINT_PATH="${VIDEOALIGN_CHECKPOINT_PATH:-${RVM_REWARD_ROOT}/VideoReward}"
export HF_HOME="${HF_HOME:-${RVM_ARTIFACT_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

# Exact FastH3 deployment policy. Change these only for a documented ablation.
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-VIDEO_SPARSE_ATTN_H3}"
export FASTVIDEO_VSA_SM100A="${FASTVIDEO_VSA_SM100A:-0}"
export FASTVIDEO_VSA_CUTEDSL="${FASTVIDEO_VSA_CUTEDSL:-0}"
export FASTVIDEO_MINIMAX_H3_FUSIONS="${FASTVIDEO_MINIMAX_H3_FUSIONS:-0}"
export FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE="${FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RVM_ARTIFACT_ROOT}" "${HF_HOME}"

activate_rvm_env() {
    # Modal/Docker images already own the Python environment. Explicitly opt
    # out of conda activation rather than installing a nested environment.
    if [[ "${RVM_SKIP_CONDA:-0}" == "1" ]]; then
        if ! command -v python >/dev/null 2>&1; then
            echo "RVM_SKIP_CONDA=1 but python is not on PATH." >&2
            exit 1
        fi
        return
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        # shellcheck disable=SC1090
        source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "conda is required. Run 00_create_conda_env.sh from a shell with conda installed." >&2
        exit 1
    fi
    conda activate "${RVM_ENV_NAME}"
}

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path is missing: ${path}" >&2
        exit 1
    fi
}

rvm_topology_args() {
    local num_gpus="${NUM_GPUS:-8}"
    local sp_size="${RVM_SP_SIZE:-4}"
    if (( num_gpus % sp_size != 0 )); then
        echo "NUM_GPUS=${num_gpus} must be divisible by RVM_SP_SIZE=${sp_size}" >&2
        exit 1
    fi
    local replicas=$((num_gpus / sp_size))
    printf '%s\n' \
        --training.distributed.num_gpus "${num_gpus}" \
        --training.distributed.sp_size "${sp_size}" \
        --training.distributed.tp_size 1 \
        --training.distributed.hsdp_replicate_dim "${replicas}" \
        --training.distributed.hsdp_shard_dim "${sp_size}"
}

run_rvm_training() {
    local config="$1"
    shift
    local topology=()
    while IFS= read -r line; do topology+=("${line}"); done < <(rvm_topology_args)
    NUM_GPUS="${NUM_GPUS:-8}" bash examples/train/run.sh "${config}" "${topology[@]}" "$@"
}
