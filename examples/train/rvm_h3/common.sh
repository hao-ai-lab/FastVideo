#!/usr/bin/env bash
set -euo pipefail

RVM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RVM_SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export RVM_ENV_NAME="${RVM_ENV_NAME:-fasth3-rvm}"
export RVM_ARTIFACT_ROOT="${RVM_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/rvm_h3}"
export FASTH3_MODEL_DIR="${FASTH3_MODEL_DIR:-${RVM_ARTIFACT_ROOT}/models/fasth3}"
export H3_TEACHER_REPO="${H3_TEACHER_REPO:-MiniMaxAI/MiniMax-H3}"
export H3_TEACHER_REVISION="${H3_TEACHER_REVISION:-bfc8ed0353f5a9733be73e6b2c98ec0948195b86}"
export H3_TEACHER_MODEL_DIR="${H3_TEACHER_MODEL_DIR:-${RVM_ARTIFACT_ROOT}/models/minimax-h3-teacher}"
export H3_REST_CACHE_ROOT="${H3_REST_CACHE_ROOT:-${RVM_ARTIFACT_ROOT}/rest_cache}"
export H3_REST_COMPACT_CACHE="${H3_REST_COMPACT_CACHE:-${H3_REST_CACHE_ROOT}/compact}"
export H3_REST_FULL_CACHE="${H3_REST_FULL_CACHE:-${H3_REST_CACHE_ROOT}/full}"
export RVM_PROMPT_DIR="${RVM_PROMPT_DIR:-${RVM_ARTIFACT_ROOT}/prompts}"
export RVM_TRAIN_DATA="${RVM_TRAIN_DATA:-${RVM_ARTIFACT_ROOT}/data/train}"
export RVM_EVAL_DATA="${RVM_EVAL_DATA:-${RVM_ARTIFACT_ROOT}/data/eval}"
export RVM_SMOKE_DATA="${RVM_SMOKE_DATA:-${RVM_ARTIFACT_ROOT}/data/train_smoke}"
export RVM_REWARD_ROOT="${RVM_REWARD_ROOT:-${RVM_ARTIFACT_ROOT}/rewards}"
export VIDEOALIGN_RUNTIME_PATH="${VIDEOALIGN_RUNTIME_PATH:-${RVM_REWARD_ROOT}/VideoAlign}"
export VIDEOALIGN_CHECKPOINT_PATH="${VIDEOALIGN_CHECKPOINT_PATH:-${RVM_REWARD_ROOT}/VideoReward}"
export MJ_VIDEO_RUNTIME_PATH="${MJ_VIDEO_RUNTIME_PATH:-${RVM_REWARD_ROOT}/MJ-Video}"
export MJ_VIDEO_MODEL_PATH="${MJ_VIDEO_MODEL_PATH:-${RVM_REWARD_ROOT}/MJ-VIDEO-2B}"
export MJ_VIDEO_BASE_MODEL_PATH="${MJ_VIDEO_BASE_MODEL_PATH:-${RVM_REWARD_ROOT}/InternVL2-2B}"
export MJ_VIDEO_CALIBRATION_PATH="${MJ_VIDEO_CALIBRATION_PATH:-${RVM_REWARD_ROOT}/physion_mj_calibration.json}"
export HF_HOME="${HF_HOME:-${RVM_ARTIFACT_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

# Exact FastH3 deployment policy. Change these only for a documented ablation.
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-VIDEO_SPARSE_ATTN_H3}"
export FASTVIDEO_VSA_SM100A="${FASTVIDEO_VSA_SM100A:-0}"
export FASTVIDEO_VSA_CUTEDSL="${FASTVIDEO_VSA_CUTEDSL:-0}"
export FASTVIDEO_MINIMAX_H3_FUSIONS="${FASTVIDEO_MINIMAX_H3_FUSIONS:-0}"
export FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE="${FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE:-1}"
export FASTVIDEO_REST_VAE_DECODE_BATCH_SIZE="${FASTVIDEO_REST_VAE_DECODE_BATCH_SIZE:-1}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RVM_ARTIFACT_ROOT}" "${H3_REST_CACHE_ROOT}" "${HF_HOME}"

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
    python examples/train/rvm_h3/verify_clean_source.py
    NUM_GPUS="${NUM_GPUS:-8}" bash examples/train/run.sh "${config}" "${topology[@]}" "$@"
}

run_h3_rest_training() {
    run_rvm_training "$@"
}
