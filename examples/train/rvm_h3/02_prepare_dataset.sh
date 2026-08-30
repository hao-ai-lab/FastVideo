#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
activate_rvm_env
require_path "${FASTH3_MODEL_DIR}"

PREPROCESS_GPUS="${RVM_PREPROCESS_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
PREPROCESS_GPUS="${PREPROCESS_GPUS:-1}"
if (( PREPROCESS_GPUS < 1 )); then PREPROCESS_GPUS=1; fi

prompt_args=(
    --output-dir "${RVM_PROMPT_DIR}"
    --seed "${RVM_PROMPT_SEED:-42}"
    --eval-prompts "${RVM_EVAL_PROMPTS:-100}"
    --smoke-prompts "${RVM_SMOKE_PROMPTS:-16}"
)
if [[ -n "${RVM_PROMPT_SOURCE:-}" ]]; then
    prompt_args+=(--source "${RVM_PROMPT_SOURCE}")
fi
if [[ -n "${RVM_MAX_TRAIN_PROMPTS:-}" ]]; then
    prompt_args+=(--max-train-prompts "${RVM_MAX_TRAIN_PROMPTS}")
fi
python examples/train/rvm_h3/prepare_prompts.py "${prompt_args[@]}"

encode_prompts() {
    local prompt_file="$1"
    local output_dir="$2"
    local shards="$3"
    if find "${output_dir}" -name '*.parquet' -print -quit 2>/dev/null | grep -q . && [[ "${RVM_FORCE_PREPROCESS:-0}" != 1 ]]; then
        echo "Skipping existing encoded dataset: ${output_dir}"
        return
    fi
    rm -rf "${output_dir}"
    mkdir -p "${output_dir}"
    local shard
    for ((shard=0; shard<shards; shard++)); do
        CUDA_VISIBLE_DEVICES="${shard}" \
            python -m fastvideo.pipelines.preprocess.preprocess_minimax_h3_text_only \
                --prompts-file "${prompt_file}" \
                --model-path "${FASTH3_MODEL_DIR}" \
                --output-dir "${output_dir}" \
                --shard-index "${shard}" \
                --num-shards "${shards}" \
                --samples-per-file 64 \
                --flush-every 64 \
                >"${output_dir}/encode_shard_${shard}.log" 2>&1 &
    done
    wait
}

# The full training bank is expensive (~6 MB of FP32 Qwen3-VL embeddings per
# prompt). Keep RVM_MAX_TRAIN_PROMPTS small for the first systems test, then
# unset it for the production bank after the 1-GPU/8-GPU gates pass.
encode_prompts "${RVM_PROMPT_DIR}/smoke_h3.txt" "${RVM_SMOKE_DATA}" 1
encode_prompts "${RVM_PROMPT_DIR}/eval_h3.txt" "${RVM_EVAL_DATA}" "${PREPROCESS_GPUS}"
encode_prompts "${RVM_PROMPT_DIR}/train_h3.txt" "${RVM_TRAIN_DATA}" "${PREPROCESS_GPUS}"

python - "${RVM_SMOKE_DATA}" "${RVM_EVAL_DATA}" "${RVM_TRAIN_DATA}" <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq
for value in sys.argv[1:]:
    root = Path(value)
    files = sorted(root.rglob("*.parquet"))
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if rows <= 0:
        raise RuntimeError(f"No encoded rows under {root}")
    print(f"{root}: {rows} rows in {len(files)} parquet files")
PY
