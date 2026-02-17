#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

CONFIG_PATH="${SCRIPT_DIR}/configs/overfit_v2.yaml"
OVERRIDE_PROMPTS_JSONL=""
OVERRIDE_OUTPUT_ROOT=""
OVERRIDE_NUM_GPUS=""
OVERRIDE_START_IDX=""
OVERRIDE_END_IDX=""
SKIP_GENERATION=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --start-idx)
      OVERRIDE_START_IDX="$2"
      shift 2
      ;;
    --end-idx)
      OVERRIDE_END_IDX="$2"
      shift 2
      ;;
    --prompts-jsonl)
      OVERRIDE_PROMPTS_JSONL="$2"
      shift 2
      ;;
    --output-root)
      OVERRIDE_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num-gpus)
      OVERRIDE_NUM_GPUS="$2"
      shift 2
      ;;
    --skip-generation)
      SKIP_GENERATION=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing config file: ${CONFIG_PATH}" >&2
  exit 1
fi

yaml_get() {
  local key="$1"
  local value
  value="$(sed -n "s/^${key}:[[:space:]]*//p" "${CONFIG_PATH}" | head -n1)"
  value="${value%%#*}"
  value="$(echo "${value}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  echo "${value}"
}

resolve_path() {
  local path_in="$1"
  if [[ "${path_in}" == /* ]]; then
    echo "${path_in}"
  else
    echo "${ROOT_DIR}/${path_in}"
  fi
}

infer_output_name_from_jsonl() {
  local jsonl_path="$1"
  local base
  base="$(basename "${jsonl_path}")"
  base="${base%.jsonl}"
  echo "${base}_output"
}

canonical_file_path() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    realpath "${p}"
  else
    local parent
    parent="$(dirname "${p}")"
    parent="$(realpath "${parent}")"
    echo "${parent}/$(basename "${p}")"
  fi
}

PYTHON_ENV="$(yaml_get python_env)"
PROMPTS_JSONL_CFG="$(yaml_get prompts_jsonl)"
DATA_ROOT_BASE="$(resolve_path "$(yaml_get data_root)")"
GENERATE_VIDEOS="$(yaml_get generate_videos)"
START_IDX="$(yaml_get start_idx)"
END_IDX="$(yaml_get end_idx)"
NUM_GPUS_GENERATE="$(yaml_get num_gpus_generate)"
MANIFEST_WORKERS="$(yaml_get manifest_workers)"

if [[ -n "${OVERRIDE_PROMPTS_JSONL}" ]]; then
  PROMPTS_JSONL="$(resolve_path "${OVERRIDE_PROMPTS_JSONL}")"
elif [[ -n "${PROMPTS_JSONL_CFG}" ]]; then
  PROMPTS_JSONL="$(resolve_path "${PROMPTS_JSONL_CFG}")"
else
  echo "Missing prompts jsonl. Provide --prompts-jsonl or set prompts_jsonl in config." >&2
  exit 1
fi

if [[ -n "${OVERRIDE_OUTPUT_ROOT}" ]]; then
  DATA_ROOT="$(resolve_path "${OVERRIDE_OUTPUT_ROOT}")"
else
  DATA_ROOT="${DATA_ROOT_BASE}/$(infer_output_name_from_jsonl "${PROMPTS_JSONL}")"
fi

if [[ -n "${OVERRIDE_NUM_GPUS}" ]]; then
  NUM_GPUS_GENERATE="${OVERRIDE_NUM_GPUS}"
fi
if [[ -n "${OVERRIDE_START_IDX}" ]]; then
  START_IDX="${OVERRIDE_START_IDX}"
fi
if [[ -n "${OVERRIDE_END_IDX}" ]]; then
  END_IDX="${OVERRIDE_END_IDX}"
fi
if [[ "${SKIP_GENERATION}" == "true" ]]; then
  GENERATE_VIDEOS="false"
fi

if ! [[ "${NUM_GPUS_GENERATE}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS_GENERATE}" -lt 1 ]]; then
  echo "--num-gpus must be a positive integer (got: ${NUM_GPUS_GENERATE})" >&2
  exit 1
fi

VIDEOS_DIR="${DATA_ROOT}/videos"
REPORTS_DIR="${DATA_ROOT}/reports"
LOGS_DIR="${DATA_ROOT}/logs"
MANIFEST_JSON="${DATA_ROOT}/videos2caption.json"
RAW_REPORT_JSON="${REPORTS_DIR}/raw_summary.json"
SOURCE_MARKER="${DATA_ROOT}/.source_jsonl"

mkdir -p "${VIDEOS_DIR}" "${REPORTS_DIR}" "${LOGS_DIR}"

echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] PROMPTS_JSONL=${PROMPTS_JSONL}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] VIDEOS_DIR=${VIDEOS_DIR}"
echo "[INFO] NUM_GPUS_GENERATE=${NUM_GPUS_GENERATE}"
echo "[INFO] GENERATE_VIDEOS=${GENERATE_VIDEOS}"

if [[ ! -f "${PROMPTS_JSONL}" ]]; then
  echo "Missing prompts jsonl: ${PROMPTS_JSONL}" >&2
  exit 1
fi

PROMPTS_JSONL_CANON="$(canonical_file_path "${PROMPTS_JSONL}")"
if [[ -f "${SOURCE_MARKER}" ]]; then
  PREV_JSONL="$(sed -n '1p' "${SOURCE_MARKER}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  if [[ -n "${PREV_JSONL}" && "${PREV_JSONL}" != "${PROMPTS_JSONL_CANON}" ]]; then
    echo "Output dir collision detected." >&2
    echo "Output root: ${DATA_ROOT}" >&2
    echo "Existing source jsonl: ${PREV_JSONL}" >&2
    echo "Requested source jsonl: ${PROMPTS_JSONL_CANON}" >&2
    exit 1
  fi
fi
echo "${PROMPTS_JSONL_CANON}" > "${SOURCE_MARKER}"

TOTAL_LINES="$(wc -l < "${PROMPTS_JSONL}")"
if [[ "${TOTAL_LINES}" -lt 1 ]]; then
  echo "Prompts jsonl is empty: ${PROMPTS_JSONL}" >&2
  exit 1
fi

if [[ -z "${START_IDX}" ]]; then
  START_IDX=0
fi
if [[ -z "${END_IDX}" ]]; then
  END_IDX="${TOTAL_LINES}"
fi

if ! [[ "${START_IDX}" =~ ^[0-9]+$ ]] || ! [[ "${END_IDX}" =~ ^[0-9]+$ ]]; then
  echo "start/end indices must be integers (start=${START_IDX}, end=${END_IDX})" >&2
  exit 1
fi
if [[ "${START_IDX}" -lt 0 ]]; then
  START_IDX=0
fi
if [[ "${END_IDX}" -gt "${TOTAL_LINES}" ]]; then
  END_IDX="${TOTAL_LINES}"
fi
if [[ "${START_IDX}" -ge "${END_IDX}" ]]; then
  echo "Invalid range: [${START_IDX}, ${END_IDX}) for total lines=${TOTAL_LINES}" >&2
  exit 1
fi

echo "[INFO] TOTAL_LINES=${TOTAL_LINES}"
echo "[INFO] START_IDX=${START_IDX}"
echo "[INFO] END_IDX=${END_IDX}"

if [[ "${GENERATE_VIDEOS}" == "true" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    AVAILABLE_GPUS="$(nvidia-smi -L | wc -l)"
    if [[ "${NUM_GPUS_GENERATE}" -gt "${AVAILABLE_GPUS}" ]]; then
      echo "--num-gpus (${NUM_GPUS_GENERATE}) exceeds available GPUs (${AVAILABLE_GPUS})." >&2
      exit 1
    fi
  fi

  RANGE_SIZE=$((END_IDX - START_IDX))
  CHUNK_SIZE=$(((RANGE_SIZE + NUM_GPUS_GENERATE - 1) / NUM_GPUS_GENERATE))
  echo "[INFO] RANGE_SIZE=${RANGE_SIZE}"
  echo "[INFO] CHUNK_SIZE=${CHUNK_SIZE}"

  PIDS=()
  STARTS=()
  ENDS=()
  GPUS=()

  for ((gpu=0; gpu<NUM_GPUS_GENERATE; gpu++)); do
    s=$((START_IDX + gpu * CHUNK_SIZE))
    e=$((s + CHUNK_SIZE))
    if [[ "${e}" -gt "${END_IDX}" ]]; then
      e="${END_IDX}"
    fi
    if [[ "${s}" -ge "${e}" ]]; then
      continue
    fi

    worker_log="${LOGS_DIR}/generate_gpu${gpu}_${s}_${e}.log"
    echo "[INFO] Launching gpu=${gpu} range=[${s}, ${e}) log=${worker_log}"

    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
        conda run --no-capture-output -n "${PYTHON_ENV}" python -u \
        "${ROOT_DIR}/examples/training/finetune/ltx2/overfit/generate_videos.py" \
        "${PROMPTS_JSONL}" \
        "${s}" \
        "${e}" \
        "${VIDEOS_DIR}" \
        --num-gpus 1 \
        --skip-existing
    ) > "${worker_log}" 2>&1 &

    PIDS+=("$!")
    STARTS+=("${s}")
    ENDS+=("${e}")
    GPUS+=("${gpu}")
  done

  if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "No generation workers launched for range [${START_IDX}, ${END_IDX})." >&2
    exit 1
  fi

  for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
      echo "Generation worker failed: gpu=${GPUS[$i]} range=[${STARTS[$i]}, ${ENDS[$i]})" >&2
      echo "See log: ${LOGS_DIR}/generate_gpu${GPUS[$i]}_${STARTS[$i]}_${ENDS[$i]}.log" >&2
      exit 1
    fi
  done
fi

conda run --no-capture-output -n "${PYTHON_ENV}" python -u \
  "${ROOT_DIR}/examples/training/finetune/ltx2/overfit/build_videos2caption_from_jsonl.py" \
  --videos-dir "${VIDEOS_DIR}" \
  --prompts-jsonl "${PROMPTS_JSONL}" \
  --output-json "${MANIFEST_JSON}" \
  --workers "${MANIFEST_WORKERS}"

conda run --no-capture-output -n "${PYTHON_ENV}" python -u \
  "${SCRIPT_DIR}/validate_dataset.py" \
  --mode raw \
  --data-root "${DATA_ROOT}" \
  --output-report "${RAW_REPORT_JSON}"

echo "[DONE] Raw dataset is ready."
echo "[DONE] Manifest: ${MANIFEST_JSON}"
echo "[DONE] Report: ${RAW_REPORT_JSON}"
