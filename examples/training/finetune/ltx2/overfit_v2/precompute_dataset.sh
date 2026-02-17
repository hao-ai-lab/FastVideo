#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

CONFIG_PATH="${SCRIPT_DIR}/configs/overfit_v2.yaml"
OVERRIDE_DATA_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data-root)
      OVERRIDE_DATA_ROOT="$2"
      shift 2
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

PYTHON_ENV="$(yaml_get python_env)"
DATA_ROOT_CFG="$(yaml_get data_root)"
if [[ -n "${OVERRIDE_DATA_ROOT}" ]]; then
  DATA_ROOT="$(resolve_path "${OVERRIDE_DATA_ROOT}")"
else
  DATA_ROOT="$(resolve_path "${DATA_ROOT_CFG}")"
fi
MODEL_PATH="$(yaml_get model_path)"
NUM_GPUS_PREPROCESS="$(yaml_get num_gpus_preprocess)"
PREPROCESS_MASTER_PORT="$(yaml_get preprocess_master_port)"
WITH_AUDIO="$(yaml_get with_audio)"
VIDEO_LOADER_TYPE="$(yaml_get preprocess_video_loader_type)"
VIDEO_BATCH_SIZE="$(yaml_get preprocess_video_batch_size)"
DATALOADER_NUM_WORKERS="$(yaml_get preprocess_dataloader_num_workers)"
MAX_HEIGHT="$(yaml_get max_height)"
MAX_WIDTH="$(yaml_get max_width)"
NUM_FRAMES="$(yaml_get num_frames)"
TRAIN_FPS="$(yaml_get train_fps)"
VIDEO_LENGTH_TOLERANCE_RANGE="$(yaml_get video_length_tolerance_range)"

REPORTS_DIR="${DATA_ROOT}/reports"
PRECOMPUTED_REPORT_JSON="${REPORTS_DIR}/precompute_summary.json"
RAW_REPORT_JSON="${REPORTS_DIR}/raw_summary.json"

mkdir -p "${REPORTS_DIR}"

echo "[INFO] CONFIG_PATH=${CONFIG_PATH}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] NUM_GPUS_PREPROCESS=${NUM_GPUS_PREPROCESS}"
echo "[INFO] WITH_AUDIO=${WITH_AUDIO}"

# Work around current Gemma connector shape bug in preprocess path:
# batch size > 1 can fail in _replace_padded_with_learnable_registers.
if [[ "${VIDEO_BATCH_SIZE}" -gt 1 ]]; then
  echo "[WARN] Overriding preprocess_video_batch_size=${VIDEO_BATCH_SIZE} -> 1 due to connector limitation"
  VIDEO_BATCH_SIZE=1
fi

conda run --no-capture-output -n "${PYTHON_ENV}" python -u \
  "${SCRIPT_DIR}/validate_dataset.py" \
  --mode raw \
  --data-root "${DATA_ROOT}" \
  --output-report "${RAW_REPORT_JSON}"

conda run --no-capture-output -n "${PYTHON_ENV}" \
  torchrun --nproc_per_node="${NUM_GPUS_PREPROCESS}" \
  --master_port="${PREPROCESS_MASTER_PORT}" \
  -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
  --model_path "${MODEL_PATH}" \
  --mode preprocess \
  --workload_type t2v \
  --preprocess.video_loader_type "${VIDEO_LOADER_TYPE}" \
  --preprocess.dataset_type merged \
  --preprocess.dataset_path "${DATA_ROOT}" \
  --preprocess.dataset_output_dir "${DATA_ROOT}" \
  --preprocess.with_audio "${WITH_AUDIO}" \
  --preprocess.preprocess_video_batch_size "${VIDEO_BATCH_SIZE}" \
  --preprocess.dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --preprocess.max_height "${MAX_HEIGHT}" \
  --preprocess.max_width "${MAX_WIDTH}" \
  --preprocess.num_frames "${NUM_FRAMES}" \
  --preprocess.train_fps "${TRAIN_FPS}" \
  --preprocess.video_length_tolerance_range "${VIDEO_LENGTH_TOLERANCE_RANGE}"

conda run --no-capture-output -n "${PYTHON_ENV}" python -u \
  "${SCRIPT_DIR}/validate_dataset.py" \
  --mode precomputed \
  --data-root "${DATA_ROOT}" \
  --output-report "${PRECOMPUTED_REPORT_JSON}"

echo "[DONE] Precomputed dataset is ready."
echo "[DONE] Report: ${PRECOMPUTED_REPORT_JSON}"
