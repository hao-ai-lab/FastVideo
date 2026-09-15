#!/usr/bin/env bash
# Run inside an existing Slurm step. Slurm owns the GPU visibility and lifetime.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_STEP_ID:-}" ]]; then
  echo "Run this launcher inside an allocated Slurm step (srun), not on the login node." >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
python_bin="${DREAMVERSE_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  echo "Set DREAMVERSE_PYTHON to a Python environment with fastvideo[dreamverse] installed." >&2
  exit 2
fi

export DREAMVERSE_MODEL_ID="${DREAMVERSE_MODEL_ID:-full-h3}"
export DREAMVERSE_SP_SIZE="${DREAMVERSE_SP_SIZE:-4}"
export FASTVIDEO_GPU_COUNT="${FASTVIDEO_GPU_COUNT:-${DREAMVERSE_SP_SIZE}}"
export FASTVIDEO_ENABLE_STARTUP_WARMUP="${FASTVIDEO_ENABLE_STARTUP_WARMUP:-0}"
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"
export STREAM_MODE="${STREAM_MODE:-av_fmp4}"
export PYTHONPATH="${repo_root}/apps/dreamverse:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

"${python_bin}" - <<'PY'
import os
import shutil

import torch

expected = int(os.environ["DREAMVERSE_SP_SIZE"])
visible = torch.cuda.device_count()
if expected < 1 or visible < expected:
    raise SystemExit(f"The Slurm step exposes {visible} GPUs; DREAMVERSE_SP_SIZE requires {expected}.")
ffmpeg = os.environ.get("FASTVIDEO_FFMPEG_BIN", "ffmpeg")
if not shutil.which(ffmpeg):
    raise SystemExit("FFmpeg is missing; install it in the compute environment or set FASTVIDEO_FFMPEG_BIN.")
print(f"Slurm job {os.environ['SLURM_JOB_ID']}: {visible} visible GPUs; using {expected} per worker")
for index in range(expected):
    properties = torch.cuda.get_device_properties(index)
    print(f"  GPU {index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB")
PY

cd "${repo_root}"
exec "${python_bin}" -m dreamverse.server_entry \
  --host "${DREAMVERSE_BIND_HOST:-0.0.0.0}" \
  --port "${DREAMVERSE_BACKEND_PORT:-8009}" "$@"
