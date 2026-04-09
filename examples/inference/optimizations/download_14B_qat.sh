#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

HF_REPO_ID="${HF_REPO_ID:-FastVideo/14B_qat_400}"
HF_REVISION="${HF_REVISION:-main}"
LOCAL_DIR="${1:-${REPO_ROOT}/checkpoints/14B_qat_400}"
PYTHON_BIN="${PYTHON:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "Missing dependency: huggingface_hub" >&2
    echo "Install it with: uv pip install huggingface_hub" >&2
    exit 1
fi

mkdir -p "${LOCAL_DIR}"

echo "Downloading ${HF_REPO_ID}@${HF_REVISION}"
echo "Local directory: ${LOCAL_DIR}"

"${PYTHON_BIN}" -c '
import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
parser.add_argument("--revision", required=True)
parser.add_argument("--local-dir", required=True)
args = parser.parse_args()

snapshot_download(
    repo_id=args.repo_id,
    revision=args.revision,
    repo_type="model",
    local_dir=args.local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
' \
    --repo-id "${HF_REPO_ID}" \
    --revision "${HF_REVISION}" \
    --local-dir "${LOCAL_DIR}"

echo
echo "Download complete."
echo "Use this in your inference script:"
echo "init_weights_from_safetensors=\"${LOCAL_DIR}\""
echo
echo "If the repo is private or gated, make sure you are logged in with:"
echo "huggingface-cli login"
