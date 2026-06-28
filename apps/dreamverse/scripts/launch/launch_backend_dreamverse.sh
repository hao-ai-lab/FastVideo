#!/usr/bin/env bash
# Launch the Dreamverse-flavored backend via the installed console command.
#
# This is the path the Next.js frontend expects today — it serves
# ``/healthz``, ``/readyz``, ``/status``, ``/curated-presets``, and the
# devtools-only routes that ``fastvideo serve`` does not. Use this for
# the full demo experience until the FE-only routes migrate into
# ``fastvideo.entrypoints.streaming.server.build_app``.
#
# Usage:
#   bash launch_backend_dreamverse.sh                 # default 0.0.0.0:8009
#   bash launch_backend_dreamverse.sh --port 8010
#
# Mirrors internal/ui defaults via the same env variables internal's
# ``config.py`` reads (see ../../../serve_configs/streaming_demo.yaml
# for the canonical list and source-of-truth comments).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${HOME}/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "${HOME}/.env"
  set +o allexport
fi

# Internal/ui parity defaults — only set if the caller hasn't already
# pinned them in ~/.env or the surrounding environment. Each value
# matches FastVideo-internal/ui/ltx2-streaming/server/config.py.
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export STREAM_MODE="${STREAM_MODE:-av_fmp4}"
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-1}"
export FASTVIDEO_ENABLE_STARTUP_WARMUP="${FASTVIDEO_ENABLE_STARTUP_WARMUP:-1}"
export FASTVIDEO_STARTUP_WARMUP_TIMEOUT_SECONDS="${FASTVIDEO_STARTUP_WARMUP_TIMEOUT_SECONDS:-2400}"
export FASTVIDEO_GENERATION_SEGMENT_CAP="${FASTVIDEO_GENERATION_SEGMENT_CAP:-6}"
export FASTVIDEO_PROMPT_AUTO_SLEEP_MS="${FASTVIDEO_PROMPT_AUTO_SLEEP_MS:-120}"
export FASTVIDEO_PROMPT_AUTO_TIMEOUT_MS="${FASTVIDEO_PROMPT_AUTO_TIMEOUT_MS:-1800}"

cd "${DREAMVERSE_ROOT}"

if ! command -v dreamverse-server >/dev/null 2>&1; then
  echo "error: dreamverse-server not found on PATH. Install FastVideo with the dreamverse extra." >&2
  exit 1
fi

echo "[launch-demo] starting dreamverse-server"
echo "  args: $*"
exec dreamverse-server "$@"
