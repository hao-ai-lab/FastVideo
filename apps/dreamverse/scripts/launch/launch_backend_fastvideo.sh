#!/usr/bin/env bash
# Launch the FastVideo streaming backend via the typed
# ``fastvideo serve --config`` entrypoint, driven by
# ``serve_configs/streaming_demo.yaml``.
#
# Usage:
#   bash launch_backend_fastvideo.sh
#   bash launch_backend_fastvideo.sh --server.port 8010 --streaming.warmup.enabled false
#
# Anything passed after the script name is forwarded verbatim to
# ``fastvideo serve``, so dotted overrides like
# ``--server.port 8010`` or ``--streaming.warmup.enabled false`` work
# without a parallel flag scheme.
#
# Caveat: the bare ``fastvideo serve`` build_app exposes ``/health``
# and ``/v1/stream`` only. Dreamverse's existing Next.js shell also
# expects ``/healthz``, ``/readyz``, and ``/curated-presets`` which
# live in dreamverse-server. Use ``launch_backend_dreamverse.sh`` for
# the full FE compatibility path; this script is for verifying the
# typed serve config and bare-streaming-server flow.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SERVE_CONFIG="${DREAMVERSE_ROOT}/serve_configs/streaming_demo.yaml"

if [[ ! -f "${SERVE_CONFIG}" ]]; then
  echo "error: serve config not found at ${SERVE_CONFIG}" >&2
  exit 1
fi

# Source ~/.env if present so prompt-enhancer credentials
# (CEREBRAS_API_KEY etc.) are visible to the worker process.
if [[ -f "${HOME}/.env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "${HOME}/.env"
  set +o allexport
fi

# Match internal/ui's attention backend default (gpu_pool.py:161).
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"

cd "${DREAMVERSE_ROOT}"

echo "[launch-demo] starting fastvideo serve"
echo "  config: ${SERVE_CONFIG}"
echo "  overrides: $*"
exec uv run fastvideo serve --config "${SERVE_CONFIG}" "$@"
