#!/usr/bin/env bash
set -euo pipefail

: "${CEREBRAS_API_KEY:?CEREBRAS_API_KEY not set on host}"
: "${GROQ_API_KEY:?GROQ_API_KEY not set on host}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${DREAMVERSE_IMAGE:-dreamverse:dev}"
PORT="${BACKEND_PORT:-8009}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUTS_DIR="${DREAMVERSE_OUTPUTS_DIR:-${DREAMVERSE_ROOT}/outputs}"
NAME="${DREAMVERSE_NAME:-dreamverse}"
TIMEOUT_SECONDS="${DREAMVERSE_SMOKE_TIMEOUT_SECONDS:-1200}"
POLL_SECONDS="${DREAMVERSE_SMOKE_POLL_SECONDS:-5}"
GPU_REQUEST="${DREAMVERSE_DOCKER_GPUS:-all}"

mkdir -p "${HF_CACHE}" "${OUTPUTS_DIR}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

env_args=(
  -e "CEREBRAS_API_KEY=${CEREBRAS_API_KEY}"
  -e "GROQ_API_KEY=${GROQ_API_KEY}"
  -e "ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-0}"
)
[[ -n "${FASTVIDEO_GPU_COUNT:-}" ]] && env_args+=(-e "FASTVIDEO_GPU_COUNT=${FASTVIDEO_GPU_COUNT}")

container_id="$(
  docker run -d --rm --gpus "${GPU_REQUEST}" --init \
    -p "${PORT}:8009" \
    "${env_args[@]}" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    -v "${OUTPUTS_DIR}:/var/lib/dreamverse/outputs" \
    --name "${NAME}" \
    "${IMAGE}"
)"

cleanup() {
  if [[ "${DREAMVERSE_KEEP_CONTAINER:-0}" != "1" ]]; then
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_endpoint() {
  local path="$1"
  local label="$2"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local url="http://127.0.0.1:${PORT}${path}"

  echo "Waiting for ${label} at ${url}"
  while (( SECONDS < deadline )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${label} ok"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
      echo "Container exited before ${label} became healthy." >&2
      docker logs "${container_id}" >&2 || true
      return 1
    fi
    sleep "${POLL_SECONDS}"
  done

  echo "Timed out waiting for ${label}." >&2
  docker logs "${container_id}" >&2 || true
  return 1
}

wait_for_endpoint "/healthz" "healthz"
wait_for_endpoint "/readyz" "readyz"

echo "Dreamverse Docker smoke passed for ${IMAGE} on host port ${PORT}."
