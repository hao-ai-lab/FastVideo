#!/usr/bin/env bash
# Launch the Dreamverse Next.js frontend in dev mode. All modes default to
# port 5299; set FE_PORT to override it.
#
# Usage:
#   bash launch_frontend.sh
#   FE_PORT=5300 bash launch_frontend.sh              # devtools on a custom port
#   FRONTEND_MODE=dev bash launch_frontend.sh         # plain dev (no devtools)
#   FRONTEND_MODE=single5s bash launch_frontend.sh   # single-5s product mode
#
# The script ``cd``'s into ``web`` and shells out to npm. It runs
# ``npm ci`` only when ``node_modules/`` is missing so repeat
# launches are fast.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WEB_ROOT="${DREAMVERSE_ROOT}/web"
FE_PORT="${FE_PORT:-5299}"
NPM="${NPM:-npm}"

if ! [[ "${FE_PORT}" =~ ^[0-9]+$ ]] || (( FE_PORT < 1 || FE_PORT > 65535 )); then
  echo "error: FE_PORT must be an integer from 1 to 65535 (got '${FE_PORT}')" >&2
  exit 1
fi

FRONTEND_MODE="${FRONTEND_MODE:-devtools}"
case "${FRONTEND_MODE}" in
  devtools|dev|single5s) ;;
  *)
    echo "error: FRONTEND_MODE must be one of devtools|dev|single5s (got '${FRONTEND_MODE}')" >&2
    exit 1
    ;;
esac

if [[ ! -d "${WEB_ROOT}" ]]; then
  echo "error: web app not found at ${WEB_ROOT}" >&2
  exit 1
fi

command -v "${NPM}" >/dev/null 2>&1 || {
  echo "error: npm command not found: ${NPM}" >&2
  exit 1
}

cd "${WEB_ROOT}"

if [[ ! -d node_modules ]]; then
  echo "[launch-demo] node_modules missing — running npm ci"
  "${NPM}" ci
fi

case "${FRONTEND_MODE}" in
  devtools)
    unset NEXT_PUBLIC_PRODUCT_MODE
    export NEXT_PUBLIC_INCLUDE_DEVTOOLS=1
    echo "[launch-demo] starting Next.js devtools mode (port ${FE_PORT})"
    exec "${NPM}" exec -- next dev --port "${FE_PORT}" "$@"
    ;;
  dev)
    unset NEXT_PUBLIC_INCLUDE_DEVTOOLS NEXT_PUBLIC_PRODUCT_MODE
    echo "[launch-demo] starting Next.js dev mode (port ${FE_PORT})"
    exec "${NPM}" exec -- next dev --port "${FE_PORT}" "$@"
    ;;
  single5s)
    unset NEXT_PUBLIC_INCLUDE_DEVTOOLS
    export NEXT_PUBLIC_PRODUCT_MODE=single5s
    echo "[launch-demo] starting Next.js single5s mode (port ${FE_PORT})"
    exec "${NPM}" exec -- next dev --port "${FE_PORT}" "$@"
    ;;
esac
