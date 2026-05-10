#!/usr/bin/env bash
# Launch the Dreamverse Next.js frontend in dev mode on port 5274
# (the devtools-enabled build the e2e tests target).
#
# Usage:
#   bash launch_frontend.sh
#   FRONTEND_MODE=dev bash launch_frontend.sh        # plain dev (5299, no devtools)
#   FRONTEND_MODE=single5s bash launch_frontend.sh   # single-5s product mode
#
# The script ``cd``'s into ``web`` and shells out to pnpm. It runs
# ``pnpm install --frozen-lockfile`` only when ``node_modules/`` is missing so repeat
# launches are fast.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DREAMVERSE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WEB_ROOT="${DREAMVERSE_ROOT}/web"

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

cd "${WEB_ROOT}"

if [[ ! -d node_modules ]]; then
  echo "[launch-demo] node_modules missing — running pnpm install --frozen-lockfile"
  pnpm install --frozen-lockfile
fi

case "${FRONTEND_MODE}" in
  devtools)
    echo "[launch-demo] starting Next.js dev:devtools (port 5274)"
    exec pnpm run dev:devtools -- "$@"
    ;;
  dev)
    echo "[launch-demo] starting Next.js dev (port 5299)"
    exec pnpm run dev -- "$@"
    ;;
  single5s)
    echo "[launch-demo] starting Next.js dev:single5s (port 5274)"
    exec pnpm run dev:single5s -- "$@"
    ;;
esac
