#!/usr/bin/env bash
# launch-frontend.sh — start the Dreamverse Next.js dev server and ngrok tunnel.
#
# Usage (from repo root):
#   bash apps/dreamverse/scripts/launch-frontend.sh
#
# Override backend or ngrok URL via env:
#   BACKEND_HOST=1.2.3.4 bash apps/dreamverse/scripts/launch-frontend.sh

set -euo pipefail

CONDA_PREFIX="$HOME/miniconda3/envs/dreamverse"
BACKEND_HOST="${BACKEND_HOST:-10.244.18.228}"
BACKEND_PORT="${BACKEND_PORT:-8009}"
NGROK_URL="${NGROK_URL:-ltx23.ngrok.app}"
WEB_DIR="$(git rev-parse --show-toplevel)/apps/dreamverse/web"

cleanup() {
    echo "==> Shutting down..."
    kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "==> Starting frontend (backend: $BACKEND_HOST:$BACKEND_PORT)"
BACKEND_HOST="$BACKEND_HOST" BACKEND_PORT="$BACKEND_PORT" \
    npm run --prefix "$WEB_DIR" dev &
FRONTEND_PID=$!

echo "==> Starting ngrok tunnel -> $NGROK_URL"
"$CONDA_PREFIX/bin/ngrok" http --url="$NGROK_URL" 5299
