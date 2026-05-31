#!/usr/bin/env bash
set -euo pipefail

source /opt/venv/bin/activate

if [[ -f /opt/FastVideo/apps/dreamverse/scripts/ffmpeg-env.sh ]]; then
  source /opt/FastVideo/apps/dreamverse/scripts/ffmpeg-env.sh
fi

: "${CEREBRAS_API_KEY:?CEREBRAS_API_KEY must be set (pass with -e CEREBRAS_API_KEY=...)}"
: "${GROQ_API_KEY:?GROQ_API_KEY must be set (pass with -e GROQ_API_KEY=...)}"

exec "$@"
