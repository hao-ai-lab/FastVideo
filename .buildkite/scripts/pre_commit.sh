#!/bin/bash
set -uo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Starting pre-commit checks ==="

cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
log "Project root: $PROJECT_ROOT"

if ! python3 -m pre_commit --version &> /dev/null; then
    log "pre-commit not found, installing..."
    python3 -m pip install --user pre-commit==4.0.1
    
    if ! python3 -m pre_commit --version &> /dev/null; then
        log "Error: Failed to install pre-commit."
        exit 1
    fi
fi

log "Pre-commit version: $(python3 -m pre_commit --version)"

log "Installing/updating pre-commit hooks..."
python3 -m pre_commit install --install-hooks

log "Running pre-commit checks on all files..."
python3 -m pre_commit run --all-files
PRE_COMMIT_EXIT_CODE=$?

if [ $PRE_COMMIT_EXIT_CODE -eq 0 ]; then
    log "Pre-commit checks completed successfully"
else
    log "Error: Pre-commit checks failed with exit code: $PRE_COMMIT_EXIT_CODE"
fi

log "=== Pre-commit checks completed with exit code: $PRE_COMMIT_EXIT_CODE ==="
exit $PRE_COMMIT_EXIT_CODE 