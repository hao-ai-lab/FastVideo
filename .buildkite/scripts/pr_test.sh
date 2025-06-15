#!/bin/bash
set -uo pipefail

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=== Starting Modal test execution ==="

# Change to the project directory
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
log "Project root: $PROJECT_ROOT"

# Install Modal if not available
if ! python3 -m modal --version &> /dev/null; then
    log "Modal not found, installing..."
    pip install modal
    
    # Verify installation
    if ! python3 -m modal --version &> /dev/null; then
        log "Error: Failed to install modal. Please install it manually."
        exit 1
    fi
fi

log "modal version: $(python3 -m modal --version)"

# Set up Modal authentication using Buildkite secrets
log "Setting up Modal authentication from Buildkite secrets..."
MODAL_TOKEN_ID=$(buildkite-agent secret get modal_token_id)
MODAL_TOKEN_SECRET=$(buildkite-agent secret get modal_token_secret)

if [ -n "$MODAL_TOKEN_ID" ] && [ -n "$MODAL_TOKEN_SECRET" ]; then
    log "Retrieved Modal credentials from Buildkite secrets"
    python3 -m modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" --profile buildkite-ci --activate --verify
    if [ $? -eq 0 ]; then
        log "Modal authentication successful"
    else
        log "Error: Failed to set Modal credentials"
        exit 1
    fi
else
    log "Error: Could not retrieve Modal credentials from Buildkite secrets."
    log "Please ensure 'modal_token_id' and 'modal_token_secret' secrets are set in Buildkite."
    exit 1
fi

MODAL_TEST_FILE="modal/pr_test.py"

if [ -z "${TEST_TYPE:-}" ]; then
    log "Error: TEST_TYPE environment variable is not set"
    exit 1
fi
log "Test type: $TEST_TYPE"

case "$TEST_TYPE" in
    "encoder")
        log "Running encoder tests..."
        MODAL_COMMAND="python3 -m modal run modal/pr_test.py::run_encoder_tests"
        ;;
    "vae")
        log "Running VAE tests..."
        MODAL_COMMAND="python3 -m modal run modal/pr_test.py::run_vae_tests"
        ;;
    "transformer")
        log "Running transformer tests..."
        MODAL_COMMAND="python3 -m modal run modal/pr_test.py::run_transformer_tests"
        ;;
    "ssim")
        log "Running SSIM tests..."
        MODAL_COMMAND="python3 -m modal run modal/pr_test.py::run_ssim_tests"
        ;;
    *)
        log "Error: Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

log "Executing: $MODAL_COMMAND"
eval "$MODAL_COMMAND"
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log "Modal test completed successfully"
else
    log "Error: Modal test failed with exit code: $TEST_EXIT_CODE"
fi

log "=== Test execution completed with exit code: $TEST_EXIT_CODE ==="
exit $TEST_EXIT_CODE
