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
    python3 -m pip install modal
    
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

# Retrieve other secrets
WANDB_API_KEY=$(buildkite-agent secret get wandb_api_key)
HF_API_KEY=$(buildkite-agent secret get hf_api_key)

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

MODAL_TEST_FILE="fastvideo/tests/modal/pr_test.py"

if [ -z "${TEST_TYPE:-}" ]; then
    log "Error: TEST_TYPE environment variable is not set"
    exit 1
fi
log "Test type: $TEST_TYPE"

MODAL_ENV="BUILDKITE_REPO=$BUILDKITE_REPO BUILDKITE_COMMIT=$BUILDKITE_COMMIT BUILDKITE_PULL_REQUEST=$BUILDKITE_PULL_REQUEST IMAGE_VERSION=$IMAGE_VERSION"

case "$TEST_TYPE" in
    "encoder")
        log "Running encoder tests..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_encoder_tests"
        ;;
    "vae")
        log "Running VAE tests..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_vae_tests"
        ;;
    "transformer")
        log "Running transformer tests..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_transformer_tests"
        ;;
    "ssim")
        log "Running SSIM tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_ssim_tests"
        ;;
    "training")
        log "Running training tests..."
        MODAL_COMMAND="$MODAL_ENV WANDB_API_KEY=$WANDB_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_training_tests"
        ;;
    "training_lora")
        log "Running LoRA training tests..."
        MODAL_COMMAND="$MODAL_ENV WANDB_API_KEY=$WANDB_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_training_lora_tests"
        ;;
    "training_vsa")
        log "Running training VSA tests..."
        MODAL_COMMAND="$MODAL_ENV WANDB_API_KEY=$WANDB_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_training_tests_VSA"
        ;;
    "inference_sta")
        log "Running inference STA tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_inference_tests_STA"
        ;;
    "precision_sta")
        log "Running precision STA tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_precision_tests_STA"
        ;;
    "precision_vsa")
        log "Running precision VSA tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_precision_tests_VSA"
        ;;
    "inference_lora")
        log "Running LoRA tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_inference_lora_tests"
        ;;
    "distillation_dmd")
        log "Running distillation DMD tests..."
        MODAL_COMMAND="$MODAL_ENV WANDB_API_KEY=$WANDB_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_distill_dmd_tests"
        ;;
        # run_inference_tests_vmoba
    "self_forcing")
        log "Running self-forcing tests..."
        MODAL_COMMAND="$MODAL_ENV WANDB_API_KEY=$WANDB_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_self_forcing_tests"
        ;;
    "inference_vmoba")
        log "Running V-MoBA inference tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_inference_tests_vmoba"
        ;;
    "precision_vmoba")
        log "Running V-MoBA precision tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_precision_tests_vmoba"
        ;;
    "unit_test")
        log "Running unit tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_unit_test"
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
