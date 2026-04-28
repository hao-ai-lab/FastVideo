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
MODAL_SSIM_TEST_FILE="fastvideo/tests/modal/ssim_test.py"

if [ -z "${TEST_TYPE:-}" ]; then
    log "Error: TEST_TYPE environment variable is not set"
    exit 1
fi
log "Test type: $TEST_TYPE"

EFFECTIVE_PR=${BUILDKITE_PULL_REQUEST:-false}
if [ "$EFFECTIVE_PR" = "false" ] && [ -n "${PR_NUMBER:-}" ]; then
    EFFECTIVE_PR=$PR_NUMBER
fi
MODAL_ENV="BUILDKITE_REPO=$BUILDKITE_REPO BUILDKITE_COMMIT=$BUILDKITE_COMMIT BUILDKITE_PULL_REQUEST=$EFFECTIVE_PR BUILDKITE_BRANCH=${BUILDKITE_BRANCH:-} TEST_SCOPE=${TEST_SCOPE:-} IMAGE_VERSION=$IMAGE_VERSION"

POST_RUN_HOOK=""

upload_performance_artifacts() {
    SHORT_SHA=${BUILDKITE_COMMIT:0:7}
    LOCAL_DIR="downloaded_reports"

    _download_reports() {
        log "Downloading perf_reports/ from Modal Volume..."
        mkdir -p "$LOCAL_DIR"
        if ! modal volume get hf-model-weights "perf_reports/" "$LOCAL_DIR"; then
            log "Error: Failed to download perf_reports/ from Modal Volume."
            return 1
        fi
    }

    _upload_dashboard() {
        local target
        target=$(find "$LOCAL_DIR" -name "dashboard_*${SHORT_SHA}*" | head -n 1)
        log "TARGET dashboard: '$target'"

        if [ -n "$target" ]; then
            log "Found dashboard: $target. Uploading to Buildkite..."
            buildkite-agent artifact upload "$target"
            buildkite-agent annotate --style info --context "perf-dashboard" < "$target"
        else
            log "Warning: Could not find a dashboard file matching $SHORT_SHA"
        fi
    }

    _upload_perf_summary() {
        local target
        target=$(find "$LOCAL_DIR" -name "perf_*${SHORT_SHA}*" | head -n 1)
        log "TARGET perf summary: '$target'"

        if [ -n "$target" ]; then
            log "Found perf summary: $target. Uploading to Buildkite..."
            buildkite-agent artifact upload "$target"
            buildkite-agent annotate --style info --context "perf-summary" < "$target"
        else
            log "Warning: Could not find a perf summary file matching $SHORT_SHA"
        fi
    }

    _cleanup_modal_volume() {
        log "Cleaning up perf_reports/ from Modal Volume..."
        if modal volume rm hf-model-weights "perf_reports/" --recursive; then
            log "Successfully deleted perf_reports/ from Modal Volume."
        else
            log "Warning: Failed to delete perf_reports/ from Modal Volume. Manual cleanup may be required."
        fi
    }

    _cleanup_local() {
        log "Cleaning up local download directory..."
        rm -rf "$LOCAL_DIR"
    }

    # --- Main flow ---
    _download_reports || { _cleanup_local; return 1; }
    _upload_dashboard
    _upload_perf_summary
    _cleanup_modal_volume
    _cleanup_local
}

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
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_SSIM_TEST_FILE::run_ssim_tests"
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
    "kernel_tests")
        log "Running kernel tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_kernel_tests"
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
    "unit_test")
        log "Running unit tests..."
        MODAL_COMMAND="$MODAL_ENV python3 -m modal run $MODAL_TEST_FILE::run_unit_test"
        ;;
    "lora_extraction")
        log "Running LoRA extraction tests..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_lora_extraction_tests"
        ;;
    "performance")
        log "Running performance tests on Modal..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_performance_tests"
        POST_RUN_HOOK="upload_performance_artifacts"
        ;;
    "api_server")
        log "Running API server integration tests..."
        MODAL_COMMAND="$MODAL_ENV HF_API_KEY=$HF_API_KEY python3 -m modal run $MODAL_TEST_FILE::run_api_server_tests"
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

if [ -n "$POST_RUN_HOOK" ]; then
    log "Executing post-run hook: $POST_RUN_HOOK"
    "$POST_RUN_HOOK"
fi

log "=== Test execution completed with exit code: $TEST_EXIT_CODE ==="
exit $TEST_EXIT_CODE
