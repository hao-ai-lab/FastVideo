#!/usr/bin/env bash
# Seed SSIM reference videos for a new test file.
#
# Runs the SSIM test on Modal with --sync-generated-to-volume, then prints the
# follow-up download / copy-local / upload commands so the operator can
# finish the seed. Designed to be safe to re-run: the Modal subdir is
# derived from the git commit + timestamp, so repeated runs don't overwrite
# each other.
#
# Usage:
#   seed_ssim.sh --test-file fastvideo/tests/ssim/test_ltx2_similarity.py \
#                [--quality-tier default|full_quality] \
#                [--model-ids Model1,Model2] \
#                [--device-folder L40S_reference_videos] \
#                [--git-repo URL] [--git-commit SHA]

set -euo pipefail

TEST_FILE=""
QUALITY_TIER="default"
MODEL_IDS=""
DEVICE_FOLDER="L40S_reference_videos"
GIT_REPO=""
GIT_COMMIT=""

usage() {
    sed -n '1,/^$/p' "$0" | sed -e 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --quality-tier)
            QUALITY_TIER="$2"
            shift 2
            ;;
        --model-ids)
            MODEL_IDS="$2"
            shift 2
            ;;
        --device-folder)
            DEVICE_FOLDER="$2"
            shift 2
            ;;
        --git-repo)
            GIT_REPO="$2"
            shift 2
            ;;
        --git-commit)
            GIT_COMMIT="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage 1
            ;;
    esac
done

if [[ -z "$TEST_FILE" ]]; then
    echo "Missing --test-file" >&2
    usage 1
fi
if [[ ! -f "$TEST_FILE" ]]; then
    echo "Test file does not exist: $TEST_FILE" >&2
    exit 2
fi

case "$QUALITY_TIER" in
    default|full_quality) ;;
    *)
        echo "quality_tier must be 'default' or 'full_quality', got: $QUALITY_TIER" >&2
        exit 2
        ;;
esac

if ! grep -q '^REQUIRED_GPUS\s*=' "$TEST_FILE"; then
    echo "Warning: $TEST_FILE has no REQUIRED_GPUS constant; Modal will schedule it as 1 GPU." >&2
fi
if ! grep -q 'MODEL_TO_PARAMS' "$TEST_FILE"; then
    echo "Warning: $TEST_FILE has no *_MODEL_TO_PARAMS dict; model_id splitting will be disabled." >&2
fi

HF_TOKEN="${HF_API_KEY:-${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_API_KEY / HUGGINGFACE_HUB_TOKEN / HF_TOKEN not set." >&2
    echo "       The upload step (step 5 in SKILL.md) will fail without it." >&2
    exit 3
fi

if [[ -z "$GIT_REPO" ]]; then
    GIT_REPO="$(git config --get remote.origin.url)"
fi
if [[ -z "$GIT_COMMIT" ]]; then
    GIT_COMMIT="$(git rev-parse HEAD)"
fi

SHORT_COMMIT="${GIT_COMMIT:0:12}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
SUBDIR="${TIMESTAMP}_${SHORT_COMMIT}"
VOLUME_PATH="ssim_generated_videos/${QUALITY_TIER}/${SUBDIR}/generated_videos"
LOCAL_DOWNLOAD="./generated_videos_modal/${QUALITY_TIER}"

FULL_QUALITY_FLAG=""
if [[ "$QUALITY_TIER" == "full_quality" ]]; then
    FULL_QUALITY_FLAG="--full-quality"
fi

MODEL_IDS_FLAG=""
if [[ -n "$MODEL_IDS" ]]; then
    MODEL_IDS_FLAG="--model-ids=$MODEL_IDS"
fi

echo "=========================================================="
echo "Seeding SSIM references:"
echo "  test_file      = $TEST_FILE"
echo "  quality_tier   = $QUALITY_TIER"
echo "  git_commit     = $GIT_COMMIT"
echo "  volume_subdir  = $SUBDIR"
echo "  device_folder  = $DEVICE_FOLDER"
echo "=========================================================="

modal run fastvideo/tests/modal/ssim_test.py \
    --git-repo="$GIT_REPO" \
    --git-commit="$GIT_COMMIT" \
    --hf-api-key="$HF_TOKEN" \
    --test-files="$TEST_FILE" \
    $MODEL_IDS_FLAG \
    --sync-generated-to-volume \
    --generated-volume-subdir="$SUBDIR" \
    --skip-reference-download \
    --no-fail-fast \
    $FULL_QUALITY_FLAG

echo
echo "Modal run finished. Next steps (copy into a shell with modal + HF creds):"
echo
echo "  # 1. Pull the generated videos off the Modal volume"
echo "  modal volume get hf-model-weights \\"
echo "    $VOLUME_PATH \\"
echo "    $LOCAL_DOWNLOAD"
echo
echo "  # 2. Stage them into the reference_videos/ layout"
echo "  python fastvideo/tests/ssim/reference_videos_cli.py copy-local \\"
echo "    --quality-tier $QUALITY_TIER \\"
echo "    --generated-dir $LOCAL_DOWNLOAD/$DEVICE_FOLDER \\"
echo "    --device-folder $DEVICE_FOLDER"
echo
echo "  # 3. Upload to HF (reads HF_API_KEY from env)"
echo "  python fastvideo/tests/ssim/reference_videos_cli.py upload \\"
echo "    --quality-tier $QUALITY_TIER \\"
echo "    --device-folder $DEVICE_FOLDER"
echo
echo "  # 4. Verify by re-running the test without --skip-reference-download"
echo "  modal run fastvideo/tests/modal/ssim_test.py \\"
echo "    --test-files=$TEST_FILE"
