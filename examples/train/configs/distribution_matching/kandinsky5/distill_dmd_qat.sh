#!/bin/bash
# Stage-2 QAT-aware DMD2 distillation for Kandinsky5 T2V (480p).
#
# Usage:
#   bash distill_dmd_qat.sh <stage1-checkpoint> [--dotted.key value ...]
#
# <stage1-checkpoint> is either:
#   - a raw stage-1 DCP checkpoint written by
#     examples/train/configs/fine_tuning/kandinsky5/finetune_qat.sh
#     (a checkpoint-<N>/ dir with a dcp/ subdir, or the stage-1 output_dir
#     itself -- dcp_to_diffusers auto-picks the latest checkpoint). It is
#     converted to a diffusers export (<checkpoint>-diffusers) with
#     `dcp_to_diffusers --role student --verify` before launching --
#     models.*.init_from need a diffusers model directory (model_index.json
#     + component subfolders), not the raw DCP layout; or
#   - an already-exported diffusers model dir (model_index.json present),
#     used as-is with no conversion.
#
# The export is passed to all three models.*.init_from overrides: student,
# teacher, and critic all load the same weights. The YAML assigns
# ATTN_QAT_TRAIN to the student and FLASH_ATTN to teacher/critic; the loader's
# _loading_teacher_critic_model gate also keeps their weights full precision.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

CHECKPOINT="${1:?Usage: $0 <stage1-checkpoint (DCP dir or diffusers export)> [extra --dotted.key overrides...]}"
shift

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="$(pwd)/${CHECKPOINT}"
fi
CHECKPOINT="${CHECKPOINT%/}"

if [[ -f "${CHECKPOINT}/model_index.json" ]]; then
    # Already a diffusers export -- use as-is.
    INIT_FROM="${CHECKPOINT}"
else
    # Convert the raw DCP checkpoint before launching training. --overwrite keeps relaunches
    # from silently reusing a stale export for a newer checkpoint;
    # --verify strictly reloads the exported transformer so a key-mapping
    # bug fails here, not deep inside the training launch below.
    INIT_FROM="${CHECKPOINT}-diffusers"
    echo "Converting stage-1 DCP checkpoint ${CHECKPOINT} -> ${INIT_FROM}"
    python -m fastvideo.train.entrypoint.dcp_to_diffusers \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${INIT_FROM}" \
        --role student \
        --overwrite \
        --verify
fi

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/dmd2_t2v_480p_qat.yaml" \
    --models.student.init_from "${INIT_FROM}" \
    --models.teacher.init_from "${INIT_FROM}" \
    --models.critic.init_from "${INIT_FROM}" \
    "$@"
