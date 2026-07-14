#!/bin/bash
# Stage-2 QAT-aware DMD2 distillation for Kandinsky5 T2V (480p).
#
# FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN keeps the student's dense/local
# attention fake-quantized during distillation; the teacher/critic are
# automatically masked back to dense attention and full-precision weights by
# the _loading_teacher_critic_model gate in
# fastvideo/models/loader/component_loader.py (family-agnostic, no
# Kandinsky5-specific handling needed).
#
# Run examples/train/configs/fine_tuning/kandinsky5/finetune_qat.sh (stage 1)
# first and point models.student/teacher/critic.init_from in
# dmd2_t2v_480p_qat.yaml at that checkpoint before launching this script.
set -euo pipefail

export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/dmd2_t2v_480p_qat.yaml" \
    "$@"
