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
# first. Its checkpoint-N/ is a raw DCP checkpoint (dcp/ + metadata/RNG
# state only) -- models.student/teacher/critic.init_from need a diffusers
# model directory (model_index.json + component subfolders), so convert it
# first:
#
#   python -m fastvideo.train.entrypoint.dcp_to_diffusers \
#       --checkpoint outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N> \
#       --output-dir outputs/kandinsky5_t2v_qat_finetune/checkpoint-<N>-diffusers \
#       --role student --verify
#
# then point models.student/teacher/critic.init_from in dmd2_t2v_480p_qat.yaml
# at that exported checkpoint-<N>-diffusers directory (all three roles use
# the same weights; teacher/critic full-precision masking happens at load
# time, not by pointing at a different export) before launching this script.
set -euo pipefail

export FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_TRAIN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/dmd2_t2v_480p_qat.yaml" \
    "$@"
