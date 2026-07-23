#!/bin/bash
# QAD recipe -- quantization-aware finetune of Kandinsky5 T2V (480p) with
# fake-quant (Attn-QAT) attention.
#
# The YAML's role-local ATTN_QAT_TRAIN backend routes Kandinsky5's dense/local
# attention through the fake-quantized
# straight-through-estimator kernel, so the DiT learns to absorb the
# quantization error instead of fighting it. No weight quantization happens
# during this stage; that's applied post-hoc at inference time (see the
# stage-2 config and README).
#
# Scope: T2V, 480p only, dense/local attention only -- NABLA sparse attention
# is never engaged by Kandinsky5Model at this resolution.
#
# Data: preprocess with fastvideo/pipelines/preprocess/preprocess_kandinsky5_overfit.py
# (or an equivalent parquet dataset matching pyarrow_schema_t2v) first.
#
# Next: this writes a raw DCP checkpoint (checkpoint-N/dcp + metadata/RNG
# state), not something stage 2 can load directly. Pass that checkpoint to
# ../../distribution_matching/kandinsky5/distill_dmd_qat.sh, which converts
# it with fastvideo.train.entrypoint.dcp_to_diffusers (--verify) and feeds
# the export to all three models.*.init_from overrides automatically.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

bash "${REPO_ROOT}/examples/train/run.sh" \
    "${SCRIPT_DIR}/t2v_480p_qat.yaml" \
    "$@"
