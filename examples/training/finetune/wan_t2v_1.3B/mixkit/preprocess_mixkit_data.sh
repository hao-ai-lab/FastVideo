#!/bin/bash
# Build the HD-Mixkit-Finetune-Wan parquet shards from raw MixKit videos.
#
# Mirrors download_mixkit_data.sh's *output* layout so finetune_qat.sh and
# distill_dmd_qat.sh can consume the result without any change. Use this when
# you want to reproduce the encoding from raw videos end-to-end, verify the
# published preprocessed dataset, or substitute your own captions.
#
# Pipeline:
#   FastVideo/Mixkit-Src (raw .mp4 + caption JSON)
#     -> v1_preprocessing_new (Wan2.1-T2V-1.3B VAE + text encoder)
#     -> ${DATA_ROOT}/{combined,validation}_parquet_dataset/
#
# Usage (run from the repo root):
#   bash examples/training/finetune/wan_t2v_1.3B/mixkit/preprocess_mixkit_data.sh \
#       [DATA_ROOT] [NUM_GPUS]
#
# Defaults: DATA_ROOT=data/HD-Mixkit-Finetune-Wan, NUM_GPUS=2.
#
# Requires roughly:
#   - ~25 GB free disk for the raw HF download (FastVideo/Mixkit-Src).
#   - ~6 GB for the parquet output (matches the published dataset size).
#   - 1+ GPUs for VAE + text encoding. NUM_GPUS=2 mirrors the README example;
#     scale via the second positional arg if you have more.
set -euo pipefail

DATA_ROOT=${1:-data/HD-Mixkit-Finetune-Wan}
NUM_GPUS=${2:-2}

# Raw HF dataset download dir and the staged layout the preprocess workflow
# expects (a sibling ``videos/`` folder and ``videos2caption.json`` file).
RAW_SRC_DIR="data/Mixkit-Src"
STAGE_DIR="data/mixkit_raw"

SRC_REPO="FastVideo/Mixkit-Src"
MODEL_PATH="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# 1. Pull the raw videos + caption JSON from the Hub.
if [ ! -d "${RAW_SRC_DIR}" ] || [ -z "$(ls -A "${RAW_SRC_DIR}" 2>/dev/null)" ]; then
    echo "[preprocess] downloading ${SRC_REPO} -> ${RAW_SRC_DIR}"
    python scripts/huggingface/download_hf.py \
        --repo_id "${SRC_REPO}" \
        --local_dir "${RAW_SRC_DIR}" \
        --repo_type "dataset"
else
    echo "[preprocess] using existing raw dataset at ${RAW_SRC_DIR}"
fi

# 2. Stage the layout expected by ``v1_preprocessing_new`` with
#    ``--preprocess.dataset_type merged``:
#
#        ${STAGE_DIR}/videos/              -> raw video tree (category subdirs)
#        ${STAGE_DIR}/videos2caption.json  -> caption list
#
#    The workflow joins ``preprocess_config.dataset_path`` with the literal
#    names ``videos`` and ``videos2caption.json``
#    (fastvideo/workflow/preprocess/components.py). ``Mixkit-Src`` already
#    organises clips into category subdirectories (Airplane/, Business/, ...)
#    whose paths match the entries in ``video2caption_replace.json``, so we
#    symlink the whole tree into place rather than copying ~25 GB.
#
#    ``video2caption_replace.json`` carries path/cap/resolution/fps/duration but
#    NOT ``num_frames`` — which the preprocess validator and batch builder hard-
#    require (components.py: _validate_data_type / _validate_frame_sampling /
#    VideoForwardBatchBuilder all index ``num_frames``). Loading the file as-is
#    yields a dataset with no such column, so ``filter(validator)`` raises
#    KeyError before any frame is read. Derive it from ``round(duration * fps)``
#    while writing the staged JSON instead of symlinking the source verbatim.
echo "[preprocess] staging ${STAGE_DIR} from ${RAW_SRC_DIR}"
mkdir -p "${STAGE_DIR}"
ln -sfn "$(realpath "${RAW_SRC_DIR}")" "${STAGE_DIR}/videos"
python - "${RAW_SRC_DIR}/video2caption_replace.json" "${STAGE_DIR}/videos2caption.json" <<'PY'
import json, sys

src, dst = sys.argv[1], sys.argv[2]
rows = json.load(open(src))
filled = 0
for r in rows:
    if not r.get("num_frames"):
        fps = r.get("fps") or 0
        dur = r.get("duration") or 0
        r["num_frames"] = int(round(dur * fps)) if fps and dur else 0
        filled += 1
with open(dst, "w") as f:
    json.dump(rows, f)
print(f"[preprocess] staged {len(rows)} caption rows ({filled} num_frames filled) -> {dst}")
PY

# 3. VAE + text encode at 480x832, 77 frames, 16 fps. ``samples_per_file=8``
#    matches the published HD-Mixkit-Finetune-Wan shard layout
#    (worker_0/data_chunk_*.parquet).
echo "[preprocess] running v1_preprocessing_new on ${NUM_GPUS} GPU(s)"
torchrun --nproc_per_node="${NUM_GPUS}" \
    -m fastvideo.pipelines.preprocess.v1_preprocessing_new \
    --model_path "${MODEL_PATH}" \
    --mode preprocess \
    --workload_type t2v \
    --preprocess.video_loader_type torchvision \
    --preprocess.dataset_type merged \
    --preprocess.dataset_path "${STAGE_DIR}" \
    --preprocess.dataset_output_dir "${DATA_ROOT}" \
    --preprocess.max_height 480 \
    --preprocess.max_width 832 \
    --preprocess.num_frames 77 \
    --preprocess.train_fps 16 \
    --preprocess.samples_per_file 8

echo
echo "Done."
echo "  Train data:      ${DATA_ROOT}/combined_parquet_dataset"
echo "  Validation data: ${DATA_ROOT}/validation_parquet_dataset"
echo "Point your training script's data path at the combined_parquet_dataset directory."
