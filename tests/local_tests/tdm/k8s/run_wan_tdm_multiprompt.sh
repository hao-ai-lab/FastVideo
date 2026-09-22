#!/usr/bin/env bash
# Phase 3.1: multi-prompt + held-out diversity follow-up for the fixed recipe.
#
# Builds a four-prompt text-only dataset, runs the validated fixed recipe on
# it (treatment: warmup 100 then TDM 100; control: warmup 200), and samples a
# training + held-out validation set with repeated rows so the report can
# separate within-caption seed spread from cross-caption spread.
#
#   train prompts       tdm_multiprompt_train_prompts.txt (4 prompts)
#   validation prompts  tdm_multiprompt_validation.json (4 train + 4 held-out,
#                       each repeated four times; the repeat makes the caption
#                       to filename mapping deterministic)
#
# Acceptance is reference-free: tdm_multiprompt_report.py reports per-caption
# frame sharpness and within-caption spread; paired metrics stay forensic.
#
# Expected pod state: tests/local_tests/tdm/k8s/pod_wan_tdm.yaml held
# four-GPU pod with /tmp/tdm-port.tar.gz uploaded by the operator.
set -Eeuo pipefail

run_name=${RUN_NAME:-wan-tdm-multiprompt}
run_root=/workspace/run/tdm-port/$run_name
test ! -e "$run_root"
mkdir -p "$run_root/logs"

record_status() {
    rc=$?
    printf 'completed_at=%s\nrc=%s\n' "$(date -u +%FT%TZ)" "$rc" > "$run_root/status.tmp"
    mv "$run_root/status.tmp" "$run_root/status"
}
trap record_status EXIT
exec > >(tee -a "$run_root/logs/stage.log") 2>&1

export HOME=/tmp/tdm-home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/issue-775/shared-cache/hf
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export NCCL_DEBUG=WARN
mkdir -p "$HOME"

printf 'START %s %s\n' "$run_name" "$(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
printf 'CODE_SHA256 %s\n' "$(sha256sum /tmp/tdm-port.tar.gz | cut -d' ' -f1)"

rm -rf /tmp/tdm-port
mkdir -p /tmp/tdm-port
tar xzf /tmp/tdm-port.tar.gz -C /tmp/tdm-port
cd /tmp/tdm-port

prompts_file=examples/train/configs/distribution_matching/wan/tdm_multiprompt_train_prompts.txt
validation_file=examples/train/configs/distribution_matching/wan/tdm_multiprompt_validation.json
dataset_dir=data/tdm_multiprompt_text_only

printf 'PREPROCESS %s\n' "$(date -u +%FT%TZ)"
rm -rf "$dataset_dir"
mkdir -p "$dataset_dir/_text_shards"
/opt/venv/bin/python - "$prompts_file" "$dataset_dir/_text_shards" <<'PY'
from pathlib import Path
import sys

prompts = [line.rstrip("\n") for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
if not prompts:
    raise SystemExit("no prompts")
shard_dir = Path(sys.argv[2])
for shard_idx in range(4):
    shard_prompts = prompts[shard_idx::4]
    (shard_dir / f"train_text_shard_{shard_idx}.txt").write_text(
        "\n".join(shard_prompts) + "\n", encoding="utf-8")
    print(f"shard {shard_idx}: {len(shard_prompts)} prompts")
PY

for gpu in 0 1 2 3; do
    mkdir -p "$dataset_dir/shard-$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" /opt/venv/bin/torchrun --nnodes=1 --nproc_per_node=1 \
        --master_port "$((29610 + gpu))" \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --data_merge_path "$dataset_dir/_text_shards/train_text_shard_${gpu}.txt" \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 448 \
        --max_width 832 \
        --num_frames 77 \
        --dataloader_num_workers 0 \
        --output_dir "$dataset_dir/shard-$gpu" \
        --train_fps 16 \
        --samples_per_file 8 \
        --flush_frequency 8 \
        --text_max_length 512 \
        --video_length_tolerance_range 5 \
        --preprocess_task text_only > "$run_root/logs/preprocess-gpu$gpu.log" 2>&1 &
done
wait

for gpu in 0 1 2 3; do
    test -f "$dataset_dir/shard-$gpu/combined_parquet_dataset/worker_0/data_chunk_0.parquet" \
        || { echo "missing shard $gpu parquet"; exit 1; }
done
printf 'DATASET_OK %s\n' "$(date -u +%FT%TZ)"

config=examples/train/configs/distribution_matching/wan/tdm_t2v_lora_fixed_recipe.yaml
student_lr=${STUDENT_LR:-1.0e-4}
fake_score_lr=${FAKE_SCORE_LR:-1.0e-4}

run_one() {
    local label=$1 warmup=$2
    local out="$run_root/$label/output"
    printf 'TRAIN %s %s warmup=%s\n' "$label" "$(date -u +%FT%TZ)" "$warmup"
    /opt/venv/bin/torchrun --standalone --nproc_per_node=4 -m fastvideo.train.entrypoint.train \
        --config "$config" \
        --training.checkpoint.output_dir "$out" \
        --training.data.data_path "$dataset_dir" \
        --training.optimizer.learning_rate "$student_lr" \
        --method.fake_score_learning_rate "$fake_score_lr" \
        --method.warmup_steps "$warmup" \
        --callbacks.validation.dataset_file "$validation_file" \
        --callbacks.validation.num_videos_per_prompt 1 \
        --callbacks.validation.every_steps 50 > "$run_root/logs/train-$label.log" 2>&1
    grep -q "Training completed" "$run_root/logs/train-$label.log"
    tail -3 "$run_root/logs/train-$label.log"
}

run_one treatment 100
run_one control 200

printf 'REPORT %s\n' "$(date -u +%FT%TZ)"
for label in treatment control; do
    PYTHONPATH=/tmp/tdm-port /opt/venv/bin/python tests/local_tests/tdm/tools/tdm_multiprompt_report.py \
        --run-dir "$run_root/$label/output" \
        --captions-json "$validation_file" \
        --label "$label" \
        --out "$run_root/$label/multiprompt_report.json" > "$run_root/logs/multiprompt-$label.log" 2>&1
    grep -E "step=|MULTIPROMPT_REPORT" "$run_root/logs/multiprompt-$label.log" | tail -8
done

printf 'STAGE_OK %s\n' "$(date -u +%FT%TZ)"
