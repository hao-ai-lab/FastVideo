#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be provided through a secret}"

run_dir="${PROMPTRL_RUN_DIR:-/root/data/promptrl_runs/wan21_prompt_only}"
max_train_steps="${PROMPTRL_MAX_TRAIN_STEPS:-1000}"
reward_port="${PROMPTRL_REWARD_PORT:-8100}"
reward_log="${run_dir}/reward-service.log"
prompt_data="${PROMPTRL_PROMPT_DATA:-examples/train/configs/rl/wan/promptrl_vbench_prompts.jsonl}"

mkdir -p "${run_dir}"

if [[ ! -f "${prompt_data}" ]]; then
  echo "Prompt dataset not found: ${prompt_data}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 python -m \
  fastvideo.train.methods.rl.promptrl.rewards.serve \
  --port "${reward_port}" >"${reward_log}" 2>&1 &
reward_pid=$!

cleanup() {
  kill "${reward_pid}" 2>/dev/null || true
  wait "${reward_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
  if curl --fail --silent --show-error \
    "http://127.0.0.1:${reward_port}/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "${reward_pid}" 2>/dev/null; then
    tail -n 200 "${reward_log}"
    exit 1
  fi
  sleep 5
done

curl --fail --silent --show-error \
  "http://127.0.0.1:${reward_port}/healthz" >/dev/null
echo "PROMPTRL_REWARD_SERVICE_READY"

torchrun --standalone --nproc-per-node=8 -m \
  fastvideo.train.entrypoint.train \
  --config examples/train/configs/rl/wan/promptrl_prompt_only.yaml \
  --method.data.data_path "${prompt_data}" \
  --method.reward.endpoint_url "http://127.0.0.1:${reward_port}" \
  --method.validation.every_steps 0 \
  --training.loop.max_train_steps "${max_train_steps}" \
  --training.checkpoint.output_dir "${run_dir}/output" \
  --training.checkpoint.training_state_checkpointing_steps 1 \
  --training.checkpoint.checkpoints_total_limit 2 \
  --training.tracker.trackers "[wandb]" \
  --training.tracker.project_name promptrl_wan \
  --training.tracker.run_name wan2.1_promptrl_prompt_only \
  --callbacks.promptrl_export.output_dir "${run_dir}/output/bundle"

echo "PROMPTRL_TRAINING_OK steps=${max_train_steps}"
