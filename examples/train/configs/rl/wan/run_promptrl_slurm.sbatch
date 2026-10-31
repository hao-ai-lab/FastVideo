#!/bin/bash
#SBATCH --job-name=promptrl-wan
#SBATCH --partition=all
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --chdir=.
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"

work_root="${PROMPTRL_WORK_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
export HOME="${PROMPTRL_HOME:-${work_root}}"

source_dir="${PROMPTRL_SOURCE_DIR:-${work_root}/promptrl/staging/source}"
run_dir="${PROMPTRL_RUN_DIR:-${work_root}/promptrl_runs/wan21_prompt_only}"
container_image="${PROMPTRL_CONTAINER_IMAGE:-ghcr.io#hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest}"
reward_port="${PROMPTRL_REWARD_PORT:-8100}"
max_train_steps="${PROMPTRL_MAX_TRAIN_STEPS:-1000}"
master_port="${PROMPTRL_MASTER_PORT:-29500}"
prompt_data="/workspace/FastVideo/examples/train/configs/rl/wan/promptrl_vbench_prompts.jsonl"

mkdir -p "${run_dir}" "${run_dir}/wandb" "${run_dir}/hf-cache"
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#nodes[@]}" -ne 3 ]]; then
  echo "Expected exactly three allocated nodes, got ${#nodes[@]}" >&2
  exit 1
fi

reward_node="${nodes[0]}"
train_node_0="${nodes[1]}"
train_node_1="${nodes[2]}"
train_nodes="${train_node_0},${train_node_1}"
reward_endpoint="http://${reward_node}:${reward_port}"
mounts="${source_dir}:/workspace/FastVideo,${run_dir}:/workspace/run"

# A user-provisioned mode-600 file may contain WANDB_API_KEY and optional
# WANDB_ENTITY. If it is absent, preserve all logs in W&B offline format so
# they can be synchronized later without delaying the queued GPU allocation.
wandb_env="${HOME}/.config/fastvideo/wandb.env"
if [[ -r "${wandb_env}" ]]; then
  # shellcheck disable=SC1090
  source "${wandb_env}"
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  export WANDB_MODE=offline
fi

export HF_HOME="${run_dir}/hf-cache"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${run_dir}/wandb"
export WANDB_PROJECT=promptrl_wan
export WANDB_RUN_ID=wan21-promptrl-prompt-only
export WANDB_RESUME=allow

echo "PROMPTRL_SLURM_LAYOUT reward=${reward_node} train=${train_nodes}"
echo "PROMPTRL_TRACKING_MODE mode=${WANDB_MODE}"

srun \
  --export=ALL \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  --nodelist="${reward_node}" \
  --gres=gpu:1 \
  --container-image="${container_image}" \
  --container-mounts="${mounts}" \
  --container-workdir=/workspace/FastVideo \
  --container-writable \
  --container-env=HF_HOME,TOKENIZERS_PARALLELISM \
  bash -lc '
    set -euo pipefail
    source /opt/venv/bin/activate
    /root/.local/bin/uv pip install qwen-vl-utils python-multipart
    python -c "import torch; assert torch.cuda.get_device_capability(0) == (10, 0); assert torch.ones(1, device=\"cuda\").item() == 1"
    python -m fastvideo.train.methods.rl.promptrl.rewards.serve \
      --host 0.0.0.0 \
      --port "'"${reward_port}"'" \
      > /workspace/run/reward-service.log 2>&1
  ' &
reward_step_pid=$!

cleanup() {
  kill "${reward_step_pid}" 2>/dev/null || true
  wait "${reward_step_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl --fail --silent --show-error \
    "${reward_endpoint}/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "${reward_step_pid}" 2>/dev/null; then
    tail -n 200 "${run_dir}/reward-service.log" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done
curl --fail --silent --show-error "${reward_endpoint}/healthz" >/dev/null
echo "PROMPTRL_REWARD_SERVICE_READY endpoint=${reward_endpoint}"

export PROMPTRL_REWARD_ENDPOINT="${reward_endpoint}"
export PROMPTRL_MASTER_ADDR="${train_node_0}"
export PROMPTRL_MASTER_PORT="${master_port}"
export PROMPTRL_MAX_TRAIN_STEPS="${max_train_steps}"

srun \
  --export=ALL \
  --overlap \
  --nodes=2 \
  --ntasks=2 \
  --ntasks-per-node=1 \
  --nodelist="${train_nodes}" \
  --gres=gpu:4 \
  --container-image="${container_image}" \
  --container-mounts="${mounts}" \
  --container-workdir=/workspace/FastVideo \
  --container-writable \
  --container-env=HF_HOME,TOKENIZERS_PARALLELISM,WANDB_API_KEY,WANDB_ENTITY,WANDB_MODE,WANDB_DIR,WANDB_PROJECT,WANDB_RUN_ID,WANDB_RESUME,PROMPTRL_REWARD_ENDPOINT,PROMPTRL_MASTER_ADDR,PROMPTRL_MASTER_PORT,PROMPTRL_MAX_TRAIN_STEPS \
  bash -lc '
    set -euo pipefail
    source /opt/venv/bin/activate
    /root/.local/bin/uv pip install qwen-vl-utils python-multipart
    python -c "import torch; assert torch.cuda.get_device_capability(0) == (10, 0); assert torch.ones(1, device=\"cuda\").item() == 1"

    resume_args=()
    if compgen -G "/workspace/run/output/checkpoint-*/dcp" >/dev/null; then
      resume_args=(--training.checkpoint.resume_from_checkpoint latest)
    fi

    torchrun \
      --nnodes=2 \
      --nproc-per-node=4 \
      --node-rank="${SLURM_PROCID}" \
      --master-addr="${PROMPTRL_MASTER_ADDR}" \
      --master-port="${PROMPTRL_MASTER_PORT}" \
      -m fastvideo.train.entrypoint.train \
      --config examples/train/configs/rl/wan/promptrl_prompt_only.yaml \
      --method.data.data_path "'"${prompt_data}"'" \
      --method.reward.endpoint_url "${PROMPTRL_REWARD_ENDPOINT}" \
      --method.validation.every_steps 0 \
      --training.loop.max_train_steps "${PROMPTRL_MAX_TRAIN_STEPS}" \
      --training.checkpoint.output_dir /workspace/run/output \
      --training.checkpoint.training_state_checkpointing_steps 10 \
      --training.checkpoint.checkpoints_total_limit 4 \
      --training.tracker.trackers "[wandb]" \
      --training.tracker.project_name promptrl_wan \
      --training.tracker.run_name wan2.1_promptrl_prompt_only \
      --callbacks.promptrl_export.output_dir /workspace/run/output/bundle \
      "${resume_args[@]}"
  '

echo "PROMPTRL_TRAINING_OK steps=${max_train_steps}"
