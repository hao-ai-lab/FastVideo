#!/bin/bash
#SBATCH --job-name=fun-i2v-rctx
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=/leonardo_work/AIFAC_P02_082/.logs/coach/fun-i2v-rctx_%j.out
#SBATCH --error=/leonardo_work/AIFAC_P02_082/.logs/coach/fun-i2v-rctx_%j.err
#SBATCH --exclusive

set -e -x

CONFIG="${CONFIG:-examples/training/finetune/Wan2.1-Fun-1.3B-InP/soccernet_pcbas_w21_480/fun_i2v_random_context.yaml}"
RUNTIME_VENV_DEFAULT="${RUNTIME_VENV:-/leonardo_work/AIFAC_P02_082/code/FastVideo/.venv-fa}"

eval "$(
"${RUNTIME_VENV_DEFAULT}/bin/python" - "$CONFIG" "$SLURM_JOB_NUM_NODES" <<'PY'
import os
import shlex
import sys

import yaml

config_path = sys.argv[1]
num_nodes = int(sys.argv[2])

with open(config_path) as f:
    cfg = yaml.safe_load(f)


def get(section, key, default=None):
    return (cfg.get(section) or {}).get(key, default)


def quote(value):
    return shlex.quote(str(value))


def assign(name, value):
    print(f"{name}={quote(value)}")


def array(name, values):
    print(f"{name}=(")
    for value in values:
        print(f"  {quote(value)}")
    print(")")


def data_path_arg(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise SystemExit("paths.data_path must not be an empty list")
        return os.pathsep.join(str(path) for path in value)
    return value


gpus_per_node = int(get("job", "gpus_per_node", 8))
hsdp_replicate_dim = get("parallel", "hsdp_replicate_dim")
if hsdp_replicate_dim is None:
    hsdp_replicate_dim = num_nodes

assign("FASTVIDEO_PATH", get("env", "fastvideo_path", "/leonardo_work/AIFAC_P02_082/code/rahim/FastVideo"))
assign("RUNTIME_VENV", get("env", "runtime_venv", "/leonardo_work/AIFAC_P02_082/code/FastVideo/.venv-fa"))
assign("CFG_WANDB_MODE", get("env", "wandb_mode", "online"))
assign("CFG_WANDB_BASE_URL", get("env", "wandb_base_url", "https://api.wandb.ai"))
assign("CFG_TOKENIZERS_PARALLELISM", get("env", "tokenizer_parallelism", "false"))
assign("CFG_TRITON_CACHE_DIR_PREFIX", get("env", "triton_cache_dir_prefix", "/tmp/triton_cache"))
assign("NUM_GPUS", gpus_per_node)

model_path = get("paths", "model_path")
data_path = get("paths", "data_path")
data_path = data_path_arg(data_path)
validation_file = get("paths", "validation_dataset_file")
output_dir = get("paths", "output_dir")

min_cond = int(get("training", "min_condition_latents", 1))
max_cond = int(get("training", "max_condition_latents", 1))
num_latent_t = int(get("training", "num_latent_t"))
if min_cond < 1:
    raise SystemExit("training.min_condition_latents must be >= 1")
if max_cond < min_cond:
    raise SystemExit("training.max_condition_latents must be >= min_condition_latents")
if max_cond >= num_latent_t:
    raise SystemExit("training.max_condition_latents must be < training.num_latent_t")

array("parallel_args", [
    "--num_gpus", gpus_per_node,
    "--sp_size", get("parallel", "sp_size", gpus_per_node),
    "--tp_size", get("parallel", "tp_size", 1),
    "--hsdp_replicate_dim", hsdp_replicate_dim,
    "--hsdp_shard_dim", get("parallel", "hsdp_shard_dim", gpus_per_node),
])

array("model_args", [
    "--model_path", model_path,
    "--pretrained_model_name_or_path", model_path,
])

array("dataset_args", [
    "--data_path", data_path,
    "--data_split", get("training", "data_split", "all"),
    "--validation_split_ratio", get("training", "validation_split_ratio", 0.0),
    "--dataloader_num_workers", get("training", "dataloader_num_workers", 1),
])

array("training_args", [
    "--tracker_project_name", get("training", "tracker_project_name", "wan_i2v_finetune"),
    "--wandb_run_name", get("training", "wandb_run_name", ""),
    "--output_dir", output_dir,
    "--max_train_steps", get("training", "max_train_steps", 2000),
    "--train_batch_size", get("training", "train_batch_size", 1),
    "--train_sp_batch_size", get("training", "train_sp_batch_size", 1),
    "--gradient_accumulation_steps", get("training", "gradient_accumulation_steps", 1),
    "--seed", get("training", "seed", 42),
    "--num_latent_t", num_latent_t,
    "--min_condition_latents", min_cond,
    "--max_condition_latents", max_cond,
    "--num_height", get("training", "num_height", 480),
    "--num_width", get("training", "num_width", 832),
    "--num_frames", get("training", "num_frames", 77),
    "--enable_gradient_checkpointing_type", get("training", "enable_gradient_checkpointing_type", "full"),
])

validation_args = []
if bool(get("validation", "log_validation", False)):
    validation_args.append("--log_validation")
validation_args.extend([
    "--validation_dataset_file", validation_file,
    "--validation_steps", get("validation", "validation_steps", 100),
    "--validation_sampling_steps", get("validation", "validation_sampling_steps", "40"),
    "--validation_guidance_scale", get("validation", "validation_guidance_scale", "6.0"),
])
array("validation_args", validation_args)

array("optimizer_args", [
    "--learning_rate", get("optimizer", "learning_rate", 1.0e-5),
    "--mixed_precision", get("training", "mixed_precision", "bf16"),
    "--weight_only_checkpointing_steps", get("optimizer", "weight_only_checkpointing_steps", 1000),
    "--training_state_checkpointing_steps", get("optimizer", "training_state_checkpointing_steps", 1000),
    "--weight_decay", get("optimizer", "weight_decay", 1.0e-4),
    "--max_grad_norm", get("optimizer", "max_grad_norm", 1.0),
])

miscellaneous_args = [
    "--inference_mode", "False",
    "--checkpoints_total_limit", get("training", "checkpoints_total_limit", 3),
    "--training_cfg_rate", get("training", "training_cfg_rate", 0.1),
    "--multi_phased_distill_schedule", get("misc", "multi_phased_distill_schedule", "4000-1"),
    "--dit_precision", get("training", "dit_precision", "fp32"),
    "--num_euler_timesteps", get("misc", "num_euler_timesteps", 50),
    "--ema_start_step", get("misc", "ema_start_step", 0),
    "--enable_gradient_checkpointing_type", get("training", "enable_gradient_checkpointing_type", "full"),
]
if bool(get("misc", "not_apply_cfg_solver", True)):
    miscellaneous_args.append("--not_apply_cfg_solver")
array("miscellaneous_args", miscellaneous_args)
PY
)"

PYTHON_BIN="${RUNTIME_VENV}/bin/python"
TORCHRUN_BIN="${RUNTIME_VENV}/bin/torchrun"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "ERROR: torchrun not found or not executable: $TORCHRUN_BIN" >&2
    exit 1
fi
if [[ ! -d "$FASTVIDEO_PATH/fastvideo" ]]; then
    echo "ERROR: FASTVIDEO_PATH does not look like a FastVideo checkout: $FASTVIDEO_PATH" >&2
    exit 1
fi

mkdir -p /leonardo_work/AIFAC_P02_082/.logs

export FASTVIDEO_PATH
export PYTHONPATH="${FASTVIDEO_PATH}:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-${CFG_WANDB_MODE}}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-${CFG_WANDB_BASE_URL}}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TRITON_CACHE_DIR="${CFG_TRITON_CACHE_DIR_PREFIX}_${SLURM_PROCID}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NODE_RANK="${SLURM_PROCID}"
export TOKENIZERS_PARALLELISM="${CFG_TOKENIZERS_PARALLELISM}"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"

echo "CONFIG: $CONFIG"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"
echo "NUM_GPUS: $NUM_GPUS"
echo "FASTVIDEO_PATH: $FASTVIDEO_PATH"
echo "RUNTIME_VENV: $RUNTIME_VENV"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "TORCHRUN_BIN: $TORCHRUN_BIN"

srun "$TORCHRUN_BIN" \
    --nnodes "$SLURM_JOB_NUM_NODES" \
    --nproc_per_node "$NUM_GPUS" \
    --node_rank "$SLURM_PROCID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$FASTVIDEO_PATH/fastvideo/training/wan_i2v_training_pipeline.py" \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}"
