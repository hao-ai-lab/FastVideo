from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


DATA_PATH = "/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset"
STREAMING_MANIFEST_PATH = "/mnt/lustre/vlm-k1kong/dataset-index/openvid/streaming-t2v-v2.json"
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
TEACHER_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
REQUIRED_ANCESTOR = "30ada30e4c6b05aa68cd1eb8940a34d149457147"
TRAIN_NUM_LATENTS = 21
TRAIN_NUM_FRAMES = 81
VALIDATION_NUM_FRAMES = 81

CONDITIONS = {
    "A12": {"sink": 1, "local": 6, "rope": "relativistic", "framewise": False},
    "A13": {"sink": 0, "local": 6, "rope": "relativistic", "framewise": False},
    "A14": {"sink": 1, "local": 6, "rope": "absolute", "framewise": False},
    "A15": {"sink": 1, "local": 6, "rope": "relativistic", "framewise": True},
}


def training(stage_dir: str, *, steps: int, keep: int) -> dict:
    return {
        "distributed": {
            "num_gpus": 4,
            "sp_size": 1,
            "tp_size": 1,
            "hsdp_replicate_dim": 1,
            "hsdp_shard_dim": 4,
        },
        "data": {
            "data_path": DATA_PATH,
            "dataloader_type": "streaming",
            "streaming_manifest_path": STREAMING_MANIFEST_PATH,
            "streaming_read_batch_size": 2,
            "streaming_shuffle_row_groups": True,
            "dataloader_num_workers": 0,
            "train_batch_size": 2,
            "training_cfg_rate": 0.0,
            "seed": 1000,
            # Keep every stage on the intended 21-latent / 81-frame training
            # distribution.  For chunk-3 conditions this is exactly seven
            # aligned blocks; A15 is length-matched and uses framewise blocks.
            "num_latent_t": TRAIN_NUM_LATENTS,
            "num_height": 480,
            "num_width": 832,
            "num_frames": TRAIN_NUM_FRAMES,
        },
        "optimizer": {
            "learning_rate": 2.0e-6,
            "betas": [0.0, 0.999],
            "weight_decay": 0.01,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
        },
        "loop": {
            "max_train_steps": steps,
            # 2 samples/rank * 4 ranks * 8 accumulation = global batch 64.
            "gradient_accumulation_steps": 8,
        },
        "checkpoint": {
            "output_dir": f"{stage_dir}/checkpoints",
            "training_state_checkpointing_steps": 1000,
            "checkpoints_total_limit": keep,
        },
        "tracker": {
            "trackers": ["wandb"],
            "project_name": "causal_forcing_openvid_a12_a15",
            "run_name": "RENDERED_BY_PREPARE_SCRIPT",
        },
        "model": {"enable_gradient_checkpointing_type": "full"},
    }


def validation(stage_dir: str, *, distilled: bool, dmd: bool = False) -> dict:
    result = {
        "pipeline_target": (
            "fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline.WanCausalDMDPipeline"
            if dmd
            else "fastvideo.pipelines.basic.wan.wan_causal_pipeline.WanCausalPipeline"
        ),
        "dataset_file": "__REPO__/examples/training/finetune/Wan2.1-VSA/Wan-Syn-Data/validation_4.json",
        "every_steps": 200,
        "sampling_steps": [4] if distilled else [40],
        "guidance_scale": 3.0 if distilled else 6.0,
        "num_frames": VALIDATION_NUM_FRAMES,
        "output_dir": f"{stage_dir}/validation",
        "offload_training_state": True,
        "unload_pipeline_after_validation": True,
    }
    if dmd:
        result["sampling_timesteps"] = [1000, 750, 500, 250]
    return result


def causal_role(*, trainable: bool, override: str | None, framewise: bool) -> dict:
    role = {
        "_target_": "fastvideo.train.models.wan.WanCausalModel",
        "init_from": MODEL_ID,
        "trainable": trainable,
    }
    if override:
        role["transformer_override_safetensor"] = override
    if framewise:
        role["num_frames_per_block"] = 1
    return role


def configs(run_root: str, condition: str, spec: dict, repo: str) -> dict[str, dict]:
    framewise = bool(spec["framewise"])
    tag = f"{condition}_sink{spec['sink']}_local{spec['local']}_{spec['rope']}" + (
        "_framewise" if framewise else "_chunk3"
    )
    pipeline = {
        "flow_shift": 5,
        "dit_config": {
            "local_attn_size": spec["local"],
            "sink_size": spec["sink"],
            "rope_cache_policy": spec["rope"],
            "causal_train_attention": "triton",
        },
    }
    tf_dir = f"{run_root}/tf"
    cd_dir = f"{run_root}/cd"
    sf_dir = f"{run_root}/sf"
    tf_export = f"{run_root}/export/tf/transformer/model.safetensors"
    cd_export = f"{run_root}/export/cd/transformer/model.safetensors"

    tf = {
        "models": {
            "student": causal_role(trainable=True, override=None, framewise=framewise),
        },
        "method": {
            "_target_": "fastvideo.train.methods.fine_tuning.tfsft.TeacherForcingSFTMethod",
            "chunk_size": 1 if framewise else 3,
        },
        "training": training(tf_dir, steps=3000, keep=3),
        "callbacks": {
            "grad_clip": {"max_grad_norm": 1.0},
            "validation": validation(tf_dir, distilled=False),
        },
        "pipeline": copy.deepcopy(pipeline),
    }
    tf["training"]["tracker"]["run_name"] = f"{tag}_tf3k_openvid_81f21l_gbs64"

    cd = {
        "models": {
            role: causal_role(
                trainable=(role == "student"),
                override=tf_export,
                framewise=framewise,
            )
            for role in ("student", "teacher", "ema")
        },
        "method": {
            "_target_": "fastvideo.train.methods.consistency_model.causal_cd.CausalConsistencyDistillationMethod",
            "discrete_cd_N": 48,
            "guidance_scale": 3.0,
            "ema_decay": 0.99,
            "ema_start_step": 200,
        },
        "training": training(cd_dir, steps=2000, keep=2),
        "callbacks": {
            "grad_clip": {"max_grad_norm": 1.0},
            "validation": validation(cd_dir, distilled=True, dmd=False),
        },
        "pipeline": copy.deepcopy(pipeline),
    }
    cd["training"]["tracker"]["run_name"] = f"{tag}_cd2k_from_tf3k_openvid_81f21l_gbs64"

    sf_student = causal_role(trainable=True, override=cd_export, framewise=framewise)
    sf = {
        "models": {
            "student": sf_student,
            "teacher": {
                "_target_": "fastvideo.train.models.wan.WanModel",
                "init_from": TEACHER_ID,
                "trainable": False,
                "disable_custom_init_weights": True,
            },
            "critic": {
                "_target_": "fastvideo.train.models.wan.WanModel",
                "init_from": MODEL_ID,
                "trainable": True,
                "disable_custom_init_weights": True,
            },
        },
        "method": {
            "_target_": "fastvideo.train.methods.distribution_matching.self_forcing.SelfForcingMethod",
            "rollout_mode": "simulate",
            "generator_update_interval": 5,
            "real_score_guidance_scale": 4.0,
            "dmd_denoising_steps": [1000, 750, 500, 250],
            "warp_denoising_step": True,
            "chunk_size": 1 if framewise else 3,
            "student_sample_type": "sde",
            "context_noise": 0.0,
            "same_step_across_blocks": True,
            "enable_gradient_in_rollout": True,
            "start_gradient_frame": 0,
            # Preserve the proven critic LR; the requested 2e-6 applies to
            # the main stage optimizer in training.optimizer.
            "fake_score_learning_rate": 4.0e-7,
            "fake_score_betas": [0.0, 0.999],
            "fake_score_lr_scheduler": "constant",
        },
        "training": training(sf_dir, steps=1000, keep=1),
        "callbacks": {
            "ema": {"decay": 0.99, "start_iter": 200},
            "grad_clip": {"max_grad_norm": 1.0},
            "validation": validation(sf_dir, distilled=True, dmd=True),
        },
        "pipeline": copy.deepcopy(pipeline),
    }
    sf["training"]["tracker"]["run_name"] = f"{tag}_sf1k_from_cd2k_openvid_81f21l_gbs64"

    for config in (tf, cd, sf):
        config["callbacks"]["validation"]["dataset_file"] = config["callbacks"]["validation"]["dataset_file"].replace(
            "__REPO__", repo
        )
    return {"tf": tf, "cd": cd, "sf": sf}


TRAIN_STAGE = r"""#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?}" "${STAGE:?}" "${MASTER_PORT:?}"
REPO="${REPO:-__REPO__}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
WANDB_MODE="${WANDB_MODE:-online}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
case "$WANDB_MODE" in
  online)
    if [[ "$PREFLIGHT_ONLY" == 0 && -z "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_MODE=online requires WANDB_API_KEY at runtime." >&2
      exit 2
    fi
    ;;
  offline)
    unset WANDB_API_KEY
    ;;
  *)
    echo "Unsupported WANDB_MODE=$WANDB_MODE; expected online or offline." >&2
    exit 2
    ;;
esac
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    echo "PREFLIGHT_ONLY must be 0 or 1, got: $PREFLIGHT_ONLY" >&2
    exit 2
    ;;
esac
REQUIRED_ANCESTOR="__COMMIT__"
STAGE_DIR="$RUN_ROOT/$STAGE"
CONFIG="$STAGE_DIR/config/run.yaml"
STATE="$STAGE_DIR/state"
LOG="$STAGE_DIR/logs/train.log"
git -C "$REPO" merge-base --is-ancestor "$REQUIRED_ANCESTOR" HEAD
mkdir -p "$STAGE_DIR"/{logs,state,checkpoints,validation,tracker}

export HF_HOME=/mnt/lustre/vlm-k1kong/hf-cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export XDG_CACHE_HOME=/mnt/lustre/vlm-k1kong/xdg-cache
export CUDA_CACHE_PATH=/mnt/lustre/vlm-k1kong/cuda-cache
export TRITON_CACHE_DIR="/mnt/lustre/vlm-k1kong/triton-cache/openvid-a12-a15/${HOSTNAME}/${CONDITION}/${STAGE}"
export TORCHINDUCTOR_CACHE_DIR="/mnt/lustre/vlm-k1kong/torchinductor-cache/openvid-a12-a15/${HOSTNAME}/${CONDITION}/${STAGE}"
export FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN
export WANDB_MODE WANDB_ENTITY=kaiqin_kong_ucsd
export WANDB_PROJECT=causal_forcing_openvid_a12_a15
export TOKENIZERS_PARALLELISM=false FASTVIDEO_DIST_TIMEOUT_MINUTES=120
export TORCH_NCCL_BLOCKING_WAIT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 VIRTUAL_ENV="$ENV_DIR" PATH="$ENV_DIR/bin:$PATH" PYTHONPATH="$REPO"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

"$ENV_DIR/bin/python" - "$CONFIG" <<'PY'
from __future__ import annotations

import sys

import yaml

from fastvideo.train.utils.config import load_run_config


config_path = sys.argv[1]
data = load_run_config(config_path).training.data
expected = {
    "data_path": "/mnt/lustre/vlm-s4duan/openvid_1m/combined_parquet_dataset",
    "dataloader_type": "streaming",
    "streaming_manifest_path": "/mnt/lustre/vlm-k1kong/dataset-index/openvid/streaming-t2v-v2.json",
    "streaming_read_batch_size": 2,
    "streaming_shuffle_row_groups": True,
    "dataloader_num_workers": 0,
}
actual = {name: getattr(data, name) for name in expected}
with open(config_path, encoding="utf-8") as handle:
    raw = yaml.safe_load(handle)
expected.update({
    "num_latent_t": 21,
    "num_frames": 81,
})
actual.update({
    "num_latent_t": data.num_latent_t,
    "num_frames": data.num_frames,
})
validation_frames = raw["callbacks"]["validation"]["num_frames"]
if validation_frames != 81:
    raise SystemExit(
        f"OpenVid sequence-length preflight failed for {config_path}: "
        f"callbacks.validation.num_frames expected 81, got {validation_frames!r}"
    )
errors = [
    f"{name}: expected {value!r}, got {actual[name]!r}"
    for name, value in expected.items()
    if actual[name] != value
]
if errors:
    raise SystemExit(
        "OpenVid streaming config preflight failed for "
        f"{config_path}: " + "; ".join(errors)
    )
print(f"OpenVid streaming config preflight passed: {config_path}")
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "Launcher preflight passed without starting training: WANDB_MODE=$WANDB_MODE config=$CONFIG"
  exit 0
fi

if [[ "$WANDB_MODE" == online ]]; then
  export WANDB_RESUME=allow
  if [[ -s "$STATE/wandb_run_id" ]]; then
    export WANDB_RUN_ID="$(<"$STATE/wandb_run_id")"
  else
    export WANDB_RUN_ID="$($ENV_DIR/bin/python -c 'import wandb; print(wandb.util.generate_id())')"
    printf '%s\n' "$WANDB_RUN_ID" > "$STATE/wandb_run_id"
  fi
else
  unset WANDB_RESUME WANDB_RUN_ID
fi
resume=()
if find "$STAGE_DIR/checkpoints" -mindepth 2 -maxdepth 2 -type d -name dcp -print -quit | grep -q .; then
  resume+=(--training.checkpoint.resume_from_checkpoint latest)
fi
cmd=("$ENV_DIR/bin/python" -m torch.distributed.run --nnodes 1 --node_rank 0
  --nproc_per_node 4 --master_addr 127.0.0.1 --master_port "$MASTER_PORT"
  -m fastvideo.train.entrypoint.train --config "$CONFIG" "${resume[@]}")
printf 'running\n' > "$STATE/status"; date -Is > "$STATE/started_at"
cd "$REPO"; set +e; "${cmd[@]}" 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}; set -e
printf '%s\n' "$rc" > "$STATE/exit_code"; date -Is > "$STATE/finished_at"
if [[ "$rc" -eq 0 ]]; then printf 'completed\n' > "$STATE/status"; else printf 'failed\n' > "$STATE/status"; fi
exit "$rc"
"""


QUEUE = r"""#!/usr/bin/env bash
set -euo pipefail
CONDITION="${1:?A12/A13/A14/A15}"
RUN_ROOT="${2:?prepared condition run root}"
BASE_PORT="${3:-29800}"
REPO="${REPO:-__REPO__}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
WANDB_MODE="${WANDB_MODE:-online}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    echo "PREFLIGHT_ONLY must be 0 or 1, got: $PREFLIGHT_ONLY" >&2
    exit 2
    ;;
esac
case "$WANDB_MODE" in
  online)
    if [[ "$PREFLIGHT_ONLY" == 0 && -z "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_MODE=online requires WANDB_API_KEY at runtime." >&2
      exit 2
    fi
    ;;
  offline)
    unset WANDB_API_KEY
    ;;
  *)
    echo "Unsupported WANDB_MODE=$WANDB_MODE; expected online or offline." >&2
    exit 2
    ;;
esac
export REPO ENV_DIR CONDITION RUN_ROOT WANDB_MODE PREFLIGHT_ONLY

run_stage() {
  local stage="$1" final="$2" port="$3"
  if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
    STAGE="$stage" MASTER_PORT="$port" bash "$RUN_ROOT/scripts/train_stage.sh"
    return
  fi
  if [[ -d "$RUN_ROOT/$stage/checkpoints/checkpoint-$final/dcp" ]] &&
     [[ "$(cat "$RUN_ROOT/$stage/state/exit_code" 2>/dev/null || true)" == 0 ]]; then
    echo "$CONDITION $stage checkpoint-$final already complete; skipping"
    return
  fi
  STAGE="$stage" MASTER_PORT="$port" bash "$RUN_ROOT/scripts/train_stage.sh"
  test -d "$RUN_ROOT/$stage/checkpoints/checkpoint-$final/dcp"
}
export_stage() {
  local stage="$1" final="$2" role="${3:-student}"
  local checkpoint="$RUN_ROOT/$stage/checkpoints/checkpoint-$final"
  local marker="$RUN_ROOT/export/$stage/.source_checkpoint_fingerprint"
  local checkpoint_hash config_hash git_head current
  checkpoint_hash="$(find "$checkpoint" -type f -printf '%P:%s:%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}')"
  config_hash="$(sha256sum "$RUN_ROOT/$stage/config/run.yaml" | awk '{print $1}')"
  git_head="$(git -C "$REPO" rev-parse HEAD)"
  current="$(printf 'git_head=%s\nrole=%s\nconfig_sha256=%s\ncheckpoint_metadata=%s\n' \
    "$git_head" "$role" "$config_hash" "$checkpoint_hash" | sha256sum | awk '{print $1}')"
  if [[ -s "$RUN_ROOT/export/$stage/transformer/model.safetensors" ]] &&
     [[ "$(cat "$marker" 2>/dev/null || true)" == "$current" ]]; then
    echo "$CONDITION $stage export already complete; skipping"
    return
  fi
  "$ENV_DIR/bin/python" -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role "$role" --config "$RUN_ROOT/$stage/config/run.yaml" \
    --checkpoint "$RUN_ROOT/$stage/checkpoints/checkpoint-$final" \
    --output-dir "$RUN_ROOT/export/$stage" --overwrite \
    2>&1 | tee "$RUN_ROOT/$stage/logs/export.log"
  printf '%s\n' "$current" > "$marker"
}

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  run_stage tf 3000 "$((BASE_PORT + 1))"
  run_stage cd 2000 "$((BASE_PORT + 2))"
  run_stage sf 1000 "$((BASE_PORT + 3))"
  echo "$CONDITION launcher preflight passed for tf, cd, and sf; no training was started."
  exit 0
fi

mkdir -p "$RUN_ROOT/state"
on_exit() {
  local rc=$?
  if [[ "$rc" -ne 0 ]]; then
    local current_status
    current_status="$(cat "$RUN_ROOT/state/status" 2>/dev/null || true)"
    if [[ "$current_status" != failed* ]]; then
      printf 'failed\n' > "$RUN_ROOT/state/status"
    fi
    date -Is > "$RUN_ROOT/state/finished_at"
  fi
}
trap on_exit EXIT

printf 'running\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/started_at"
run_stage tf 3000 "$((BASE_PORT + 1))"; export_stage tf 3000 student
run_stage cd 2000 "$((BASE_PORT + 2))"; export_stage cd 2000 student
run_stage sf 1000 "$((BASE_PORT + 3))"; export_stage sf 1000 student_ema
ema_hash="$(sha256sum "$RUN_ROOT/sf/checkpoints/checkpoint-1000/ema/student.safetensors" | awk '{print $1}')"
export_hash="$(sha256sum "$RUN_ROOT/export/sf/transformer/model.safetensors" | awk '{print $1}')"
printf 'checkpoint_ema %s\nexported_ema %s\n' "$ema_hash" "$export_hash" \
  > "$RUN_ROOT/state/ema_sha256.txt"
if [[ "$ema_hash" != "$export_hash" ]]; then
  printf 'failed_ema_hash_mismatch\n' > "$RUN_ROOT/state/status"
  exit 1
fi
printf 'completed\n' > "$RUN_ROOT/state/status"; date -Is > "$RUN_ROOT/state/finished_at"
"""


SEQUENCE = r"""#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?prepared experiment root}"
WANDB_MODE="${WANDB_MODE:-online}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    echo "PREFLIGHT_ONLY must be 0 or 1, got: $PREFLIGHT_ONLY" >&2
    exit 2
    ;;
esac
case "$WANDB_MODE" in
  online)
    if [[ "$PREFLIGHT_ONLY" == 0 && -z "${WANDB_API_KEY:-}" ]]; then
      echo "WANDB_MODE=online requires WANDB_API_KEY at runtime." >&2
      exit 2
    fi
    ;;
  offline)
    unset WANDB_API_KEY
    ;;
  *)
    echo "Unsupported WANDB_MODE=$WANDB_MODE; expected online or offline." >&2
    exit 2
    ;;
esac
export WANDB_MODE PREFLIGHT_ONLY
for condition in A12 A13 A14 A15; do
  bash "$ROOT/scripts/run_condition.sh" "$condition" "$ROOT/$condition" "$((29800 + 10#${condition#A} * 10))"
done
"""


README = """# OpenVid causal A12-A15 plan (NOT STARTED)

Prepared from commit `{commit}`. This directory contains configuration only;
no training command was launched.

Data source: `{data}` (4494 parquet files, about 5.5 TiB). It is owned by
`vlm-s4duan`; filesystem read permission exists, but obtain the owner's consent
before launching and coordinate I/O. Do not write into that directory.

The opt-in streaming loader projects only the 15 T2V columns, reads each
assigned row group sequentially, and stores its JSON manifest at
`{manifest}` in user-owned Lustre. It uses zero DataLoader workers and never
writes a cache or index into the shared source tree.

All stages use 4 GPUs, microbatch 2/rank, gradient accumulation 8, hence global
batch = 2 * 4 * 8 = 64. `dataloader_num_workers=0` limits shared-memory and
Lustre prefetch pressure.

All four conditions use exactly 21 latent frames / 81 raw frames in TF, CD,
SF, and validation. Chunk-3 conditions therefore use seven identical
three-latent blocks. A15 is length-matched and uses framewise blocks.

A15 "framewise" means `num_frames_per_block=1` on every causal role, plus
`method.chunk_size=1` in TF and SF. Causal CD has no independent-frame timestep
option: it samples one t/t_next pair and broadcasts it over T, so A15 CD is
framewise causal attention but not framewise diffusion-time sampling.

The requested LR/betas apply to each stage's main optimizer: 2e-6 and
(0.0, 0.999). SF critic keeps the proven DMD value 4e-7 with (0.0, 0.999).

To launch one condition later:

    export WANDB_API_KEY=...
    bash {root}/scripts/run_condition.sh A12 {root}/A12 29820

To run all four sequentially on one 4-GPU node:

    export WANDB_API_KEY=...
    bash {root}/scripts/run_all_sequential.sh {root}

Online W&B is the default. For an intentionally offline launch, omit the key
and set `WANDB_MODE=offline`. Training checkpoints still resume from `latest`,
but W&B does not merge separate offline process restarts into one run; sync the
resulting offline runs individually later.

To validate all three configs for one condition without starting training,
creating W&B state, or requiring a key:

    PREFLIGHT_ONLY=1 bash {root}/scripts/run_condition.sh A12 {root}/A12 29820

`PREFLIGHT_ONLY=1` is also supported by `run_all_sequential.sh`; it validates
all twelve configs and exits before queue state or checkpoint checks.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    repo = str(Path(args.repo).resolve())
    root = str(Path(args.experiment_root))

    config_root = Path(repo) / "examples/train/configs/ablation/openvid_causal_a12_a15"
    scripts_root = Path(repo) / "scripts/train/openvid_causal_a12_a15"
    config_root.mkdir(parents=True, exist_ok=True)
    scripts_root.mkdir(parents=True, exist_ok=True)
    experiment_root = Path(root)
    (experiment_root / "scripts").mkdir(parents=True, exist_ok=True)

    train_stage = TRAIN_STAGE.replace("__REPO__", repo).replace("__COMMIT__", REQUIRED_ANCESTOR)
    queue = QUEUE.replace("__REPO__", repo)
    (scripts_root / "train_stage.sh").write_text(train_stage, encoding="utf-8")
    (scripts_root / "run_condition.sh").write_text(queue, encoding="utf-8")
    (scripts_root / "run_all_sequential.sh").write_text(SEQUENCE, encoding="utf-8")
    for path in scripts_root.glob("*.sh"):
        path.chmod(0o755)

    for condition, spec in CONDITIONS.items():
        run_root = str(experiment_root / condition)
        rendered = configs(run_root, condition, spec, repo)
        repo_condition = config_root / condition
        repo_condition.mkdir(parents=True, exist_ok=True)
        run_condition = experiment_root / condition
        for subdir in ("tf", "cd", "sf"):
            (run_condition / subdir / "config").mkdir(parents=True, exist_ok=True)
            (run_condition / subdir / "logs").mkdir(parents=True, exist_ok=True)
            (run_condition / subdir / "state").mkdir(parents=True, exist_ok=True)
            (run_condition / subdir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_condition / subdir / "validation").mkdir(parents=True, exist_ok=True)
            text = yaml.safe_dump(rendered[subdir], sort_keys=False)
            (repo_condition / f"{subdir}.yaml").write_text(text, encoding="utf-8")
            (run_condition / subdir / "config/run.yaml").write_text(text, encoding="utf-8")
        (run_condition / "scripts").mkdir(parents=True, exist_ok=True)
        (run_condition / "state").mkdir(parents=True, exist_ok=True)
        (run_condition / "scripts/train_stage.sh").write_text(train_stage, encoding="utf-8")
        (run_condition / "state/READY_NOT_STARTED").write_text("prepared\n", encoding="utf-8")
        (run_condition / "scripts/train_stage.sh").chmod(0o755)

    (experiment_root / "scripts/run_condition.sh").write_text(queue, encoding="utf-8")
    (experiment_root / "scripts/run_all_sequential.sh").write_text(SEQUENCE, encoding="utf-8")
    for path in (experiment_root / "scripts").glob("*.sh"):
        path.chmod(0o755)
    readme = README.format(
        commit=REQUIRED_ANCESTOR,
        data=DATA_PATH,
        manifest=STREAMING_MANIFEST_PATH,
        root=root,
    )
    (config_root / "README.md").write_text(readme, encoding="utf-8")
    (experiment_root / "README.md").write_text(readme, encoding="utf-8")
    (experiment_root / "READY_NOT_STARTED").write_text("prepared; training not launched\n", encoding="utf-8")

    print(config_root)
    print(scripts_root)
    print(experiment_root)


if __name__ == "__main__":
    main()
