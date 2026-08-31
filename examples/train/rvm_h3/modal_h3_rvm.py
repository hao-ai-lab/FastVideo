# pyright: reportAttributeAccessIssue=false
"""Launch one- or four-GPU FastH3 RVM correctness pilots on Modal.

The launcher follows the operational pattern used by the recent FastVideo
VPTD/PT-PDD and DiffusionNFT assets: clone the exact experiment branch at
runtime, use persistent model/data and run volumes, execute strict preflights
before optimization, and commit logs/checkpoints even when a run fails.

Examples:

    # Download/cache assets, run public FastH3 inference, load every reward,
    # build the real training config, and stop before optimization.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 1 --mode preflight

    # Four optimizer updates on one B200. Includes baseline validation at step 0.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 1 --mode smoke

    # Production-capacity LoRA and SP4 on four B200s, but only ten updates.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 4 --mode pilot --max-steps 10 --eval-prompts 8

    # Resume the same persistent run directory.
    modal run examples/train/rvm_h3/modal_h3_rvm.py \
        --gpus 4 --mode pilot --run-name my-rvm-pilot --resume
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any

import modal


FASTVIDEO_REPOSITORY = "https://github.com/Abecid/FastVideo.git"
DEFAULT_BRANCH = "adam/h3-rvm-posttraining"
DEFAULT_IMAGE = "ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:latest"
DEFAULT_1GPU_CONFIG = "examples/train/configs/rl/minimax_h3/rvm_h3_modal_1gpu.yaml"
DEFAULT_4GPU_CONFIG = "examples/train/configs/rl/minimax_h3/rvm_h3_modal_4gpu.yaml"

IMAGE_REF = os.environ.get("FASTVIDEO_MODAL_IMAGE", DEFAULT_IMAGE)
GPU_1 = os.environ.get("H3_RVM_MODAL_GPU_1", "B200")
GPU_4 = os.environ.get("H3_RVM_MODAL_GPU_4", "B200:4")
SECRET_NAME = os.environ.get("H3_RVM_MODAL_SECRET", "fastvideo-training")

image = (
    modal.Image.from_registry(IMAGE_REF, add_python="3.12")
    .apt_install(
        "ffmpeg",
        "git",
        "git-lfs",
        "libgl1",
        "libglib2.0-0",
        "ninja-build",
    )
    .run_commands("python -m pip install --upgrade uv")
)

app = modal.App("fastvideo-h3-rvm-pilot")
asset_volume = modal.Volume.from_name(
    "fastvideo-h3-rvm-assets",
    create_if_missing=True,
)
run_volume = modal.Volume.from_name(
    "fastvideo-h3-rvm-runs",
    create_if_missing=True,
)
training_secret = modal.Secret.from_name(SECRET_NAME)

_ALLOWED_MODES = {"prepare", "preflight", "smoke", "pilot", "all"}


def _run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("+", shlex.join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _capture(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
) -> str:
    print("+", shlex.join(command), flush=True)
    return subprocess.check_output(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _safe_run_name(value: str, *, gpu_count: int) -> str:
    text = value.strip()
    if not text or text == "auto":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        text = f"h3-rvm-{gpu_count}gpu-{timestamp}"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    if not sanitized or sanitized in {".", ".."}:
        raise ValueError(f"Invalid run_name: {value!r}")
    return sanitized


def _checkout_fastvideo(
    *,
    workspace: Path,
    branch: str,
    commit: str,
    env: dict[str, str],
    log_dir: Path,
) -> tuple[Path, str]:
    destination = workspace / "FastVideo"
    shutil.rmtree(destination, ignore_errors=True)
    command = [
        "git",
        "clone",
        "--filter=blob:none",
        "--no-tags",
        "--branch",
        branch,
        FASTVIDEO_REPOSITORY,
        str(destination),
    ]
    _run_logged(
        command,
        cwd=workspace,
        env=env,
        log_path=log_dir / "00_clone.log",
    )
    if commit.strip():
        _run_logged(
            ["git", "fetch", "--depth", "1", "origin", commit],
            cwd=destination,
            env=env,
            log_path=log_dir / "00_clone.log",
        )
        _run_logged(
            ["git", "checkout", "--detach", commit],
            cwd=destination,
            env=env,
            log_path=log_dir / "00_clone.log",
        )
    resolved = _capture(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        env=env,
    )
    if commit.strip() and resolved != commit:
        raise RuntimeError(f"Requested commit {commit}, checked out {resolved}")
    return destination, resolved


def _install_runtime(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
) -> None:
    _run_logged(
        ["uv", "pip", "install", "--system", "-e", ".[eval,test]"],
        cwd=repository,
        env=env,
        log_path=log_dir / "01_install.log",
    )
    _run_logged(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "decord",
            "hpsv3==1.0.0",
            "liger-kernel",
            "qwen-vl-utils",
            "safetensors",
            "trl>=0.18",
        ],
        cwd=repository,
        env=env,
        log_path=log_dir / "01_install.log",
    )


def _link_artifacts(repository: Path, artifact_root: Path) -> None:
    link = repository / "artifacts" / "rvm_h3"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(artifact_root, target_is_directory=True)


def _parquet_rows(root: Path) -> int:
    files = sorted(root.rglob("*.parquet")) if root.is_dir() else []
    if not files:
        return 0
    import pyarrow.parquet as pq

    return sum(int(pq.ParquetFile(path).metadata.num_rows) for path in files)


def _prepare_assets(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
    gpu_count: int,
    max_train_prompts: int,
    eval_prompts: int,
    smoke_prompts: int,
    force_prepare: bool,
) -> dict[str, int]:
    artifact_root = Path(env["RVM_ARTIFACT_ROOT"])
    model = Path(env["FASTH3_MODEL_DIR"])
    reward_checkpoint = Path(env["VIDEOALIGN_CHECKPOINT_PATH"])
    reward_runtime = Path(env["VIDEOALIGN_RUNTIME_PATH"])

    models_ready = (
        (model / "transformer").is_dir()
        and (model / "vae").is_dir()
        and (model / "text_encoder").is_dir()
        and reward_checkpoint.is_dir()
        and (reward_runtime / "inference.py").is_file()
    )
    if force_prepare or not models_ready:
        _run_logged(
            ["bash", "examples/train/rvm_h3/01_download_models.sh"],
            cwd=repository,
            env=env,
            log_path=log_dir / "02_download_models.log",
        )
        asset_volume.commit()

    expected = {
        "train": int(max_train_prompts),
        "eval": int(eval_prompts),
        "smoke": int(smoke_prompts),
    }
    roots = {
        "train": Path(env["RVM_TRAIN_DATA"]),
        "eval": Path(env["RVM_EVAL_DATA"]),
        "smoke": Path(env["RVM_SMOKE_DATA"]),
    }
    current = {name: _parquet_rows(path) for name, path in roots.items()}
    needs_data = force_prepare or any(current[name] < expected[name] for name in expected)
    if needs_data:
        data_env = dict(env)
        data_env.update(
            {
                "RVM_EVAL_PROMPTS": str(eval_prompts),
                "RVM_FORCE_PREPROCESS": "1",
                "RVM_MAX_TRAIN_PROMPTS": str(max_train_prompts),
                "RVM_PREPROCESS_GPUS": str(gpu_count),
                "RVM_SMOKE_PROMPTS": str(smoke_prompts),
            }
        )
        _run_logged(
            ["bash", "examples/train/rvm_h3/02_prepare_dataset.sh"],
            cwd=repository,
            env=data_env,
            log_path=log_dir / "03_prepare_dataset.log",
        )
        asset_volume.commit()
        current = {name: _parquet_rows(path) for name, path in roots.items()}

    missing = {
        name: {"required": expected[name], "found": current[name]}
        for name in expected
        if current[name] < expected[name]
    }
    if missing:
        raise RuntimeError(f"Prepared prompt datasets are incomplete: {missing}")
    _link_artifacts(repository, artifact_root)
    return current


def _run_preflight(
    *,
    repository: Path,
    env: dict[str, str],
    log_dir: Path,
    run_dir: Path,
    gpu_count: int,
) -> None:
    preflight_env = dict(env)
    preflight_env["RVM_PREFLIGHT_REWARD_OUTPUT"] = str(
        run_dir / "preflight_reward_scores.json"
    )
    _run_logged(
        ["bash", "examples/train/rvm_h3/03_public_inference_smoke.sh"],
        cwd=repository,
        env=preflight_env,
        log_path=log_dir / "04_public_inference.log",
    )
    inference_root = Path(env["RVM_ARTIFACT_ROOT"]) / "inference_smoke"
    if inference_root.is_dir():
        shutil.copytree(
            inference_root,
            run_dir / "public_inference_smoke",
            dirs_exist_ok=True,
        )
    _run_logged(
        ["bash", "examples/train/rvm_h3/03_preflight_1gpu.sh"],
        cwd=repository,
        env=preflight_env,
        log_path=log_dir / "05_preflight.log",
    )

    if gpu_count == 4:
        dry_run_env = dict(env)
        dry_run_env["NUM_GPUS"] = "4"
        _run_logged(
            [
                "bash",
                "examples/train/run.sh",
                DEFAULT_4GPU_CONFIG,
                "--dry-run",
                "--training.distributed.num_gpus",
                "4",
                "--training.distributed.sp_size",
                "4",
                "--training.distributed.tp_size",
                "1",
                "--training.distributed.hsdp_replicate_dim",
                "1",
                "--training.distributed.hsdp_shard_dim",
                "4",
            ],
            cwd=repository,
            env=dry_run_env,
            log_path=log_dir / "05_preflight_4gpu_config.log",
        )


def _checkpoint_exists(output_dir: Path) -> bool:
    return any(
        (path / "dcp" / ".metadata").is_file()
        for path in output_dir.glob("checkpoint-*")
        if path.is_dir()
    )


def _training_command(
    *,
    gpu_count: int,
    config: str,
    output_dir: Path,
    run_name: str,
    max_steps: int,
    eval_prompts: int,
    learning_rate: float,
    resume: bool,
    env: dict[str, str],
) -> list[str]:
    data_path = env["RVM_SMOKE_DATA"] if gpu_count == 1 else env["RVM_TRAIN_DATA"]
    checkpoint_interval = max(1, max_steps // 2)
    command = [
        "bash",
        "examples/train/run.sh",
        config,
        "--models.student.init_from",
        env["FASTH3_MODEL_DIR"],
        "--training.model_path",
        env["FASTH3_MODEL_DIR"],
        "--training.data.data_path",
        data_path,
        "--method.validation.data_path",
        env["RVM_EVAL_DATA"],
        "--method.validation.num_prompts",
        str(min(100, eval_prompts)),
        "--method.validation.log_sample_limit",
        str(min(8, eval_prompts)),
        "--method.validation.every_steps",
        "0",
        "--training.loop.max_train_steps",
        str(max_steps),
        "--training.checkpoint.output_dir",
        str(output_dir),
        "--training.checkpoint.training_state_checkpointing_steps",
        str(checkpoint_interval),
        "--training.tracker.run_name",
        run_name,
        "--training.distributed.num_gpus",
        str(gpu_count),
        "--training.distributed.sp_size",
        str(gpu_count),
        "--training.distributed.tp_size",
        "1",
        "--training.distributed.hsdp_replicate_dim",
        "1",
        "--training.distributed.hsdp_shard_dim",
        str(gpu_count),
    ]
    if learning_rate > 0:
        command.extend(
            [
                "--training.optimizer.learning_rate",
                str(learning_rate),
            ]
        )
    if resume and _checkpoint_exists(output_dir):
        command.extend(
            [
                "--training.checkpoint.resume_from_checkpoint",
                "latest",
            ]
        )
    return command


def _environment_report(
    *,
    repository: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, command in {
        "git_head": ["git", "rev-parse", "HEAD"],
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        "python": ["python", "--version"],
    }.items():
        try:
            report[name] = _capture(command, cwd=repository, env=env)
        except Exception as exc:
            report[name] = f"ERROR: {exc}"
    try:
        versions = _capture(
            [
                "python",
                "-c",
                (
                    "import json, torch, transformers, diffusers; "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'cuda': torch.version.cuda, "
                    "'transformers': transformers.__version__, "
                    "'diffusers': diffusers.__version__}, sort_keys=True))"
                ),
            ],
            cwd=repository,
            env=env,
        )
        report["packages"] = json.loads(versions.splitlines()[-1])
    except Exception as exc:
        report["packages"] = {"error": str(exc)}
    return report


def _run_job(
    *,
    gpu_count: int,
    mode: str,
    branch: str,
    commit: str,
    config: str,
    run_name: str,
    max_steps: int,
    eval_prompts: int,
    max_train_prompts: int,
    smoke_prompts: int,
    learning_rate: float,
    force_prepare: bool,
    resume: bool,
    skip_preflight: bool,
) -> dict[str, Any]:
    if gpu_count not in {1, 4}:
        raise ValueError("gpu_count must be one or four")
    mode = mode.strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {sorted(_ALLOWED_MODES)}")
    if max_steps < 0:
        raise ValueError("max_steps must be zero (use config default) or positive")
    if not 1 <= eval_prompts <= 100:
        raise ValueError("eval_prompts must be in [1, 100]")
    if max_train_prompts <= 0 or smoke_prompts <= 0:
        raise ValueError("prompt counts must be positive")
    if learning_rate < 0:
        raise ValueError("learning_rate must be zero (use config) or positive")

    effective_steps = max_steps or (4 if gpu_count == 1 else 10)
    selected_config = config.strip() or (
        DEFAULT_1GPU_CONFIG if gpu_count == 1 else DEFAULT_4GPU_CONFIG
    )
    safe_name = _safe_run_name(run_name, gpu_count=gpu_count)
    run_dir = Path("/runs/h3-rvm") / safe_name
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with suppress(Exception):
        asset_volume.reload()
    with suppress(Exception):
        run_volume.reload()

    workspace = Path("/workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    env.update(
        {
            "FASTH3_MODEL_DIR": "/cache/rvm_h3/models/fasth3",
            "FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN_H3",
            "FASTVIDEO_MINIMAX_H3_FUSIONS": "0",
            "FASTVIDEO_RVM_VAE_DECODE_BATCH_SIZE": "1",
            "FASTVIDEO_VSA_CUTEDSL": "0",
            "FASTVIDEO_VSA_SM100A": "0",
            "HF_HOME": "/cache/huggingface",
            "LOG_DIR": str(log_dir),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RVM_ARTIFACT_ROOT": "/cache/rvm_h3",
            "RVM_EVAL_DATA": "/cache/rvm_h3/data/eval",
            "RVM_PROMPT_DIR": "/cache/rvm_h3/prompts",
            "RVM_REWARD_ROOT": "/cache/rvm_h3/rewards",
            "RVM_SKIP_CONDA": "1",
            "RVM_SMOKE_DATA": "/cache/rvm_h3/data/train_smoke",
            "RVM_TRAIN_DATA": "/cache/rvm_h3/data/train",
            "TOKENIZERS_PARALLELISM": "false",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TRANSFORMERS_CACHE": "/cache/huggingface/hub",
            "VIDEOALIGN_CHECKPOINT_PATH": "/cache/rvm_h3/rewards/VideoReward",
            "VIDEOALIGN_RUNTIME_PATH": "/cache/rvm_h3/rewards/VideoAlign",
            "WANDB_DIR": str(run_dir / "wandb"),
            "WANDB_MODE": (
                "online" if (env.get("WANDB_API_KEY") or "").strip() else "offline"
            ),
            "WANDB_RESUME": "allow",
            "WANDB_RUN_ID": hashlib.sha1(safe_name.encode("utf-8")).hexdigest()[:16],
        }
    )

    status: dict[str, Any] = {
        "branch": branch,
        "commit_requested": commit or None,
        "config": selected_config,
        "eval_prompts": int(eval_prompts),
        "gpu_count": int(gpu_count),
        "learning_rate_override": float(learning_rate),
        "max_steps": int(effective_steps),
        "mode": mode,
        "run_dir": str(run_dir),
        "run_name": safe_name,
        "status": "starting",
    }
    _write_json(run_dir / "modal_manifest.json", status)

    repository: Path | None = None
    try:
        repository, resolved = _checkout_fastvideo(
            workspace=workspace,
            branch=branch,
            commit=commit,
            env=env,
            log_dir=log_dir,
        )
        env["PYTHONPATH"] = str(repository)
        status["commit_resolved"] = resolved
        status["status"] = "installing"
        _write_json(run_dir / "modal_manifest.json", status)

        _install_runtime(repository=repository, env=env, log_dir=log_dir)
        _link_artifacts(repository, Path(env["RVM_ARTIFACT_ROOT"]))
        status["environment"] = _environment_report(repository=repository, env=env)

        status["status"] = "preparing_assets"
        _write_json(run_dir / "modal_manifest.json", status)
        status["dataset_rows"] = _prepare_assets(
            repository=repository,
            env=env,
            log_dir=log_dir,
            gpu_count=gpu_count,
            max_train_prompts=max_train_prompts,
            eval_prompts=eval_prompts,
            smoke_prompts=smoke_prompts,
            force_prepare=force_prepare,
        )

        should_preflight = mode in {"preflight", "smoke", "pilot", "all"} and not skip_preflight
        if should_preflight:
            status["status"] = "preflight"
            _write_json(run_dir / "modal_manifest.json", status)
            _run_preflight(
                repository=repository,
                env=env,
                log_dir=log_dir,
                run_dir=run_dir,
                gpu_count=gpu_count,
            )

        should_train = mode in {"smoke", "pilot", "all"}
        if should_train:
            status["status"] = "training"
            _write_json(run_dir / "modal_manifest.json", status)
            train_env = dict(env)
            train_env["NUM_GPUS"] = str(gpu_count)
            train_env["RVM_SP_SIZE"] = str(gpu_count)
            command = _training_command(
                gpu_count=gpu_count,
                config=selected_config,
                output_dir=run_dir,
                run_name=safe_name,
                max_steps=effective_steps,
                eval_prompts=eval_prompts,
                learning_rate=learning_rate,
                resume=resume,
                env=env,
            )
            _run_logged(
                command,
                cwd=repository,
                env=train_env,
                log_path=log_dir / "06_training.log",
            )

        from fastvideo.train.methods.rl.rvm_local_metrics import (
            collect_initial_reward_results,
        )

        reward_results = collect_initial_reward_results(run_dir)
        _write_json(run_dir / "initial_reward_results.json", reward_results)
        status["initial_reward_results"] = reward_results
        status["status"] = "succeeded"
        _write_json(run_dir / "modal_manifest.json", status)
        return status
    except Exception as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)
        _write_json(run_dir / "modal_manifest.json", status)
        raise
    finally:
        with suppress(Exception):
            run_volume.commit()
        with suppress(Exception):
            asset_volume.commit()


@app.function(
    image=image,
    gpu=GPU_1,
    cpu=32,
    memory=131_072,
    ephemeral_disk=500_000,
    timeout=86_400,
    startup_timeout=4_800,
    secrets=[training_secret],
    volumes={"/cache": asset_volume, "/runs": run_volume},
)
def run_1gpu(
    *,
    mode: str = "smoke",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    return _run_job(
        gpu_count=1,
        mode=mode,
        branch=branch,
        commit=commit,
        config=config,
        run_name=run_name,
        max_steps=max_steps,
        eval_prompts=eval_prompts,
        max_train_prompts=max_train_prompts,
        smoke_prompts=smoke_prompts,
        learning_rate=learning_rate,
        force_prepare=force_prepare,
        resume=resume,
        skip_preflight=skip_preflight,
    )


@app.function(
    image=image,
    gpu=GPU_4,
    cpu=64,
    memory=262_144,
    ephemeral_disk=500_000,
    timeout=86_400,
    startup_timeout=4_800,
    secrets=[training_secret],
    volumes={"/cache": asset_volume, "/runs": run_volume},
)
def run_4gpu(
    *,
    mode: str = "pilot",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> dict[str, Any]:
    return _run_job(
        gpu_count=4,
        mode=mode,
        branch=branch,
        commit=commit,
        config=config,
        run_name=run_name,
        max_steps=max_steps,
        eval_prompts=eval_prompts,
        max_train_prompts=max_train_prompts,
        smoke_prompts=smoke_prompts,
        learning_rate=learning_rate,
        force_prepare=force_prepare,
        resume=resume,
        skip_preflight=skip_preflight,
    )


@app.local_entrypoint()
def main(
    gpus: int = 1,
    mode: str = "smoke",
    branch: str = DEFAULT_BRANCH,
    commit: str = "",
    config: str = "",
    run_name: str = "auto",
    max_steps: int = 0,
    eval_prompts: int = 8,
    max_train_prompts: int = 64,
    smoke_prompts: int = 16,
    learning_rate: float = 0.0,
    force_prepare: bool = False,
    resume: bool = False,
    skip_preflight: bool = False,
) -> None:
    kwargs = {
        "mode": mode,
        "branch": branch,
        "commit": commit,
        "config": config,
        "run_name": run_name,
        "max_steps": max_steps,
        "eval_prompts": eval_prompts,
        "max_train_prompts": max_train_prompts,
        "smoke_prompts": smoke_prompts,
        "learning_rate": learning_rate,
        "force_prepare": force_prepare,
        "resume": resume,
        "skip_preflight": skip_preflight,
    }
    if gpus == 1:
        result = run_1gpu.remote(**kwargs)
    elif gpus == 4:
        result = run_4gpu.remote(**kwargs)
    else:
        raise ValueError("--gpus must be 1 or 4")
    print(json.dumps(result, indent=2, sort_keys=True))
