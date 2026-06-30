"""Run DMDR Wan video RL training on Modal.

This launcher is intentionally thin: prompt extraction, text-only
preprocessing, VideoReward checkpoint setup, reward preflight, and generated
run-config writing are handled by examples/train/prepare_dmdr_assets.py so the
reproducible experiment path is not Modal-specific.

Expected Modal resources in the hao-ai-lab workspace:
- Volume `fastvideo-data` mounted under the repo for text-only parquet prompts.
- Volume `fastvideo-runs` mounted under the repo for checkpoints/logs.
- Volume `fastvideo-cache` mounted under the repo for Hugging Face caches.
- Secrets `wandb-adamlee00` and `hf-adamlee00`.
"""

from __future__ import annotations

import modal

app = modal.App("fastvideo-dmdr-wan")

CONFIG_PATH = "examples/train/configs/rl/wan/dmdr_videoalign.yaml"
PROJECT_ROOT = "/root/FastVideo"
MODAL_DATA_ROOT = f"{PROJECT_ROOT}/.modal_data"
MODAL_CACHE_ROOT = f"{PROJECT_ROOT}/.modal_cache"
OUTPUT_DIR_BASE = f"{PROJECT_ROOT}/outputs/dmdr_wan"
WANDB_ENTITY = "adamlee00"
WANDB_SECRET_NAME = "wandb-adamlee00"
HF_SECRET_NAME = "hf-adamlee00"
DEFAULT_GPU_TYPE = "A100-80GB"
DEFAULT_NUM_GPUS = 4
DEFAULT_HSDP_REPLICATE_DIM = 1
DEFAULT_HSDP_SHARD_DIM = DEFAULT_NUM_GPUS
DEFAULT_MAX_TRAIN_STEPS = 100
DEFAULT_CHECKPOINT_STEPS = 20
DEFAULT_NUM_SAMPLES_PER_PROMPT = 4
DEFAULT_NUM_BATCHES_PER_EPOCH = 1
DEFAULT_COLLECTION_BATCH_SIZE = 1
DEFAULT_INNER_EPOCHS = 1
DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_LEARNING_RATE = 2.0e-6
DEFAULT_COLD_START_STEPS = 0
DEFAULT_DYNAMIC_STEP = 10_000
DEFAULT_GUIDANCE_UPDATE_RATIO = 1
DEFAULT_SAMPLE_NUM_STEPS = 4
DEFAULT_SAMPLE_FLOW_SHIFT = 8.0
DEFAULT_SAMPLE_GUIDANCE_SCALE = 6.0
DEFAULT_VALIDATION_EVERY_STEPS = 10
DEFAULT_VALIDATION_NUM_STEPS = 4
DEFAULT_VALIDATION_NUM_PROMPTS = 4
DEFAULT_VALIDATION_BATCH_SIZE = 1
DEFAULT_NUM_FRAMES = 17
DEFAULT_NUM_LATENT_T = 0
DEFAULT_LOG_SAMPLE_MAX_VIDEOS = 4
DEFAULT_PREPROCESS_BATCH_SIZE = 128
DEFAULT_PREPROCESS_NUM_GPUS = 1
DEFAULT_DATASET = "world-r1-enhanced"
DEFAULT_REWARD = "videoalign"
DEFAULT_MAX_PROMPTS = "64"
VIDEOALIGN_CKPT_ROOT = f"{MODAL_CACHE_ROOT}/VideoReward"
MODAL_CONTEXT_IGNORE = [
    ".git/fsmonitor--daemon.ipc",
    ".cache",
    ".cache/**",
    "__pycache__",
    "**/__pycache__/**",
    "*.pyc",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "logs",
    "logs/**",
    "outputs",
    "outputs/**",
    "wandb",
    "wandb/**",
]

data_vol = modal.Volume.from_name("fastvideo-data")
runs_vol = modal.Volume.from_name("fastvideo-runs")
cache_vol = modal.Volume.from_name("fastvideo-cache")

image = (modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",
).entrypoint([]).apt_install(
    "git",
    "git-lfs",
    "ffmpeg",
    "libgl1",
    "libglib2.0-0",
    "build-essential",
    "ninja-build",
    "cmake",
).pip_install("uv").add_local_dir(
    ".",
    PROJECT_ROOT,
    copy=True,
    ignore=MODAL_CONTEXT_IGNORE,
).run_commands(
    "cd /root/FastVideo && uv pip install --system --prerelease=allow -e .",
    "uv pip install --system --prerelease=allow "
    "--index-url https://download.pytorch.org/whl/cu128 "
    "--upgrade torch torchvision torchaudio",
    "uv pip install --system --no-cache-dir "
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/"
    "releases/download/v0.7.16/"
    "flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl",
    "cd /root/FastVideo && uv pip install --system "
    "-r examples/train/requirements-dmdr.txt",
    "python -c 'import flash_attn; print(\"flash_attn ok\")'",
    "python -c 'import cv2, imageio; print(\"video io ok\")'",
    "python -c 'import cloudpickle, pyarrow, torchdata; print(\"training deps ok\")'",
    "python -c 'import datasets, peft, qwen_vl_utils, safetensors, timm, trl; "
    "from transformers import Qwen2VLForConditionalGeneration; print(\"reward deps ok\")'",
).env({
    "WANDB_MODE": "online",
    "WANDB_ENTITY": WANDB_ENTITY,
    "TOKENIZERS_PARALLELISM": "false",
    "NUM_GPUS": str(DEFAULT_NUM_GPUS),
    "HF_HOME": f"{MODAL_CACHE_ROOT}/huggingface",
    "HF_HUB_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
    "TRANSFORMERS_CACHE": f"{MODAL_CACHE_ROOT}/huggingface",
    "VIDEOALIGN_CHECKPOINT_PATH": VIDEOALIGN_CKPT_ROOT,
    "FASTVIDEO_ATTENTION_BACKEND": "FLASH_ATTN",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}))


@app.function(
    image=image,
    gpu=f"{DEFAULT_GPU_TYPE}:{DEFAULT_NUM_GPUS}",
    timeout=3 * 60 * 60,
    volumes={
        MODAL_DATA_ROOT: data_vol,
        f"{PROJECT_ROOT}/outputs": runs_vol,
        MODAL_CACHE_ROOT: cache_vol,
    },
    secrets=[
        modal.Secret.from_name(WANDB_SECRET_NAME),
        modal.Secret.from_name(HF_SECRET_NAME),
    ],
)
def train(
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    checkpoint_steps: int = DEFAULT_CHECKPOINT_STEPS,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
    num_samples_per_prompt: int = DEFAULT_NUM_SAMPLES_PER_PROMPT,
    num_batches_per_epoch: int = DEFAULT_NUM_BATCHES_PER_EPOCH,
    collection_batch_size: int = DEFAULT_COLLECTION_BATCH_SIZE,
    inner_epochs: int = DEFAULT_INNER_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    cold_start_steps: int = DEFAULT_COLD_START_STEPS,
    dynamic_step: int = DEFAULT_DYNAMIC_STEP,
    guidance_update_ratio: int = DEFAULT_GUIDANCE_UPDATE_RATIO,
    sample_num_steps: int = DEFAULT_SAMPLE_NUM_STEPS,
    sample_flow_shift: float = DEFAULT_SAMPLE_FLOW_SHIFT,
    sample_guidance_scale: float = DEFAULT_SAMPLE_GUIDANCE_SCALE,
    validation_every_steps: int = DEFAULT_VALIDATION_EVERY_STEPS,
    validation_num_steps: int = DEFAULT_VALIDATION_NUM_STEPS,
    validation_num_prompts: int = DEFAULT_VALIDATION_NUM_PROMPTS,
    validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_latent_t: int = DEFAULT_NUM_LATENT_T,
    log_sample_max_videos: int = DEFAULT_LOG_SAMPLE_MAX_VIDEOS,
    preprocess_batch_size: int = DEFAULT_PREPROCESS_BATCH_SIZE,
    preprocess_num_gpus: int = DEFAULT_PREPROCESS_NUM_GPUS,
    dataset: str = DEFAULT_DATASET,
    reward: str = DEFAULT_REWARD,
    max_prompts: str = DEFAULT_MAX_PROMPTS,
    check_rewards: bool = True,
):
    from datetime import datetime, timezone
    import json
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(PROJECT_ROOT)
    dataset = dataset.strip().lower()
    reward = reward.strip().lower()
    preprocess_batch_size = int(preprocess_batch_size)
    preprocess_num_gpus = int(preprocess_num_gpus)
    log_sample_max_videos = int(log_sample_max_videos)
    checkpoint_steps = int(checkpoint_steps)
    cold_start_steps = int(cold_start_steps)
    dynamic_step = int(dynamic_step)
    guidance_update_ratio = int(guidance_update_ratio)
    if checkpoint_steps <= 0:
        raise ValueError("--checkpoint-steps must be positive")
    if preprocess_batch_size <= 0:
        raise ValueError("--preprocess-batch-size must be positive")
    if (num_frames - 1) % 4 != 0:
        raise ValueError("Wan frame counts must satisfy num_frames = (num_latent_t - 1) * 4 + 1. "
                         f"Got --num-frames {num_frames}; try 17 for smoke or 49 for a larger video run.")
    if sample_num_steps <= 0:
        raise ValueError("--sample-num-steps must be positive")
    if cold_start_steps < 0:
        raise ValueError("--cold-start-steps must be >= 0")
    if dynamic_step < 0:
        raise ValueError("--dynamic-step must be >= 0")
    if guidance_update_ratio <= 0:
        raise ValueError("--guidance-update-ratio must be positive")
    if validation_every_steps <= 0:
        raise ValueError("--validation-every-steps must be positive")
    if validation_num_steps <= 0:
        raise ValueError("--validation-num-steps must be positive")
    if validation_num_prompts <= 0:
        raise ValueError("--validation-num-prompts must be positive")
    if validation_batch_size <= 0:
        raise ValueError("--validation-batch-size must be positive")
    if preprocess_num_gpus != 1:
        raise ValueError("FastVideo text preprocessing currently supports "
                         "--preprocess-num-gpus 1 only.")
    if DEFAULT_HSDP_REPLICATE_DIM * DEFAULT_HSDP_SHARD_DIM != DEFAULT_NUM_GPUS:
        raise ValueError(
            "Invalid HSDP mesh: replicate_dim * shard_dim must equal "
            f"num_gpus ({DEFAULT_HSDP_REPLICATE_DIM} * "
            f"{DEFAULT_HSDP_SHARD_DIM} != {DEFAULT_NUM_GPUS}).")
    if (DEFAULT_NUM_GPUS * collection_batch_size) % num_samples_per_prompt != 0:
        raise ValueError("DMDR K-repeat sampling requires num_gpus * collection_batch_size "
                         "to be divisible by --num-samples-per-prompt "
                         f"({DEFAULT_NUM_GPUS} * {collection_batch_size} vs "
                         f"{num_samples_per_prompt}).")
    if (DEFAULT_NUM_GPUS * train_batch_size) % num_samples_per_prompt != 0:
        raise ValueError("DMDR training batches should keep full prompt repeat groups: "
                         "num_gpus * train_batch_size must be divisible by "
                         "--num-samples-per-prompt "
                         f"({DEFAULT_NUM_GPUS} * {train_batch_size} vs "
                         f"{num_samples_per_prompt}).")

    if output_dir:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = repo / output_path
        output_dir = str(output_path)
    else:
        output_dir = (f"{OUTPUT_DIR_BASE}_"
                      f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    if resume_from_checkpoint:
        resume_checkpoint_path = Path(resume_from_checkpoint)
        if str(resume_from_checkpoint).strip().lower() == "latest":
            has_checkpoint = any(Path(output_dir).glob("checkpoint-*"))
            if has_checkpoint:
                resume_from_checkpoint = "latest"
            else:
                print("resume_from_checkpoint=latest requested, but no checkpoint exists yet; "
                      "starting from the base model.", flush=True)
                resume_from_checkpoint = None
        elif not resume_checkpoint_path.is_absolute():
            resume_from_checkpoint = str(repo / resume_checkpoint_path)
    prep_cmd = [
        "python",
        "examples/train/prepare_dmdr_assets.py",
        "--repo-root",
        str(repo),
        "--config",
        CONFIG_PATH,
        "--data-root",
        MODAL_DATA_ROOT,
        "--cache-root",
        MODAL_CACHE_ROOT,
        "--output-dir",
        output_dir,
        "--run-config-dir",
        f"{PROJECT_ROOT}/outputs/dmdr_run_configs",
        "--videoalign-checkpoint-path",
        VIDEOALIGN_CKPT_ROOT,
        "--dataset",
        dataset,
        "--reward",
        reward,
        "--max-prompts",
        str(max_prompts),
        "--num-frames",
        str(num_frames),
        "--num-latent-t",
        str(num_latent_t),
        "--num-gpus",
        str(DEFAULT_NUM_GPUS),
        "--hsdp-replicate-dim",
        str(DEFAULT_HSDP_REPLICATE_DIM),
        "--hsdp-shard-dim",
        str(DEFAULT_HSDP_SHARD_DIM),
        "--max-train-steps",
        str(max_train_steps),
        "--gradient-accumulation-steps",
        str(gradient_accumulation_steps),
        "--learning-rate",
        str(learning_rate),
        "--num-samples-per-prompt",
        str(num_samples_per_prompt),
        "--num-batches-per-epoch",
        str(num_batches_per_epoch),
        "--collection-batch-size",
        str(collection_batch_size),
        "--inner-epochs",
        str(inner_epochs),
        "--train-batch-size",
        str(train_batch_size),
        "--cold-start-steps",
        str(cold_start_steps),
        "--dynamic-step",
        str(dynamic_step),
        "--guidance-update-ratio",
        str(guidance_update_ratio),
        "--sample-num-steps",
        str(sample_num_steps),
        "--sample-flow-shift",
        str(sample_flow_shift),
        "--sample-guidance-scale",
        str(sample_guidance_scale),
        "--validation-every-steps",
        str(validation_every_steps),
        "--validation-num-steps",
        str(validation_num_steps),
        "--validation-num-prompts",
        str(validation_num_prompts),
        "--validation-batch-size",
        str(validation_batch_size),
        "--log-sample-max-videos",
        str(log_sample_max_videos),
        "--preprocess-batch-size",
        str(preprocess_batch_size),
        "--preprocess-num-gpus",
        str(preprocess_num_gpus),
        "--json",
    ]
    if check_rewards:
        prep_cmd.append("--check-rewards")

    print("Preparing DMDR assets with tracked example CLI:", flush=True)
    completed = subprocess.run(
        prep_cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        check=True,
    )
    print(completed.stdout, end="", flush=True)
    summary = json.loads(completed.stdout.strip().splitlines()[-1])
    run_config_path = Path(summary["run_config"])
    parquet_dir = Path(summary["parquet_dir"])
    output_dir = summary["output_dir"]
    resolved_num_frames = int(summary["num_frames"])
    resolved_num_latent_t = int(summary["num_latent_t"])
    cache_vol.commit()
    data_vol.commit()

    print(f"Output dir: {output_dir}", flush=True)
    print(
        "DMDR probe settings: "
        f"max_train_steps={max_train_steps} "
        f"checkpoint_steps={checkpoint_steps} "
        f"resume_from_checkpoint={resume_from_checkpoint or ''} "
        f"num_samples_per_prompt={num_samples_per_prompt} "
        f"num_batches_per_epoch={num_batches_per_epoch} "
        f"collection_batch_size={collection_batch_size} "
        f"inner_epochs={inner_epochs} "
        f"train_batch_size={train_batch_size} "
        f"gradient_accumulation_steps={gradient_accumulation_steps} "
        f"learning_rate={learning_rate} "
        f"cold_start_steps={cold_start_steps} "
        f"dynamic_step={dynamic_step} "
        f"guidance_update_ratio={guidance_update_ratio} "
        f"sample_num_steps={sample_num_steps} "
        f"sample_flow_shift={sample_flow_shift} "
        f"sample_guidance_scale={sample_guidance_scale} "
        f"validation_every_steps={validation_every_steps} "
        f"validation_num_steps={validation_num_steps} "
        f"validation_num_prompts={validation_num_prompts} "
        f"validation_batch_size={validation_batch_size} "
        f"num_frames={resolved_num_frames} "
        f"num_latent_t={resolved_num_latent_t} "
        f"log_sample_max_videos={log_sample_max_videos} "
        f"preprocess_batch_size={preprocess_batch_size} "
        f"preprocess_num_gpus={preprocess_num_gpus} "
        f"dataset={dataset} "
        f"reward={reward} "
        f"max_prompts={max_prompts} "
        f"resolved_prompts={summary['prompt_count']}",
        flush=True,
    )

    cmd = [
        "bash",
        "examples/train/run.sh",
        str(run_config_path),
        "--training.data.data_path",
        str(parquet_dir),
        "--training.checkpoint.output_dir",
        output_dir,
        "--training.tracker.run_name",
        Path(output_dir).name,
        "--training.loop.max_train_steps",
        str(int(max_train_steps)),
        "--training.checkpoint.training_state_checkpointing_steps",
        str(int(checkpoint_steps)),
        "--training.distributed.num_gpus",
        str(DEFAULT_NUM_GPUS),
        "--training.distributed.hsdp_replicate_dim",
        str(DEFAULT_HSDP_REPLICATE_DIM),
        "--training.distributed.hsdp_shard_dim",
        str(DEFAULT_HSDP_SHARD_DIM),
        "--training.loop.gradient_accumulation_steps",
        str(int(gradient_accumulation_steps)),
        "--training.data.num_frames",
        str(resolved_num_frames),
        "--training.data.num_latent_t",
        str(resolved_num_latent_t),
        "--training.optimizer.learning_rate",
        str(float(learning_rate)),
        "--method.cold_start_steps",
        str(int(cold_start_steps)),
        "--method.dynamic_step",
        str(int(dynamic_step)),
        "--method.guidance_update_ratio",
        str(int(guidance_update_ratio)),
        "--method.num_video_per_prompt",
        str(int(num_samples_per_prompt)),
        "--method.num_batches_per_epoch",
        str(int(num_batches_per_epoch)),
        "--method.sample_train_batch_size",
        str(int(collection_batch_size)),
        "--method.num_inner_epochs",
        str(int(inner_epochs)),
        "--method.train_batch_size",
        str(int(train_batch_size)),
        "--method.validation.every_steps",
        str(int(validation_every_steps)),
        "--method.validation.num_steps",
        str(int(validation_num_steps)),
        "--method.validation.num_prompts",
        str(int(validation_num_prompts)),
        "--method.validation.batch_size",
        str(int(validation_batch_size)),
    ]
    if resume_from_checkpoint:
        cmd.extend([
            "--training.checkpoint.resume_from_checkpoint",
            resume_from_checkpoint,
        ])

    subprocess.run(["nvidia-smi"], check=True)
    subprocess.run(
        cmd,
        cwd=repo,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )

    runs_vol.commit()
    cache_vol.commit()


@app.local_entrypoint()
def main(
    max_train_steps: int = DEFAULT_MAX_TRAIN_STEPS,
    checkpoint_steps: int = DEFAULT_CHECKPOINT_STEPS,
    output_dir: str | None = None,
    resume_from_checkpoint: str | None = None,
    num_samples_per_prompt: int = DEFAULT_NUM_SAMPLES_PER_PROMPT,
    num_batches_per_epoch: int = DEFAULT_NUM_BATCHES_PER_EPOCH,
    collection_batch_size: int = DEFAULT_COLLECTION_BATCH_SIZE,
    inner_epochs: int = DEFAULT_INNER_EPOCHS,
    train_batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    cold_start_steps: int = DEFAULT_COLD_START_STEPS,
    dynamic_step: int = DEFAULT_DYNAMIC_STEP,
    guidance_update_ratio: int = DEFAULT_GUIDANCE_UPDATE_RATIO,
    sample_num_steps: int = DEFAULT_SAMPLE_NUM_STEPS,
    sample_flow_shift: float = DEFAULT_SAMPLE_FLOW_SHIFT,
    sample_guidance_scale: float = DEFAULT_SAMPLE_GUIDANCE_SCALE,
    validation_every_steps: int = DEFAULT_VALIDATION_EVERY_STEPS,
    validation_num_steps: int = DEFAULT_VALIDATION_NUM_STEPS,
    validation_num_prompts: int = DEFAULT_VALIDATION_NUM_PROMPTS,
    validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE,
    num_frames: int = DEFAULT_NUM_FRAMES,
    num_latent_t: int = DEFAULT_NUM_LATENT_T,
    log_sample_max_videos: int = DEFAULT_LOG_SAMPLE_MAX_VIDEOS,
    preprocess_batch_size: int = DEFAULT_PREPROCESS_BATCH_SIZE,
    preprocess_num_gpus: int = DEFAULT_PREPROCESS_NUM_GPUS,
    dataset: str = DEFAULT_DATASET,
    reward: str = DEFAULT_REWARD,
    max_prompts: str = DEFAULT_MAX_PROMPTS,
    check_rewards: bool = True,
):
    train.spawn(
        max_train_steps=max_train_steps,
        checkpoint_steps=checkpoint_steps,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        num_samples_per_prompt=num_samples_per_prompt,
        num_batches_per_epoch=num_batches_per_epoch,
        collection_batch_size=collection_batch_size,
        inner_epochs=inner_epochs,
        train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        cold_start_steps=cold_start_steps,
        dynamic_step=dynamic_step,
        guidance_update_ratio=guidance_update_ratio,
        sample_num_steps=sample_num_steps,
        sample_flow_shift=sample_flow_shift,
        sample_guidance_scale=sample_guidance_scale,
        validation_every_steps=validation_every_steps,
        validation_num_steps=validation_num_steps,
        validation_num_prompts=validation_num_prompts,
        validation_batch_size=validation_batch_size,
        num_frames=num_frames,
        num_latent_t=num_latent_t,
        log_sample_max_videos=log_sample_max_videos,
        preprocess_batch_size=preprocess_batch_size,
        preprocess_num_gpus=preprocess_num_gpus,
        dataset=dataset,
        reward=reward,
        max_prompts=max_prompts,
        check_rewards=check_rewards,
    )
