# SPDX-License-Identifier: Apache-2.0
"""
Maps workload types to FastVideo training modules and builds CLI arguments.
"""

from __future__ import annotations

from typing import Any

# (script_path for torchrun, relative to repo root; workload_type; use_vsa; is_lora)
# Only T2V workflows are supported for finetuning and distillation.
# full_t2v and lora_t2v route to LTX2 pipeline when model is LTX-2 (see get_training_module_info).
WORKLOAD_TO_MODULE: dict[str, tuple[str, str, bool, bool]] = {
    # Finetuning
    "full_t2v": ("fastvideo/training/wan_training_pipeline.py", "t2v", False, False),
    "vsa_t2v": ("fastvideo/training/wan_training_pipeline.py", "t2v", True, False),
    "ode_init": ("fastvideo/training/ode_causal_pipeline.py", "t2v", False, False),
    # Distillation
    "dmd_t2v": ("fastvideo/training/wan_distillation_pipeline.py", "t2v", False, False),
    "self_forcing_t2v": ("fastvideo/training/wan_self_forcing_distillation_pipeline.py", "t2v", False, False),
    # LoRA
    "lora_t2v": ("fastvideo/training/wan_training_pipeline.py", "t2v", False, True),
}

LTX2_TRAINING_MODULE: tuple[str, str, bool, bool] = (
    "fastvideo/training/ltx2_training_pipeline.py",
    "t2v",
    False,
    False,
)


def is_ltx2_model(model_path: str) -> bool:
    """True if the model path identifies an LTX-2 model (for training pipeline routing)."""
    lower = (model_path or "").lower()
    return "ltx2" in lower or "ltx-2" in lower


def get_training_module_info(workload_type: str, model_id: str) -> tuple[str, str, bool, bool] | None:
    """Resolve (script_path, workload_type, use_vsa, is_lora). Routes full_t2v and lora_t2v to LTX2 pipeline when model is LTX-2."""
    base = WORKLOAD_TO_MODULE.get(workload_type)
    if base is None:
        return None
    if workload_type in ("full_t2v", "lora_t2v") and is_ltx2_model(model_id):
        return LTX2_TRAINING_MODULE
    return base


def get_training_env(use_vsa: bool) -> dict[str, str]:
    """Environment variables for training subprocess."""
    env: dict[str, str] = {
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "offline",
    }
    if use_vsa:
        env["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    return env


def build_training_args(job: dict[str, Any], output_dir: str) -> list[str]:
    """Build CLI arguments for the training subprocess."""
    model_id = job.get("model_id", "")
    data_path = job.get("data_path", "")
    workload_type = job.get("workload_type", "full_t2v")
    num_gpus = job.get("num_gpus", 1)
    max_train_steps = job.get("max_train_steps", 1000)
    train_batch_size = job.get("train_batch_size", 1)
    learning_rate = job.get("learning_rate", 5e-5)
    num_latent_t = job.get("num_latent_t", 20)
    num_height = job.get("num_height") or job.get("height", 480)
    num_width = job.get("num_width") or job.get("width", 832)
    num_frames = job.get("num_frames", 77)
    validation_dataset_file = job.get("validation_dataset_file", "")
    lora_rank = job.get("lora_rank", 32)

    module_info = WORKLOAD_TO_MODULE.get(workload_type)
    if not module_info:
        raise ValueError(f"Unknown workload type: {workload_type}")

    _module_path, pipeline_workload, use_vsa, is_lora = module_info

    args = [
        "--model-path",
        model_id,
        "--pretrained-model-name-or-path",
        model_id,
        "--data-path",
        data_path,
        "--output-dir",
        output_dir,
        "--inference-mode",
        "False",
        "--max-train-steps",
        str(max_train_steps),
        "--train-batch-size",
        str(train_batch_size),
        "--train-sp-batch-size",
        str(train_batch_size),
        "--learning-rate",
        str(learning_rate),
        "--num-gpus",
        str(num_gpus),
        "--num-latent-t",
        str(num_latent_t),
        "--num-height",
        str(num_height),
        "--num-width",
        str(num_width),
        "--num-frames",
        str(num_frames),
        "--mixed-precision",
        "bf16",
        "--gradient-accumulation-steps",
        "8",
        "--enable-gradient-checkpointing-type",
        "full",
        "--weight-decay",
        "1e-4",
        "--max-grad-norm",
        "1.0",
        "--checkpoints-total-limit",
        "3",
        "--training-cfg-rate",
        "0.1",
        "--multi-phased-distill-schedule",
        "4000-1",
        "--not-apply-cfg-solver",
        "--dit-precision",
        "fp32",
        "--num-euler-timesteps",
        "50",
        "--ema-start-step",
        "0",
        "--weight-only-checkpointing-steps",
        "500",
        "--training-state-checkpointing-steps",
        "500",
        "--tracker-project-name",
        "fastvideo_ui_training",
        "--workload-type",
        pipeline_workload,
        "--dataloader-num-workers",
        "1",
    ]

    if validation_dataset_file:
        args.extend([
            "--log-validation",
            "--validation-dataset-file",
            validation_dataset_file,
            "--validation-steps",
            "200",
            "--validation-sampling-steps",
            "50",
            "--validation-guidance-scale",
            "3.0",
        ])

    if is_lora:
        args.extend(["--lora-training", "True", "--lora-rank", str(lora_rank)])

    # Distillation-specific
    if workload_type.startswith("dmd_") or workload_type.startswith("self_forcing_"):
        real_score = job.get("real_score_model_path", "") or model_id
        fake_score = job.get("fake_score_model_path", "") or model_id
        args.extend([
            "--real-score-model-path",
            real_score,
            "--fake-score-model-path",
            fake_score,
            "--training-cfg-rate",
            "0.0",
            "--dmd-denoising-steps",
            job.get("dmd_denoising_steps", "1000,757,522") or "1000,757,522",
            "--min-timestep-ratio",
            str(job.get("min_timestep_ratio", 0.02)),
            "--max-timestep-ratio",
            str(job.get("max_timestep_ratio", 0.98)),
            "--generator-update-interval",
            str(job.get("generator_update_interval", 5)),
            "--real-score-guidance-scale",
            str(job.get("real_score_guidance_scale", 3.5)),
        ])
        if job.get("dmd_use_vsa"):
            args.extend([
                "--VSA-sparsity",
                str(job.get("dmd_vsa_sparsity", 0.8)),
            ])

    # ODE init specific
    if workload_type == "ode_init":
        args.extend([
            "--warp-denoising-step",
            "--multi-phased-distill-schedule",
            "4000-1",
        ])

    if use_vsa:
        args.extend(["--VSA-sparsity", "0.8"])

    # LTX-2 specific
    if is_ltx2_model(model_id):
        p = job.get("ltx2_first_frame_conditioning_p")
        if p is not None:
            args.extend(["--ltx2-first-frame-conditioning-p", str(float(p))])

    # HSDP / parallelism
    args.extend([
        "--sp-size",
        str(min(num_gpus, 8)),
        "--tp-size",
        "1",
        "--hsdp-replicate-dim",
        "1",
        "--hsdp-shard-dim",
        str(num_gpus),
    ])

    return args
