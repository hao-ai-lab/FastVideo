# SPDX-License-Identifier: Apache-2.0
"""
Maps studio workload types onto the modular trainer (fastvideo/train).

Each training job is described by a YAML run config for
``fastvideo.train.entrypoint.train`` (methods x models x callbacks).
``build_training_config`` assembles that config as a plain dict; the job
runner serialises it next to the job's output directory and launches the
entrypoint via torchrun.
"""

from __future__ import annotations

import os
from typing import Any

WAN_MODEL = "fastvideo.train.models.wan.WanModel"
WAN_CAUSAL_MODEL = "fastvideo.train.models.wan.WanCausalModel"

FINETUNE_METHOD = "fastvideo.train.methods.fine_tuning.finetune.FineTuneMethod"
DMD2_METHOD = "fastvideo.train.methods.distribution_matching.dmd2.DMD2Method"
SELF_FORCING_METHOD = "fastvideo.train.methods.distribution_matching.self_forcing.SelfForcingMethod"
KD_CAUSAL_METHOD = "fastvideo.train.methods.knowledge_distillation.kd.KDCausalMethod"

WAN_PIPELINE = "fastvideo.pipelines.basic.wan.wan_pipeline.WanPipeline"
WAN_DMD_PIPELINE = "fastvideo.pipelines.basic.wan.wan_dmd_pipeline.WanDMDPipeline"
WAN_CAUSAL_DMD_PIPELINE = "fastvideo.pipelines.basic.wan.wan_causal_dmd_pipeline.WanCausalDMDPipeline"

# Only T2V workflows are supported for finetuning and distillation.
SUPPORTED_WORKLOADS: frozenset[str] = frozenset({
    # Finetuning
    "full_t2v",
    "vsa_t2v",
    "ode_init",
    # Distillation
    "dmd_t2v",
    "self_forcing_t2v",
    # LoRA
    "lora_t2v",
})

DISTILL_WORKLOADS = ("dmd_t2v", "self_forcing_t2v")


def is_ltx2_model(model_path: str) -> bool:
    """True if the model path identifies an LTX-2 model (unsupported by fastvideo/train)."""
    lower = (model_path or "").lower()
    return "ltx2" in lower or "ltx-2" in lower


def get_training_env() -> dict[str, str]:
    """Environment variables for the training subprocess.

    The train entrypoint selects the attention backend itself from the
    config's ``training.vsa_sparsity``, so no backend env var is needed here.
    """
    return {
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "offline",
    }


def _parse_denoising_steps(raw: str | None) -> list[int]:
    """Parse the UI's comma-separated denoising steps into the YAML list form."""
    text = (raw or "").strip() or "1000,757,522"
    try:
        steps = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid DMD denoising steps '{raw}': expected comma-separated integers") from exc
    if not steps:
        raise ValueError(f"Invalid DMD denoising steps '{raw}': expected at least one integer")
    return steps


def _models_section(job: dict[str, Any], workload_type: str) -> dict[str, Any]:
    model_id = job.get("model_id", "")
    student_target = (WAN_CAUSAL_MODEL if workload_type in ("self_forcing_t2v", "ode_init") else WAN_MODEL)
    models: dict[str, Any] = {
        "student": {
            "_target_": student_target,
            "init_from": model_id,
            "trainable": True,
        },
    }

    if workload_type == "lora_t2v":
        lora_rank = int(job.get("lora_rank", 32) or 32)
        models["student"]["lora"] = {
            "enable": True,
            "rank": lora_rank,
            "alpha": 2 * lora_rank,
            "target_modules": ["to_q", "to_k", "to_v", "to_out"],
        }

    if workload_type in DISTILL_WORKLOADS:
        real_score = job.get("real_score_model_path", "") or model_id
        fake_score = job.get("fake_score_model_path", "") or model_id
        models["teacher"] = {
            "_target_": WAN_MODEL,
            "init_from": real_score,
            "trainable": False,
            "disable_custom_init_weights": True,
        }
        models["critic"] = {
            "_target_": WAN_MODEL,
            "init_from": fake_score,
            "trainable": True,
            "disable_custom_init_weights": True,
        }
    elif workload_type == "ode_init":
        models["teacher"] = {
            "_target_": WAN_MODEL,
            "init_from": model_id,
            "trainable": False,
            "disable_custom_init_weights": True,
        }

    return models


def _method_section(job: dict[str, Any], workload_type: str, output_dir: str) -> dict[str, Any]:
    if workload_type in ("full_t2v", "vsa_t2v", "lora_t2v"):
        return {"_target_": FINETUNE_METHOD}

    # Knobs equal to the method's own defaults are omitted throughout this
    # module so the generated YAML records only deliberate studio choices.
    if workload_type == "ode_init":
        # KD against the teacher's ODE trajectories, cached under the job dir
        # (the first stage of the ode_init_self_forcing_wan_causal scenario).
        return {
            "_target_": KD_CAUSAL_METHOD,
            "teacher_path_cache": os.path.join(output_dir, "kd_cache"),
            "t_list": [995, 937, 833, 625, 0],
            "teacher_guidance_scale": 3.5,
        }

    # DMD / self-forcing distillation. The studio exposes a single learning
    # rate, used for both the generator and the critic.
    learning_rate = job.get("learning_rate", 5e-5)
    method: dict[str, Any] = {
        "_target_": (SELF_FORCING_METHOD if workload_type == "self_forcing_t2v" else DMD2_METHOD),
        "rollout_mode": "simulate",
        "generator_update_interval": int(job.get("generator_update_interval", 5) or 5),
        "real_score_guidance_scale": float(job.get("real_score_guidance_scale", 3.5) or 3.5),
        "dmd_denoising_steps": _parse_denoising_steps(job.get("dmd_denoising_steps")),
        "fake_score_learning_rate": learning_rate,
        "fake_score_betas": [0.0, 0.999],
        "fake_score_lr_scheduler": "constant",
    }
    if workload_type == "self_forcing_t2v":
        # Non-default rollout knobs from
        # examples/train/configs/distribution_matching/wan/self_forcing_causal_t2v.yaml.
        method.update({
            "warp_denoising_step": True,
            "same_step_across_blocks": True,
        })
    return method


def _training_section(job: dict[str, Any], workload_type: str, output_dir: str) -> dict[str, Any]:
    num_gpus = int(job.get("num_gpus", 1) or 1)
    distill = workload_type in DISTILL_WORKLOADS

    vsa_sparsity = 0.0
    if workload_type == "vsa_t2v":
        vsa_sparsity = 0.8
    elif distill and job.get("dmd_use_vsa"):
        vsa_sparsity = float(job.get("dmd_vsa_sparsity", 0.8) or 0.8)

    return {
        "distributed": {
            "num_gpus": num_gpus,
            "hsdp_shard_dim": num_gpus,
        },
        "data": {
            "data_path": job.get("data_path", ""),
            "dataloader_num_workers": 1,
            "train_batch_size": int(job.get("train_batch_size", 1) or 1),
            "training_cfg_rate": 0.0 if (distill or workload_type == "ode_init") else 0.1,
            "seed": 1000,
            "num_latent_t": int(job.get("num_latent_t", 20) or 20),
            "num_height": int(job.get("num_height") or job.get("height") or 480),
            "num_width": int(job.get("num_width") or job.get("width") or 832),
            "num_frames": int(job.get("num_frames", 77) or 77),
        },
        "optimizer": {
            "learning_rate": job.get("learning_rate", 5e-5),
            # Finetunes keep the schema's (0.9, 0.999) default.
            **({
                "betas": [0.0, 0.999]
            } if distill else {}),
            "weight_decay": 1e-4,
        },
        "loop": {
            "max_train_steps": int(job.get("max_train_steps", 1000) or 1000),
            "gradient_accumulation_steps": 8,
        },
        "checkpoint": {
            "output_dir": output_dir,
            "training_state_checkpointing_steps": 500,
            "checkpoints_total_limit": 3,
        },
        "tracker": {
            "project_name": "fastvideo_ui_training",
        },
        # The trainer reads sparsity from training.vsa.sparsity; a flat
        # training.vsa_sparsity key is silently dropped by the config loader.
        "vsa": {
            "sparsity": vsa_sparsity,
        },
        "model": {
            "enable_gradient_checkpointing_type": "full",
        },
    }


def _callbacks_section(job: dict[str, Any], workload_type: str) -> dict[str, Any]:
    callbacks: dict[str, Any] = {
        "grad_clip": {
            "_target_": "fastvideo.train.callbacks.grad_clip.GradNormClipCallback",
            "max_grad_norm": 1.0,
        },
    }

    if workload_type in DISTILL_WORKLOADS:
        callbacks["ema"] = {
            "_target_": "fastvideo.train.callbacks.ema.EMACallback",
            "decay": 0.99 if workload_type == "self_forcing_t2v" else 0.98,
            "start_iter": 200 if workload_type == "self_forcing_t2v" else 0,
        }

    # The KD/ODE-init method has no sampling-based validation pipeline; skip.
    validation_dataset_file = job.get("validation_dataset_file", "")
    if validation_dataset_file and workload_type != "ode_init":
        validation: dict[str, Any] = {
            "_target_": "fastvideo.train.callbacks.validation.ValidationCallback",
            "dataset_file": validation_dataset_file,
            "every_steps": 200,
            "guidance_scale": 3.0,
        }
        if workload_type in DISTILL_WORKLOADS:
            steps = _parse_denoising_steps(job.get("dmd_denoising_steps"))
            validation["sampling_steps"] = [len(steps)]
            validation["sampling_timesteps"] = steps
            if workload_type == "self_forcing_t2v":
                validation["pipeline_target"] = WAN_CAUSAL_DMD_PIPELINE
                validation["num_frames"] = int(job.get("num_frames", 77) or 77)
            else:
                validation["pipeline_target"] = WAN_DMD_PIPELINE
        else:
            validation["pipeline_target"] = WAN_PIPELINE
            validation["sampling_steps"] = [50]
        callbacks["validation"] = validation

    return callbacks


def _pipeline_section(workload_type: str) -> dict[str, Any]:
    if workload_type == "dmd_t2v":
        return {"flow_shift": 8}
    if workload_type == "self_forcing_t2v":
        return {
            "flow_shift": 5,
            "dit_config": {
                "local_attn_size": -1,
                "sink_size": 0
            },
        }
    if workload_type == "ode_init":
        return {"flow_shift": 5}
    return {"flow_shift": 3}


def build_training_config(job: dict[str, Any], output_dir: str) -> dict[str, Any]:
    """Build the fastvideo/train YAML run config (as a dict) for a studio job."""
    workload_type = job.get("workload_type", "full_t2v")
    if workload_type not in SUPPORTED_WORKLOADS:
        raise ValueError(f"Unknown workload type: {workload_type}. "
                         f"Supported: {sorted(SUPPORTED_WORKLOADS)}")

    model_id = job.get("model_id", "")
    if is_ltx2_model(model_id):
        raise ValueError("LTX-2 training is not supported by the modular trainer "
                         "(fastvideo/train has no LTX-2 model plugin). "
                         "Choose a Wan-family model.")

    return {
        "models": _models_section(job, workload_type),
        "method": _method_section(job, workload_type, output_dir),
        "training": _training_section(job, workload_type, output_dir),
        "callbacks": _callbacks_section(job, workload_type),
        "pipeline": _pipeline_section(workload_type),
    }
