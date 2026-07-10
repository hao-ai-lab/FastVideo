# SPDX-License-Identifier: Apache-2.0
"""Pipeline latent parity for Flux2 Klein.

Compares a FastVideo ``Flux2KleinPipeline`` run against Diffusers'
``Flux2KleinPipeline``
for the canonical Klein prompt, seed, 1024x1024 resolution, and four denoising
steps. The comparison uses final denoised latents for determinism and speed.

Activate locally with:

    FLUX2_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-klein-4B \
    pytest tests/local_tests/pipelines/test_flux2_pipeline_parity.py -v -s

The test is skipped in CI unless CUDA and ``FLUX2_MODEL_DIR`` are available.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor
from safetensors.torch import safe_open
from torch.testing import assert_close


DEFAULT_PROMPT = "a photo of a banana on a wooden table, studio lighting"
PROMPT = os.getenv("FLUX2_PROMPT", DEFAULT_PROMPT)
SEED = int(os.getenv("FLUX2_SEED", "0"))
HEIGHT = 1024
WIDTH = 1024
NUM_FRAMES = 1
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0
ATOL = 1e-4
RTOL = 1e-4
TP_ATOL = 2.5
TP_RTOL = 0.2
TP_MAX_MEAN_DIFF = 0.03
TP_MAX_ABS_MEAN_DRIFT = 0.001
FULL_TP_ATOL = 0.075
FULL_TP_RTOL = 0.1
FULL_TP_MAX_MEAN_DIFF = 0.01
FULL_TP_MAX_ABS_MEAN_DRIFT = 0.0015
FULL_INPUT_VARIANTS_ENABLED = os.getenv("FLUX2_FULL_RUN_INPUT_VARIANTS", "0") == "1"
FULL_INPUT_VARIANT_MAX_ABS_MEAN_DRIFT = 0.0025
MODEL_DIR = Path(os.getenv("FLUX2_MODEL_DIR", ""))
FULL_MODEL_DIR = Path(os.getenv("FLUX2_FULL_MODEL_DIR", ""))
FULL_HEIGHT = int(os.getenv("FLUX2_FULL_HEIGHT", "128"))
FULL_WIDTH = int(os.getenv("FLUX2_FULL_WIDTH", "128"))
FULL_NUM_INFERENCE_STEPS = int(os.getenv("FLUX2_FULL_STEPS", "1"))
FULL_GUIDANCE_SCALE = float(os.getenv("FLUX2_FULL_GUIDANCE_SCALE", "4.0"))
FULL_MAX_SEQUENCE_LENGTH = int(os.getenv("FLUX2_FULL_MAX_SEQUENCE_LENGTH", "64"))
FULL_NUM_GPUS = int(os.getenv("FLUX2_FULL_NUM_GPUS", "2"))
FULL_TP_SIZE = int(os.getenv("FLUX2_FULL_TP_SIZE", str(FULL_NUM_GPUS)))
FULL_SP_SIZE = int(
    os.getenv(
        "FLUX2_FULL_SP_SIZE",
        "1" if FULL_NUM_GPUS > 1 else str(FULL_NUM_GPUS),
    )
)
FULL_INPUT_VARIANTS = (
    (
        "changed_prompt_seed0",
        "a watercolor painting of a red bicycle leaning against a stone wall",
        0,
    ),
    ("default_prompt_seed123", DEFAULT_PROMPT, 123),
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 Klein pipeline parity requires CUDA",
)


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    tensor_f32 = tensor.detach().float()
    print(
        f"[FLUX2 PIPELINE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"mean={tensor_f32.mean().item():.6f} "
        f"abs_mean={tensor_f32.abs().mean().item():.6f} "
        f"min={tensor_f32.min().item():.6f} max={tensor_f32.max().item():.6f}"
    )


def _print_assert_close_means(
    label: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    expected_f32 = expected.detach().float()
    actual_f32 = actual.detach().float()
    print(
        f"[{label}] assert_close means "
        f"expected_mean={expected_f32.mean().item():.6f} "
        f"actual_mean={actual_f32.mean().item():.6f} "
        f"expected_abs_mean={expected_f32.abs().mean().item():.6f} "
        f"actual_abs_mean={actual_f32.abs().mean().item():.6f}"
    )


def _require_tp_size() -> int:
    raw_tp_size = os.getenv("FLUX2_TP_SIZE", "")
    if not raw_tp_size:
        pytest.skip("Set FLUX2_TP_SIZE>1 to activate Flux2 tensor-parallel parity")
    tp_size = int(raw_tp_size)
    if tp_size <= 1:
        pytest.skip("Flux2 tensor-parallel parity requires FLUX2_TP_SIZE>1")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(
            f"Flux2 tensor-parallel parity requires {tp_size} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    return tp_size


def _require_model_dir() -> Path:
    if not MODEL_DIR.exists():
        pytest.skip("Set FLUX2_MODEL_DIR to activate Flux2 Klein pipeline parity")
    return MODEL_DIR


def _require_full_model_dir() -> Path:
    if not FULL_MODEL_DIR.exists():
        pytest.skip("Set FLUX2_FULL_MODEL_DIR to activate full Flux2 pipeline parity")
    return FULL_MODEL_DIR


def _require_full_parallel_config() -> tuple[int, int, int]:
    if torch.cuda.device_count() < FULL_NUM_GPUS:
        pytest.skip(
            f"Full Flux2 pipeline parity requires {FULL_NUM_GPUS} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )
    return FULL_NUM_GPUS, FULL_TP_SIZE, FULL_SP_SIZE


def _compare_tensor(
    label: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    expected_cpu = expected.detach().float().cpu()
    actual_cpu = actual.detach().float().cpu()
    diff = (expected_cpu - actual_cpu).abs()
    print(
        f"[FLUX2 SETUP] {label}: shape={tuple(expected.shape)} "
        f"expected_dtype={expected.dtype} actual_dtype={actual.dtype} "
        f"diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    assert_close(expected, actual, atol=atol, rtol=rtol)


def _print_diff_quantization(label: str, diff: torch.Tensor) -> None:
    diff_flat = diff.detach().float().reshape(-1)
    if diff_flat.numel() == 0:
        return

    top_values, top_indices = torch.topk(
        diff_flat,
        k=min(8, diff_flat.numel()),
    )
    unique_values, counts = torch.unique(
        diff_flat,
        sorted=True,
        return_counts=True,
    )
    nonzero_mask = unique_values > 0
    unique_nonzero = unique_values[nonzero_mask]
    counts_nonzero = counts[nonzero_mask]
    largest_unique = unique_nonzero[-8:] if unique_nonzero.numel() else unique_nonzero
    largest_counts = counts_nonzero[-8:] if counts_nonzero.numel() else counts_nonzero

    q_128 = 1.0 / 128.0
    q_16 = 1.0 / 16.0
    q128_hits = torch.isclose(
        diff_flat / q_128,
        torch.round(diff_flat / q_128),
        atol=1e-6,
        rtol=0.0,
    )
    q16_hits = torch.isclose(
        diff_flat / q_16,
        torch.round(diff_flat / q_16),
        atol=1e-6,
        rtol=0.0,
    )
    print(
        f"[FLUX2 {label}] diff quantization "
        f"nonzero={int((diff_flat > 0).sum().item())}/{diff_flat.numel()} "
        f"multiples_of_1/128={int(q128_hits.sum().item())}/{diff_flat.numel()} "
        f"multiples_of_1/16={int(q16_hits.sum().item())}/{diff_flat.numel()}"
    )
    print(
        f"[FLUX2 {label}] top_abs_diffs="
        f"{[round(v.item(), 6) for v in top_values]} "
        f"flat_indices={[int(i.item()) for i in top_indices]}"
    )
    if unique_nonzero.numel():
        print(
            f"[FLUX2 {label}] largest_unique_abs_diffs="
            f"{[(round(v.item(), 6), int(c.item())) for v, c in zip(largest_unique, largest_counts, strict=True)]}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return cast(dict[str, Any], json.load(f))


def _load_flux2_vae_bn_stats(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    vae_dir = model_dir / "vae"
    keys = ("bn.running_mean", "bn.running_var")
    tensors: dict[str, torch.Tensor] = {}

    single = vae_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        paths = [single]
    else:
        index = vae_dir / "diffusion_pytorch_model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(
                f"Missing Flux2 VAE safetensors checkpoint in {vae_dir}"
            )
        weight_map = _load_json(index)["weight_map"]
        paths = sorted({vae_dir / cast(str, weight_map[key]) for key in keys})

    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            available = set(f.keys())
            for key in keys:
                if key in available:
                    tensors[key] = f.get_tensor(key)

    missing = [key for key in keys if key not in tensors]
    if missing:
        raise KeyError(f"Missing Flux2 VAE BN stats {missing} in {vae_dir}")
    return tensors["bn.running_mean"], tensors["bn.running_var"]


def _unpatchify_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels // 4, 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)


def _normalize_fastvideo_flux2_latents(
    latents: torch.Tensor,
    model_dir: Path,
) -> torch.Tensor:
    if latents.ndim == 5:
        if latents.shape[2] != 1:
            raise AssertionError(
                f"Expected Flux2 image latents with T=1, got {latents.shape}"
            )
        latents = latents[:, :, 0]

    running_mean, running_var = _load_flux2_vae_bn_stats(model_dir)
    if latents.ndim == 4 and latents.shape[1] == running_mean.numel():
        cfg = _load_json(model_dir / "vae" / "config.json")
        eps = float(cfg.get("batch_norm_eps", 1e-5))
        running_mean = running_mean.view(1, -1, 1, 1).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        running_var = running_var.view(1, -1, 1, 1).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = latents * torch.sqrt(torch.clamp(running_var + eps, min=1e-6))
        latents = latents + running_mean
        latents = _unpatchify_flux2_latents(latents)

    return latents


def _pack_fastvideo_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 5:
        if latents.shape[2] != 1:
            raise AssertionError(
                f"Expected Flux2 packed image latents with T=1, got {latents.shape}"
            )
        return latents.permute(0, 2, 3, 4, 1).reshape(
            latents.shape[0],
            latents.shape[3] * latents.shape[4],
            latents.shape[1],
        )
    if latents.ndim == 4:
        return latents.permute(0, 2, 3, 1).reshape(
            latents.shape[0],
            latents.shape[2] * latents.shape[3],
            latents.shape[1],
        )
    if latents.ndim == 3:
        return latents
    raise AssertionError(f"Unexpected Flux2 latent shape: {latents.shape}")


def _pack_diffusers_flux2_image_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim == 3:
        return latents.detach().float().cpu()
    if latents.ndim == 4:
        return latents.permute(0, 2, 3, 1).reshape(
            latents.shape[0],
            latents.shape[2] * latents.shape[3],
            latents.shape[1],
        ).detach().float().cpu()
    raise AssertionError(f"Unexpected Diffusers Flux2 latent shape: {latents.shape}")


def _run_diffusers_flux_pipeline(model_dir: Path) -> tuple[torch.Tensor, list[torch.Tensor]]:
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # FastVideo seeds CPU generators in InputValidationStage for cross-GPU
    # reproducibility, so use the same generator placement on the Diffusers side.
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    trajectory: list[torch.Tensor] = []

    def _capture_step_latents(_pipe, _step: int, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        trajectory.append(latents.detach().float().cpu())
        return {}

    pipe = Flux2KleinPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        with torch.no_grad():
            output = pipe(
                prompt=PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
                output_type="latent",
                return_dict=True,
                callback_on_step_end=_capture_step_latents,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        output = cast(Any, output)
        latents = output.images
        if not torch.is_tensor(latents):
            latents = torch.as_tensor(latents)
        return latents.detach().float().cpu(), trajectory
    finally:
        del pipe
        torch.cuda.empty_cache()


def _run_fastvideo_flux2_pipeline(
    model_dir: Path,
    *,
    num_gpus: int = 1,
    tp_size: int = 1,
    sp_size: int | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from fastvideo import VideoGenerator

    if sp_size is None:
        sp_size = num_gpus
    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    try:
        executor_world_size = getattr(generator.executor, "world_size", None)
        print(
            "[FLUX2 PIPELINE] fastvideo parallel config "
            f"num_gpus={generator.fastvideo_args.num_gpus} "
            f"tp_size={generator.fastvideo_args.tp_size} "
            f"sp_size={generator.fastvideo_args.sp_size} "
            f"executor_world_size={executor_world_size}"
        )
        assert generator.fastvideo_args.num_gpus == num_gpus
        assert generator.fastvideo_args.tp_size == tp_size
        assert generator.fastvideo_args.sp_size == sp_size
        if executor_world_size is not None:
            assert executor_world_size == num_gpus

        result = generator.generate_video(
            prompt=PROMPT,
            output_path="outputs_video/flux2_klein_pipeline_parity",
            save_video=False,
            return_frames=True,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=SEED,
            return_trajectory_latents=True,
        )
        assert isinstance(result, dict)
        result_dict = cast(dict[str, Any], result)
        latents = result_dict["samples"]
        assert torch.is_tensor(latents), "FastVideo did not return latent samples"
        trajectory: list[torch.Tensor] = []
        trajectory_raw = result_dict.get("trajectory")
        if torch.is_tensor(trajectory_raw):
            trajectory = [
                _pack_fastvideo_flux2_latents(trajectory_raw[:, step].detach()).float().cpu()
                for step in range(trajectory_raw.shape[1])
            ]
        else:
            print("[FLUX2 PIPELINE] fastvideo trajectory latents unavailable")
        latents = _normalize_fastvideo_flux2_latents(latents.detach(), model_dir)
        return latents.float().cpu(), trajectory
    finally:
        generator.shutdown()
        torch.cuda.empty_cache()


def _run_diffusers_flux2_full_pipeline(
    model_dir: Path,
    *,
    prompt: str = PROMPT,
    seed: int = SEED,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from diffusers import Flux2Pipeline

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trajectory: list[torch.Tensor] = []

    def _capture_step_latents(_pipe, _step: int, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        trajectory.append(_pack_diffusers_flux2_image_latents(latents))
        return {}

    prompt_pipe = Flux2Pipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    )
    with torch.no_grad():
        prompt_embeds = prompt_pipe._get_mistral_3_small_prompt_embeds(  # noqa: SLF001
            text_encoder=prompt_pipe.text_encoder,
            tokenizer=prompt_pipe.tokenizer,
            prompt=[prompt],
            device=torch.device("cpu"),
            max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
            system_message=prompt_pipe.system_message,
            hidden_states_layers=(10, 20, 30),
        )
    del prompt_pipe
    gc.collect()
    torch.cuda.empty_cache()

    max_memory = {idx: "42GiB" for idx in range(torch.cuda.device_count())}
    pipe = Flux2Pipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        text_encoder=None,
        tokenizer=None,
        device_map="balanced",
        max_memory=max_memory,
    )
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    try:
        with torch.no_grad():
            output = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                height=FULL_HEIGHT,
                width=FULL_WIDTH,
                num_inference_steps=FULL_NUM_INFERENCE_STEPS,
                guidance_scale=FULL_GUIDANCE_SCALE,
                max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
                generator=generator,
                output_type="latent",
                return_dict=True,
                callback_on_step_end=_capture_step_latents,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        output = cast(Any, output)
        latents = output.images
        if not torch.is_tensor(latents):
            latents = torch.as_tensor(latents)
        return _pack_diffusers_flux2_image_latents(latents), trajectory
    finally:
        del pipe
        torch.cuda.empty_cache()


def _run_fastvideo_flux2_full_pipeline(
    model_dir: Path,
    *,
    num_gpus: int,
    tp_size: int,
    sp_size: int,
    prompt: str = PROMPT,
    seed: int = SEED,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2Pipeline",
    )
    try:
        result = generator.generate_video(
            prompt=prompt,
            output_path="outputs_video/flux2_full_pipeline_parity",
            save_video=False,
            return_frames=True,
            height=FULL_HEIGHT,
            width=FULL_WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=FULL_NUM_INFERENCE_STEPS,
            guidance_scale=FULL_GUIDANCE_SCALE,
            max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
            seed=seed,
            return_trajectory_latents=True,
        )
        assert isinstance(result, dict)
        result_dict = cast(dict[str, Any], result)
        trajectory: list[torch.Tensor] = []
        trajectory_raw = result_dict.get("trajectory")
        if torch.is_tensor(trajectory_raw):
            trajectory = [
                _pack_fastvideo_flux2_latents(trajectory_raw[:, step].detach()).float().cpu()
                for step in range(trajectory_raw.shape[1])
            ]
        else:
            raise AssertionError("FastVideo full Flux2 trajectory latents unavailable")

        samples = result_dict["samples"]
        assert torch.is_tensor(samples), "FastVideo did not return latent samples"
        if trajectory:
            return trajectory[-1], trajectory
        return _pack_fastvideo_flux2_latents(samples.detach()).float().cpu(), trajectory
    finally:
        generator.shutdown()
        torch.cuda.empty_cache()


def _assert_flux2_full_pipeline_case(
    model_dir: Path,
    *,
    label: str,
    num_gpus: int,
    tp_size: int,
    sp_size: int,
    prompt: str,
    seed: int,
    max_mean_diff: float | None = FULL_TP_MAX_MEAN_DIFF,
    max_abs_mean_drift: float | None = FULL_TP_MAX_ABS_MEAN_DRIFT,
) -> None:
    print(
        f"[FLUX2 {label}] inputs prompt={prompt!r} seed={seed} "
        f"num_gpus={num_gpus} tp_size={tp_size} sp_size={sp_size}"
    )

    diffusers_latents, diffusers_trajectory = _run_diffusers_flux2_full_pipeline(
        model_dir,
        prompt=prompt,
        seed=seed,
    )
    _log_tensor_stats(f"diffusers_{label.lower()}_latents", diffusers_latents)

    fastvideo_latents, fastvideo_trajectory = _run_fastvideo_flux2_full_pipeline(
        model_dir,
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        prompt=prompt,
        seed=seed,
    )
    _log_tensor_stats(f"fastvideo_{label.lower()}_latents", fastvideo_latents)

    _assert_flux2_pipeline_latent_parity(
        label,
        diffusers_latents,
        fastvideo_latents,
        diffusers_trajectory,
        fastvideo_trajectory,
        atol=FULL_TP_ATOL,
        rtol=FULL_TP_RTOL,
        max_mean_diff=max_mean_diff,
        max_abs_mean_drift=max_abs_mean_drift,
    )


def _assert_flux2_pipeline_latent_parity(
    label: str,
    diffusers_latents: torch.Tensor,
    fastvideo_latents: torch.Tensor,
    diffusers_trajectory: list[torch.Tensor],
    fastvideo_trajectory: list[torch.Tensor],
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
    max_mean_diff: float | None = None,
    max_abs_mean_drift: float | None = None,
) -> None:
    if len(diffusers_trajectory) == len(fastvideo_trajectory):
        for step, (diffusers_step, fastvideo_step) in enumerate(
            zip(diffusers_trajectory, fastvideo_trajectory, strict=True)
        ):
            step_diff = (diffusers_step - fastvideo_step).abs()
            print(
                f"[FLUX2 {label}] trajectory_step={step} "
                f"diff max={step_diff.max().item():.6f} "
                f"mean={step_diff.mean().item():.6f} "
                f"median={step_diff.median().item():.6f}"
            )
            _print_assert_close_means(
                f"FLUX2 {label} trajectory step {step}",
                diffusers_step,
                fastvideo_step,
            )
    else:
        print(
            f"[FLUX2 {label}] trajectory length mismatch "
            f"diffusers={len(diffusers_trajectory)} fastvideo={len(fastvideo_trajectory)}"
        )

    assert diffusers_latents.shape == fastvideo_latents.shape, (
        f"shape mismatch: diffusers={diffusers_latents.shape} "
        f"fastvideo={fastvideo_latents.shape}"
    )

    diff = (diffusers_latents - fastvideo_latents).abs()
    print(
        f"[FLUX2 {label}] diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    _print_diff_quantization(label, diff)
    print(
        f"[FLUX2 {label}] abs-mean drift "
        f"diffusers={diffusers_latents.abs().mean().item():.6f} "
        f"fastvideo={fastvideo_latents.abs().mean().item():.6f}"
    )
    if max_mean_diff is not None:
        assert diff.mean().item() <= max_mean_diff, (
            f"{label} mean diff {diff.mean().item():.6f} exceeds {max_mean_diff}"
        )
    if max_abs_mean_drift is not None:
        abs_mean_drift = abs(
            diffusers_latents.abs().mean().item() -
            fastvideo_latents.abs().mean().item()
        )
        assert abs_mean_drift <= max_abs_mean_drift, (
            f"{label} abs-mean drift {abs_mean_drift:.6f} exceeds "
            f"{max_abs_mean_drift}"
        )

    _print_assert_close_means(f"FLUX2 {label}", diffusers_latents, fastvideo_latents)
    assert_close(diffusers_latents, fastvideo_latents, atol=atol, rtol=rtol)


def test_flux2_klein_pipeline_latent_parity() -> None:
    model_dir = _require_model_dir()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    diffusers_latents, diffusers_trajectory = _run_diffusers_flux_pipeline(model_dir)
    _log_tensor_stats("diffusers_latents", diffusers_latents)

    fastvideo_latents, fastvideo_trajectory = _run_fastvideo_flux2_pipeline(model_dir)
    _log_tensor_stats("fastvideo_latents", fastvideo_latents)

    _assert_flux2_pipeline_latent_parity(
        "PIPELINE",
        diffusers_latents,
        fastvideo_latents,
        diffusers_trajectory,
        fastvideo_trajectory,
    )


def test_flux2_klein_pipeline_tensor_parallel_latent_parity() -> None:
    model_dir = _require_model_dir()
    tp_size = _require_tp_size()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    diffusers_latents, diffusers_trajectory = _run_diffusers_flux_pipeline(model_dir)
    _log_tensor_stats("diffusers_tp_reference_latents", diffusers_latents)

    fastvideo_latents, fastvideo_trajectory = _run_fastvideo_flux2_pipeline(
        model_dir,
        num_gpus=tp_size,
        tp_size=tp_size,
        sp_size=1,
    )
    _log_tensor_stats(f"fastvideo_tp{tp_size}_latents", fastvideo_latents)

    _assert_flux2_pipeline_latent_parity(
        f"PIPELINE TP{tp_size}",
        diffusers_latents,
        fastvideo_latents,
        diffusers_trajectory,
        fastvideo_trajectory,
        atol=TP_ATOL,
        rtol=TP_RTOL,
        max_mean_diff=TP_MAX_MEAN_DIFF,
        max_abs_mean_drift=TP_MAX_ABS_MEAN_DRIFT,
    )


def test_flux2_full_pipeline_setup_matches_diffusers() -> None:
    """Compare full Flux2 pre-denoising setup against Diffusers.

    This isolates the power-of-two-looking full pipeline drift from setup
    concerns before loading the FastVideo full transformer: text formatting,
    packed latent layout, position ids, scheduler timesteps, guidance tensor,
    and scheduler shape handling must all match here.
    """
    model_dir = _require_full_model_dir()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Pipeline
    from diffusers.pipelines.flux2.pipeline_flux2 import (
        compute_empirical_mu as diffusers_compute_empirical_mu,
        retrieve_timesteps,
    )
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from fastvideo.configs.pipelines.flux_2 import Flux2PipelineConfig
    from fastvideo.pipelines.basic.flux_2.flux_2_latent_preparation import (
        Flux2LatentPreparationStage,
    )
    from fastvideo.pipelines.basic.flux_2.flux_2_text_encoding import (
        FLUX2_SYSTEM_MESSAGE,
        Flux2TextEncodingStage,
        _prepare_flux2_text_ids,
    )
    from fastvideo.pipelines.basic.flux_2.flux_2_timestep_preparation import (
        Flux2TimestepPreparationStage,
        compute_empirical_mu as fastvideo_compute_empirical_mu,
    )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages.input_validation import InputValidationStage

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.cuda.set_device(device)

    processor = AutoProcessor.from_pretrained(
        str(model_dir / "tokenizer"),
        local_files_only=True,
    )
    text_encoder = AutoModelForImageTextToText.from_pretrained(
        str(model_dir / "text_encoder"),
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()

    cfg = Flux2PipelineConfig()
    text_stage = Flux2TextEncodingStage(
        text_encoders=[text_encoder],
        tokenizers=[processor],
    )
    fv_text_batch = ForwardBatch(
        data_type="image",
        prompt=PROMPT,
        height=FULL_HEIGHT,
        width=FULL_WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=FULL_NUM_INFERENCE_STEPS,
        guidance_scale=FULL_GUIDANCE_SCALE,
        max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
        seed=SEED,
    )
    fv_text_batch = text_stage.forward(
        fv_text_batch,
        cast(Any, SimpleNamespace(pipeline_config=cfg)),
    )
    fv_prompt_embeds = fv_text_batch.prompt_embeds[0]
    fv_text_ids = fv_text_batch.extra["flux2_txt_ids"]

    with torch.no_grad():
        diffusers_prompt_embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(  # noqa: SLF001
            text_encoder=text_encoder,
            tokenizer=processor,
            prompt=[PROMPT],
            device=torch.device("cpu"),
            max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
            system_message=FLUX2_SYSTEM_MESSAGE,
            hidden_states_layers=(10, 20, 30),
        )
    diffusers_text_ids = Flux2Pipeline._prepare_text_ids(diffusers_prompt_embeds)  # noqa: SLF001

    _compare_tensor(
        "prompt_embeds",
        diffusers_prompt_embeds,
        fv_prompt_embeds,
    )
    assert torch.equal(diffusers_text_ids, fv_text_ids.cpu())
    assert torch.equal(_prepare_flux2_text_ids(fv_prompt_embeds).cpu(), diffusers_text_ids)
    print(
        f"[FLUX2 SETUP] text_ids exact shape={tuple(diffusers_text_ids.shape)} "
        f"last={diffusers_text_ids[0, -1].tolist()}"
    )

    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    batch_size = diffusers_prompt_embeds.shape[0]
    num_latents_channels = 32
    prepared_height = 2 * (int(FULL_HEIGHT) // (8 * 2))
    prepared_width = 2 * (int(FULL_WIDTH) // (8 * 2))
    raw_shape = (
        batch_size,
        num_latents_channels * 4,
        prepared_height // 2,
        prepared_width // 2,
    )
    diffusers_generator = torch.Generator(device="cpu").manual_seed(SEED)
    diffusers_raw_latents = randn_tensor(
        raw_shape,
        generator=diffusers_generator,
        device=device,
        dtype=dtype,
    )
    diffusers_latent_ids = Flux2Pipeline._prepare_latent_ids(diffusers_raw_latents).to(device)  # noqa: SLF001
    diffusers_packed_latents = Flux2Pipeline._pack_latents(diffusers_raw_latents)  # noqa: SLF001

    class _DummyTransformer:
        num_channels_latents = num_latents_channels * 4
        hidden_size = diffusers_prompt_embeds.shape[-1]

        def parameters(self):
            yield torch.empty((), device=device, dtype=dtype)

    fv_scheduler_for_latents = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir / "scheduler"),
        local_files_only=True,
    )
    fv_latent_batch = ForwardBatch(
        data_type="image",
        prompt=PROMPT,
        prompt_embeds=[diffusers_prompt_embeds.to(device=device, dtype=dtype)],
        height=FULL_HEIGHT,
        width=FULL_WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=FULL_NUM_INFERENCE_STEPS,
        guidance_scale=FULL_GUIDANCE_SCALE,
        max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
        seed=SEED,
    )
    fv_latent_batch = InputValidationStage().forward(
        fv_latent_batch,
        cast(Any, SimpleNamespace(pipeline_config=cfg)),
    )
    fv_latent_batch = Flux2LatentPreparationStage(
        scheduler=fv_scheduler_for_latents,
        transformer=_DummyTransformer(),
    ).forward(
        fv_latent_batch,
        cast(Any, SimpleNamespace(pipeline_config=cfg)),
    )
    fv_raw_latents = fv_latent_batch.latents
    assert fv_raw_latents is not None
    fv_packed_latents = _pack_fastvideo_flux2_latents(fv_raw_latents)
    fv_latent_ids = fv_latent_batch.extra["flux2_img_ids"]

    _compare_tensor(
        "raw_latents",
        diffusers_raw_latents,
        fv_raw_latents[:, :, 0],
    )
    _compare_tensor(
        "packed_latents",
        diffusers_packed_latents,
        fv_packed_latents,
    )
    assert torch.equal(diffusers_latent_ids, fv_latent_ids)
    print(
        f"[FLUX2 SETUP] img_ids exact shape={tuple(diffusers_latent_ids.shape)} "
        f"last={diffusers_latent_ids[0, -1].tolist()}"
    )

    diffusers_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir / "scheduler"),
        local_files_only=True,
    )
    sigmas = np.linspace(
        1.0,
        1.0 / FULL_NUM_INFERENCE_STEPS,
        FULL_NUM_INFERENCE_STEPS,
    )
    if getattr(diffusers_scheduler.config, "use_flow_sigmas", False):
        sigmas = None
    diffusers_mu = diffusers_compute_empirical_mu(
        image_seq_len=diffusers_packed_latents.shape[1],
        num_steps=FULL_NUM_INFERENCE_STEPS,
    )
    diffusers_timesteps, _ = retrieve_timesteps(
        diffusers_scheduler,
        FULL_NUM_INFERENCE_STEPS,
        device,
        sigmas=sigmas,
        mu=diffusers_mu,
    )

    fv_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(model_dir / "scheduler"),
        local_files_only=True,
    )
    fv_time_batch = ForwardBatch(
        data_type="image",
        height=FULL_HEIGHT,
        width=FULL_WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=FULL_NUM_INFERENCE_STEPS,
        n_tokens=diffusers_packed_latents.shape[1],
    )
    fv_time_batch = Flux2TimestepPreparationStage(scheduler=fv_scheduler).forward(
        fv_time_batch,
        cast(Any, SimpleNamespace(pipeline_config=cfg)),
    )
    assert fv_time_batch.timesteps is not None
    _compare_tensor(
        "timesteps",
        diffusers_timesteps,
        fv_time_batch.timesteps,
    )
    assert diffusers_mu == pytest.approx(
        fastvideo_compute_empirical_mu(
            diffusers_packed_latents.shape[1],
            FULL_NUM_INFERENCE_STEPS,
        )
    )

    diffusers_guidance = torch.full(
        [1],
        FULL_GUIDANCE_SCALE,
        device=device,
        dtype=torch.float32,
    ).expand(batch_size)
    fv_guidance = torch.tensor(
        [FULL_GUIDANCE_SCALE] * batch_size,
        dtype=torch.float32,
        device=device,
    ).to(dtype)
    _compare_tensor(
        "guidance_after_transformer_cast",
        diffusers_guidance.to(dtype) * 1000,
        fv_guidance.to(dtype) * 1000,
    )

    noise_generator = torch.Generator(device="cpu").manual_seed(SEED + 123)
    diffusers_noise = randn_tensor(
        diffusers_packed_latents.shape,
        generator=noise_generator,
        device=device,
        dtype=dtype,
    )
    fv_noise = diffusers_noise.reshape(
        batch_size,
        1,
        prepared_height // 2,
        prepared_width // 2,
        num_latents_channels * 4,
    ).permute(0, 4, 1, 2, 3).contiguous()

    diffusers_step = diffusers_scheduler.step(
        diffusers_noise,
        diffusers_timesteps[0],
        diffusers_packed_latents,
        return_dict=False,
    )[0]
    fv_step = fv_scheduler.step(
        fv_noise,
        fv_time_batch.timesteps[0],
        fv_raw_latents,
        return_dict=False,
    )[0]
    _compare_tensor(
        "scheduler_step_packed_vs_5d",
        diffusers_step,
        _pack_fastvideo_flux2_latents(fv_step),
    )

    del processor
    gc.collect()
    torch.cuda.empty_cache()


def test_flux2_full_pipeline_latent_parity() -> None:
    model_dir = _require_full_model_dir()
    num_gpus, tp_size, sp_size = _require_full_parallel_config()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    _assert_flux2_full_pipeline_case(
        model_dir,
        label="FULL PIPELINE",
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        prompt=PROMPT,
        seed=SEED,
    )


@pytest.mark.parametrize(("case_name", "prompt", "seed"), FULL_INPUT_VARIANTS)
def test_flux2_full_pipeline_latent_parity_input_variants(
    case_name: str,
    prompt: str,
    seed: int,
) -> None:
    if not FULL_INPUT_VARIANTS_ENABLED:
        pytest.skip(
            "Set FLUX2_FULL_RUN_INPUT_VARIANTS=1 to run the full Flux2 prompt/seed drift diagnostic"
        )
    model_dir = _require_full_model_dir()
    num_gpus, tp_size, sp_size = _require_full_parallel_config()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    _assert_flux2_full_pipeline_case(
        model_dir,
        label=f"FULL PIPELINE {case_name}",
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_size=sp_size,
        prompt=prompt,
        seed=seed,
        max_abs_mean_drift=FULL_INPUT_VARIANT_MAX_ABS_MEAN_DRIFT,
    )
