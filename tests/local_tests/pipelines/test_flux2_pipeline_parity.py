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

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import safe_open
from torch.testing import assert_close


PROMPT = "a photo of a banana on a wooden table, studio lighting"
SEED = 0
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
MODEL_DIR = Path(os.getenv("FLUX2_MODEL_DIR", ""))


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
