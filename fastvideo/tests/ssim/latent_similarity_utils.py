# SPDX-License-Identifier: Apache-2.0
"""Latent-space regression helpers for numerically fragile SSIM tests.

Motivation
----------
Pixel-space SSIM is a poor regression signal for distilled / few-step
models (e.g. LTX-2 distilled): a single mis-rounded bf16 accumulator in
the VAE decoder can drive mean SSIM from ~0.95 to ~0.50 without any real
quality regression. diffusers works around this by comparing small
signature slices of the pre-VAE latent via cosine distance
(see ``diffusers/tests/pipelines/test_ltx_pipeline.py``).

This module ports the same idea to FastVideo so that CI can reliably
detect actual drift without relying on the pixel-space cliff-edge.

Design
------
* Inference is run with ``output_type='latent'`` so the VAE is skipped
  inside ``DecodingStage``; numerically this is where the bulk of
  run-to-run variance originates.
* The reference artefact is a ``.pt`` bundle (tensor + metadata) committed
  to the same HF dataset as the mp4 references, selected by
  ``<GPU>_reference_videos/<model_id>/<backend>/<prompt>.pt``.
* Two assertions are performed:
    1. A small signature slice (``latent[0, :, 0, :3, :3]``) is compared
       via cosine distance with a loose tolerance. This is the primary
       pass/fail gate.
    2. The full latent is compared via cosine distance with a slightly
       tighter tolerance, guarding against shape-correct but globally
       drifted outputs.
* Tolerances default to 5e-3 (slice) and 1e-2 (full). These mirror the
  order of magnitude diffusers uses (``1e-3``), relaxed to absorb
  cross-GPU-arch bf16 differences on the rented CI pool (A40/L40S/H100).

The helper intentionally reuses ``build_init_kwargs`` /
``build_generation_kwargs`` from :mod:`inference_similarity_utils` so
model params (vae tiling, sp_size, flow shift, …) flow through a single
source of truth.
"""

from __future__ import annotations

import os
from logging import Logger
from typing import Any

import torch
from torch.nn.functional import cosine_similarity

from fastvideo import VideoGenerator
from fastvideo.tests.ssim.inference_similarity_utils import (
    attention_backend,
    build_generation_kwargs,
    build_init_kwargs,
    shutdown_executor,
)
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    select_ssim_params,
)

LATENT_REFERENCE_EXTENSION = ".pt"
LATENT_REFERENCE_FORMAT_VERSION = 1

# ``latent[0, :, 0, :3, :3]`` — first sample, all channels, first latent
# frame, top-left 3x3 spatial patch. Matches the "corner patch" pattern
# used by diffusers but keeps the channel axis so distilled checkpoints
# (C=128 for LTX-2) still contribute rich signal.
DEFAULT_SLICE_SPEC: dict[str, Any] = {
    "kind": "corner_3x3_first_frame",
    "version": 1,
}


def _extract_expected_slice(
    latent: torch.Tensor,
    spec: dict[str, Any],
) -> torch.Tensor:
    """Return a 1-D signature slice extracted from ``latent``.

    ``latent`` must be 5-D ``[B, C, T, H, W]``. The result is always
    fp32 and detached so it can be persisted via ``torch.save`` or fed
    into cosine-distance math without further casts.
    """
    if latent.dim() != 5:
        raise ValueError(
            f"Expected 5-D latent [B,C,T,H,W]; got shape {tuple(latent.shape)}"
        )
    kind = spec.get("kind", "corner_3x3_first_frame")
    if kind == "corner_3x3_first_frame":
        _, _, t, h, w = latent.shape
        if t < 1 or h < 3 or w < 3:
            raise ValueError(
                "corner_3x3_first_frame requires T>=1, H>=3, W>=3; got "
                f"shape {tuple(latent.shape)}")
        return (latent[0, :, 0, :3, :3]
                .detach()
                .to(torch.float32)
                .reshape(-1)
                .contiguous())
    raise ValueError(f"Unknown slice kind: {kind!r}")


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return ``1 - cos(a, b)`` as a Python float, operating in fp32."""
    a32 = a.detach().to(torch.float32).reshape(-1)
    b32 = b.detach().to(torch.float32).reshape(-1)
    if a32.shape != b32.shape:
        raise ValueError(
            f"Cosine shape mismatch: {tuple(a32.shape)} vs {tuple(b32.shape)}"
        )
    sim = cosine_similarity(a32.unsqueeze(0), b32.unsqueeze(0), dim=1).item()
    return 1.0 - float(sim)


def save_latent_reference(
    path: str,
    latent: torch.Tensor,
    *,
    metadata: dict[str, Any],
    slice_spec: dict[str, Any] | None = None,
) -> None:
    """Persist a latent bundle to ``path``.

    Storage format (dict pickled via ``torch.save``):

    * ``latent``: full latent as fp16 on cpu
    * ``shape``: original shape (list)
    * ``dtype_original``: str
    * ``expected_slice``: fp32 1-D signature slice
    * ``slice_spec``: dict describing how the slice was built
    * ``metadata``: caller-provided context (prompt, backend, steps, …)
    * ``format_version``: int

    fp16 is lossy but bounded; it keeps ref artefacts small (~a few MB
    per prompt) while preserving enough dynamic range for cosine-based
    regression. Slice values stay fp32 because the primary assertion is
    computed against them.
    """
    spec = slice_spec if slice_spec is not None else DEFAULT_SLICE_SPEC
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    latent_cpu = latent.detach().to("cpu")
    payload: dict[str, Any] = {
        "latent": latent_cpu.to(torch.float16),
        "shape": list(latent_cpu.shape),
        "dtype_original": str(latent_cpu.dtype),
        "expected_slice": _extract_expected_slice(latent_cpu, spec),
        "slice_spec": spec,
        "metadata": metadata,
        "format_version": LATENT_REFERENCE_FORMAT_VERSION,
    }
    torch.save(payload, path)


def load_latent_reference(path: str) -> dict[str, Any]:
    """Inverse of :func:`save_latent_reference` — always loads to cpu."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_latent_similarity(
    *,
    logger: Logger,
    gen_latent: torch.Tensor,
    reference_path: str,
    slice_cosine_threshold: float,
    full_cosine_threshold: float,
    model_id: str,
    attention_backend_name: str,
) -> dict[str, float]:
    ref = load_latent_reference(reference_path)
    expected_slice = ref["expected_slice"]
    ref_full = ref["latent"].to(torch.float32)
    ref_shape = tuple(ref.get("shape", ref_full.shape))
    slice_spec = ref.get("slice_spec", DEFAULT_SLICE_SPEC)

    gen_cpu = gen_latent.detach().to("cpu")
    if tuple(gen_cpu.shape) != ref_shape:
        raise AssertionError(
            f"Generated latent shape {tuple(gen_cpu.shape)} does not match "
            f"reference shape {ref_shape} for {model_id} with backend "
            f"{attention_backend_name}"
        )

    gen_slice = _extract_expected_slice(gen_cpu, slice_spec)
    slice_cos = _cosine_distance(gen_slice, expected_slice)
    # The reference full tensor was fp16-quantized at seed time. Round-trip
    # the generated tensor through fp16 so both sides share the same
    # quantization floor and the cosine distance is symmetric. This does
    # NOT affect ``slice_cos`` because the reference slice is persisted in
    # fp32 (see :func:`save_latent_reference`).
    gen_full_matched = gen_cpu.to(torch.float16).to(torch.float32)
    full_cos = _cosine_distance(gen_full_matched, ref_full)
    max_abs_diff = float(
        (gen_full_matched - ref_full).abs().max().item())

    metrics: dict[str, float] = {
        "slice_cosine_distance": slice_cos,
        "full_cosine_distance": full_cos,
        "max_abs_diff": max_abs_diff,
    }
    logger.info(
        "Latent regression metrics for %s/%s: %s",
        model_id,
        attention_backend_name,
        metrics,
    )

    failures: list[str] = []
    if slice_cos > slice_cosine_threshold:
        failures.append(
            f"slice cosine {slice_cos:.6e} > threshold "
            f"{slice_cosine_threshold:.6e}")
    if full_cos > full_cosine_threshold:
        failures.append(
            f"full cosine {full_cos:.6e} > threshold "
            f"{full_cosine_threshold:.6e}")

    if failures:
        raise AssertionError(
            f"Latent regression exceeded tolerance for {model_id} with "
            f"backend {attention_backend_name}: {'; '.join(failures)}. "
            f"Full metrics: {metrics}")

    return metrics


def _extract_latent_from_result(result: Any) -> torch.Tensor:
    """Pull a 5-D fp32 cpu latent tensor out of ``generate_video`` output."""
    if not isinstance(result, dict):
        raise RuntimeError(
            "VideoGenerator.generate_video returned unexpected payload "
            f"(type={type(result)!r}); expected dict with 'samples'.")
    samples = result.get("samples")
    if samples is None:
        raise RuntimeError(
            "VideoGenerator did not return latent samples. Ensure "
            "output_type='latent' and return_frames=True for this call.")
    if not isinstance(samples, torch.Tensor):
        raise RuntimeError(
            f"Expected torch.Tensor samples; got type={type(samples)!r}.")
    gen_latent = samples.detach().to(torch.float32).cpu()
    if gen_latent.dim() != 5:
        raise RuntimeError(
            "Expected 5-D latent (B,C,T,H,W); got shape "
            f"{tuple(gen_latent.shape)}")
    return gen_latent


def run_text_to_latent_similarity_test(
    *,
    logger: Logger,
    script_dir: str,
    device_reference_folder: str,
    prompt: str,
    attention_backend_name: str,
    model_id: str,
    default_params_map: dict[str, dict[str, object]],
    full_quality_params_map: dict[str, dict[str, object]],
    slice_cosine_threshold: float = 5e-3,
    full_cosine_threshold: float = 1e-2,
    init_kwargs_override: dict[str, object] | None = None,
    generation_kwargs_override: dict[str, object] | None = None,
    slice_spec: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Run T2V inference with ``output_type='latent'`` and compare to ref.

    Returns the computed metrics dict on success. Raises ``AssertionError``
    if any cosine tolerance is exceeded and ``FileNotFoundError`` if the
    reference artefact is missing.
    """
    spec = slice_spec if slice_spec is not None else DEFAULT_SLICE_SPEC
    with attention_backend(attention_backend_name):
        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            model_id,
            attention_backend_name,
        )
        prompt_prefix = prompt[:100].strip()
        output_latent_name = f"{prompt_prefix}{LATENT_REFERENCE_EXTENSION}"
        os.makedirs(output_dir, exist_ok=True)

        params_map = select_ssim_params(
            default_params_map,
            full_quality_params_map,
        )
        base_params = params_map[model_id]
        num_inference_steps = int(base_params["num_inference_steps"])

        init_kwargs = build_init_kwargs(base_params)
        init_kwargs["output_type"] = "latent"
        if init_kwargs_override:
            init_kwargs.update(init_kwargs_override)

        generation_kwargs = build_generation_kwargs(
            base_params,
            num_inference_steps,
            output_dir,
        )
        # We serialize latents ourselves; skip the RGB encoder path.
        generation_kwargs["save_video"] = False
        generation_kwargs["return_frames"] = True
        if generation_kwargs_override:
            generation_kwargs.update(generation_kwargs_override)

        generator: VideoGenerator | None = None
        try:
            generator = VideoGenerator.from_pretrained(
                model_path=base_params["model_path"],
                **init_kwargs,
            )
            result = generator.generate_video(prompt, **generation_kwargs)
        finally:
            shutdown_executor(generator)

    gen_latent = _extract_latent_from_result(result)

    generated_latent_path = os.path.join(output_dir, output_latent_name)
    save_latent_reference(
        generated_latent_path,
        gen_latent,
        metadata={
            "prompt": prompt,
            "model_id": model_id,
            "attention_backend": attention_backend_name,
            "num_inference_steps": num_inference_steps,
        },
        slice_spec=spec,
    )
    logger.info("Saved generated latent to %s", generated_latent_path)

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        attention_backend_name,
    )
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference folder does not exist: {reference_folder}\n"
            f"To download references, run:\n"
            f"  python fastvideo/tests/ssim/reference_videos_cli.py download")

    reference_latent_path = os.path.join(
        reference_folder,
        output_latent_name,
    )
    if not os.path.exists(reference_latent_path):
        raise FileNotFoundError(
            "Reference latent missing for prompt/backend: "
            f"{reference_latent_path}")

    return _assert_latent_similarity(
        logger=logger,
        gen_latent=gen_latent,
        reference_path=reference_latent_path,
        slice_cosine_threshold=slice_cosine_threshold,
        full_cosine_threshold=full_cosine_threshold,
        model_id=model_id,
        attention_backend_name=attention_backend_name,
    )
