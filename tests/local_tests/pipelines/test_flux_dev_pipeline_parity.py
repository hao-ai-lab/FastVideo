# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline parity test for FLUX.1-dev.

Compares FastVideo FluxPipeline decoded image output against Diffusers
FluxPipeline under identical prompt, seed, and inference parameters.

Coverage scope: production_loader + implementation_subcomponent
Comparison target: decoded RGB image (CHW float32, values in [0, 1])

Requires ``official_weights/FLUX.1-dev`` (or set ``FLUX_DEV_ROOT``) and CUDA.

Run from repo root::

    FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
    pytest tests/local_tests/pipelines/test_flux_dev_pipeline_parity.py -vs
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29521")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLUX_DEV_ROOT = os.environ.get(
    "FLUX_DEV_ROOT",
    str(_REPO_ROOT / "official_weights" / "FLUX.1-dev"),
)

# Parity test parameters — small enough to run quickly, large enough to be meaningful.
_PROMPT = "a photo of a cat"
_HEIGHT = 256
_WIDTH = 256
_NUM_INFERENCE_STEPS = 4
_SEED = 42
_GUIDANCE_SCALE = 3.5

# Tolerance for full pipeline (text encode → denoise → VAE decode accumulates error).
# Tighter than 1e-2 is unrealistic across different attention backends and precision paths.
_ATOL = 5e-2
_RTOL = 5e-2


def _log_stats(label: str, t: torch.Tensor) -> None:
    tf = t.float()
    print(
        f"[FLUX PARITY] {label}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={tf.min():.4f} max={tf.max():.4f} mean={tf.mean():.4f} std={tf.std():.4f}"
    )


def _requires_weights() -> bool:
    return Path(_FLUX_DEV_ROOT, "model_index.json").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not _requires_weights(), reason="requires official_weights/FLUX.1-dev (set FLUX_DEV_ROOT)")
def test_flux_dev_pipeline_image_parity() -> None:
    """FastVideo FluxPipeline decoded image matches Diffusers FluxPipeline."""

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ------------------------------------------------------------------
    # 1. Diffusers reference
    # ------------------------------------------------------------------
    print("\n[FLUX PARITY] Running Diffusers reference pipeline...")
    from diffusers import FluxPipeline as HFFluxPipeline

    hf_pipe = HFFluxPipeline.from_pretrained(
        _FLUX_DEV_ROOT,
        torch_dtype=dtype,
    ).to(device)
    hf_pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(_SEED)
    hf_out = hf_pipe(
        prompt=_PROMPT,
        height=_HEIGHT,
        width=_WIDTH,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        guidance_scale=_GUIDANCE_SCALE,
        generator=generator,
        output_type="pt",
    )
    # hf_out.images: [B, C, H, W] float in [0,1] when output_type="pt"
    hf_image = hf_out.images[0].float().cpu()  # [C, H, W]
    _log_stats("Diffusers output", hf_image)

    del hf_pipe
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2. FastVideo pipeline
    # ------------------------------------------------------------------
    print("[FLUX PARITY] Running FastVideo pipeline...")
    from fastvideo.configs.pipelines.flux import FluxPipelineConfig
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
    from fastvideo.pipelines.basic.flux.flux_pipeline import FluxPipeline
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    torch.manual_seed(_SEED)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        args = FastVideoArgs(
            model_path=_FLUX_DEV_ROOT,
            pipeline_config=FluxPipelineConfig(),
            workload_type=WorkloadType.T2I,
            hsdp_shard_dim=1,
            pin_cpu_memory=False,
            distributed_executor_backend="mp",
        )
        pipeline = FluxPipeline(_FLUX_DEV_ROOT, args)

        batch = ForwardBatch(
            data_type="image",
            prompt=_PROMPT,
            height=_HEIGHT,
            width=_WIDTH,
            seed=_SEED,
            num_inference_steps=_NUM_INFERENCE_STEPS,
            guidance_scale=_GUIDANCE_SCALE,
            use_embedded_guidance=True,
            true_cfg_scale=1.0,
            num_videos_per_prompt=1,
            save_video=False,
        )
        fv_out = pipeline.forward(batch, args)
        # fv_out.output: [B, C, F, H, W] — squeeze frame dim for T2I
        fv_image = fv_out.output[0, :, 0].float().cpu()  # [C, H, W]
        _log_stats("FastVideo output", fv_image)
    finally:
        cleanup_dist_env_and_memory()

    # ------------------------------------------------------------------
    # 3. Compare
    # ------------------------------------------------------------------
    print("[FLUX PARITY] Comparing outputs...")
    assert hf_image.shape == fv_image.shape, (
        f"Shape mismatch: Diffusers {hf_image.shape} vs FastVideo {fv_image.shape}"
    )

    abs_diff = (hf_image - fv_image).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    print(f"[FLUX PARITY] max_diff={max_diff:.4f}  mean_diff={mean_diff:.4f}")

    assert_close(hf_image, fv_image, atol=_ATOL, rtol=_RTOL)
    print("[FLUX PARITY] PASSED")
