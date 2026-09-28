# SPDX-License-Identifier: Apache-2.0
"""Local, weighted Wan-VACE single-GPU versus two-GPU SP2 pipeline checks.

Set WAN_VACE_SP_E2E=1 and the two WAN_VACE_*_MODEL_DIR variables to run.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.pipelines.test_wan_vace_end_to_end_parity import _prepare_inputs
from tests.local_tests.wan.parity_stats import compute_parity_stats
from tests.local_tests.wan.vace_parity_helpers import (
    resolve_model_dir,
    run_fastvideo_stages,
    video_generator_kwargs,
)

CASES = (
    ("1.3b", "reference", "WAN_VACE_MODEL_DIR"),
    ("1.3b", "video_mask", "WAN_VACE_MODEL_DIR"),
    ("14b", "reference", "WAN_VACE_14B_MODEL_DIR"),
)


def _run_pipeline(model_dir: Path, request: dict[str, Any], sp_size: int) -> list[dict[str, Any]]:
    from fastvideo import VideoGenerator

    kwargs = video_generator_kwargs()
    kwargs.update(num_gpus=sp_size, sp_size=sp_size, output_type="video")
    generator = VideoGenerator.from_pretrained(str(model_dir), **kwargs)
    try:
        return cast(list[dict[str, Any]], generator.executor.collective_rpc(
            run_fastvideo_stages,
            kwargs={"model_path": str(model_dir), "request": request, "capture_decoded": True,
                    "trace_sp_layers": os.environ.get("WAN_VACE_SP_TRACE") == "1"},
        ))
    finally:
        generator.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


def _frame_ssim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    from pytorch_msssim import ssim

    assert actual.shape == expected.shape == (1, 3, 5, 64, 64)
    frames_a = actual.permute(0, 2, 1, 3, 4).flatten(0, 1).clamp(0, 1)
    frames_b = expected.permute(0, 2, 1, 3, 4).flatten(0, 1).clamp(0, 1)
    with torch.inference_mode():
        return float(ssim(frames_a, frames_b, data_range=1.0).item())


def _assert_same_conditions(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("video", "prompt_embeds", "control", "timesteps", "initial_latents"):
        assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
    if expected["mask"] is None:
        assert actual["mask"] is None
    else:
        assert_close(actual["mask"], expected["mask"], rtol=0, atol=0, msg="mask")
    assert len(actual["references"]) == len(expected["references"])
    for actual_ref, expected_ref in zip(actual["references"], expected["references"], strict=True):
        assert_close(actual_ref, expected_ref, rtol=0, atol=0, msg="reference")
    assert len(actual["noise_preds"]) == len(expected["noise_preds"]) == 2


def _report_layer_boundaries(single: dict[str, Any], ranks: list[dict[str, Any]]) -> None:
    """Print the first-step boundary sequence before applying the numeric gates."""
    sharded = {"norm_out"}
    sharded.update(name for name in single["layers"] if name.startswith(("blocks.", "vace_blocks.")))
    for name, outputs in single["layers"].items():
        if name in sharded:
            actual = torch.cat([rank["layers"][name][0] for rank in ranks], dim=1)
        else:
            actual = ranks[0]["layers"][name][0]
        expected = outputs[0]
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        stats = compute_parity_stats(actual, expected, name)
        print(f"VACE_SP_TRACE {stats.as_row()}", flush=True)


@pytest.mark.skipif(os.environ.get("WAN_VACE_SP_E2E") != "1", reason="set WAN_VACE_SP_E2E=1")
@pytest.mark.xfail(
    strict=True,
    reason="BF16 GEMM rounding depends on SP-sharded token row count; "
    "single-GPU vs SP2 differs from first VACE/FFN linear",
)
@pytest.mark.parametrize("size,mode,env_name", CASES)
def test_wan_vace_sp2_matches_single_pipeline(size: str, mode: str, env_name: str, tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Wan-VACE SP2 requires two CUDA devices")
    model_dir = resolve_model_dir(env_name)
    fastvideo_inputs, _, num_refs = _prepare_inputs(mode, tmp_path)
    latent_frames = (5 - 1) // 4 + 1 + num_refs
    latents = torch.randn((1, 16, latent_frames, 8, 8), generator=torch.Generator().manual_seed(42))
    request = dict(
        prompt="a red panda reading a book", negative_prompt="", height=64, width=64,
        num_frames=5, fps=16, num_inference_steps=2, guidance_scale=1.0,
        latents=latents, save_video=False, return_frames=True, **fastvideo_inputs,
    )
    single = _run_pipeline(model_dir, request, sp_size=1)[0]
    parallel_ranks = _run_pipeline(model_dir, request, sp_size=2)
    assert len(parallel_ranks) == 2
    if os.environ.get("WAN_VACE_SP_TRACE") == "1":
        _report_layer_boundaries(single, parallel_ranks)
    failures = []
    for rank, parallel in enumerate(parallel_ranks):
        _assert_same_conditions(parallel, single)
        for step, (actual, expected) in enumerate(zip(parallel["noise_preds"], single["noise_preds"], strict=True)):
            stats = compute_parity_stats(actual, expected, f"rank{rank} step{step} noise")
            print(f"VACE_SP size={size} mode={mode} {stats.as_row()}", flush=True)
            try:
                assert_close(actual, expected, atol=0.02, rtol=0.02, msg=stats.as_row())
            except AssertionError:
                failures.append(stats.as_row())
        final_stats = compute_parity_stats(parallel["denoised_latents"], single["denoised_latents"],
                                           f"rank{rank} final latent")
        score = _frame_ssim(parallel["latents"], single["latents"])
        print(f"VACE_SP size={size} mode={mode} {final_stats.as_row()} frame_ssim={score:.8f}", flush=True)
        if final_stats.abs_mean_drift_ratio >= 0.01:
            failures.append(final_stats.as_row())
        if score < 0.99:
            failures.append(f"rank{rank} frame_ssim={score:.8f} < 0.99")
    assert not failures, "\n".join(failures)
