# SPDX-License-Identifier: Apache-2.0
"""Deterministic Wan-VACE pipeline parity against Diffusers.

Set WAN_VACE_MODEL_DIR (1.3B) and WAN_VACE_14B_MODEL_DIR (14B) to local snapshots.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, cast

import imageio
import numpy as np
import pytest
import torch
from PIL import Image
from torch.testing import assert_close

from tests.local_tests.wan.parity_stats import assert_dit_parity, compute_parity_stats
from tests.local_tests.wan.vace_parity_helpers import (
    assert_bf16_bitwise_equal,
    prepare_official_pipeline,
    resolve_model_dir,
    run_fastvideo_stages,
    set_parity_cuda_flags,
    restore_parity_cuda_flags,
    video_generator_kwargs,
)

# VACE e2e gates (see tests/local_tests/wan/README.md):
# - conditioning inputs: bf16 bitwise (prompt/control/video/mask/ref/timesteps/initial latents)
# - step0 prediction: DiT component gate (assert_dit_parity)
# - final latent: multi-step drift tolerated below 5% abs-mean drift (FLUX2-style)
VACE_E2E_MAX_ABS_MEAN_DRIFT = 0.05
NUM_INFERENCE_STEPS = 2

CASES = (
    ("1.3b", "t2v", "WAN_VACE_MODEL_DIR"),
    ("1.3b", "reference", "WAN_VACE_MODEL_DIR"),
    ("1.3b", "video_mask", "WAN_VACE_MODEL_DIR"),
    ("14b", "reference", "WAN_VACE_14B_MODEL_DIR"),
)


@pytest.fixture
def parity_runtime(monkeypatch):
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.utils import get_open_port

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(get_open_port()))
    monkeypatch.setenv("DISABLE_SP", "1")
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    tf32_previous = set_parity_cuda_flags()
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        restore_parity_cuda_flags(tf32_previous)
        cleanup_dist_env_and_memory()


def _prepare_inputs(mode: str, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    if mode == "t2v":
        return {}, {}, 0
    if mode == "reference":
        image_path = tmp_path / "reference.png"
        Image.new("RGB", (64, 32), (240, 32, 24)).save(image_path)
        return {"references": [str(image_path)]}, {"reference_images": [Image.open(image_path).convert("RGB")]}, 1

    video_path = tmp_path / "video.mp4"
    mask_path = tmp_path / "mask.mp4"
    video_frames = np.stack([np.full((64, 64, 3), (i * 24, 64, 160), dtype=np.uint8) for i in range(5)])
    mask_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    mask_frames[:, :, 32:] = 255
    imageio.mimwrite(video_path, video_frames, fps=16, codec="libx264")
    imageio.mimwrite(mask_path, mask_frames, fps=16, codec="libx264")
    from fastvideo.models.vision_utils import load_video

    decoded_video, _ = load_video(str(video_path), return_fps=True)
    decoded_mask, _ = load_video(str(mask_path), return_fps=True)
    return ({"video_path": str(video_path), "mask_path": str(mask_path)},
            {"video": decoded_video, "mask": decoded_mask}, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan-VACE end-to-end parity requires CUDA")
@pytest.mark.parametrize("size,mode,env_name", CASES)
def test_wan_vace_pipeline_matches_diffusers(size, mode, env_name, tmp_path, parity_runtime):
    model_dir = resolve_model_dir(env_name)
    fastvideo_inputs, official_inputs, num_refs = _prepare_inputs(mode, tmp_path)
    latent_frames = (5 - 1) // 4 + 1 + num_refs
    latents = torch.randn((1, 16, latent_frames, 8, 8), generator=torch.Generator().manual_seed(42))
    prompt = "a red panda reading a book"
    device = torch.device("cuda")

    official = prepare_official_pipeline(model_dir, device)
    official_calls: dict[str, Any] = {"model_inputs": [], "noise_preds": []}
    from tests.local_tests.wan.vace_parity_helpers import register_transformer_capture

    capture_handles = register_transformer_capture(official.transformer, official_calls)
    try:
        with torch.inference_mode():
            official_prompt, _ = official.encode_prompt(prompt,
                                                        do_classifier_free_guidance=False,
                                                        max_sequence_length=512,
                                                        device=device)
            video, mask, references = official.preprocess_conditions(**official_inputs,
                                                                     height=64,
                                                                     width=64,
                                                                     num_frames=5,
                                                                     dtype=torch.float32,
                                                                     device=device)
            control_video = official.prepare_video_latents(video, mask, references, None, device)
            control_mask = official.prepare_masks(mask, references)
            official_control = torch.cat([control_video, control_mask], dim=1).cpu()
            official_latents = official(prompt=prompt,
                                        guidance_scale=1.0,
                                        num_inference_steps=NUM_INFERENCE_STEPS,
                                        height=64,
                                        width=64,
                                        num_frames=5,
                                        latents=latents.clone(),
                                        output_type="latent",
                                        max_sequence_length=512,
                                        **official_inputs).frames.detach().cpu()
            official_timesteps = official.scheduler.timesteps.detach().cpu()
    finally:
        for capture_handle in capture_handles:
            capture_handle.remove()
        del official
        gc.collect()
        torch.cuda.empty_cache()

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(str(model_dir), **video_generator_kwargs())
    try:
        request = dict(prompt=prompt,
                       negative_prompt="",
                       height=64,
                       width=64,
                       num_frames=5,
                       fps=16,
                       num_inference_steps=NUM_INFERENCE_STEPS,
                       guidance_scale=1.0,
                       latents=latents.clone(),
                       save_video=False,
                       return_frames=True,
                       **fastvideo_inputs)
        actual = cast(dict[str, Any], generator.executor.collective_rpc(
            run_fastvideo_stages,
            kwargs={"model_path": str(model_dir), "request": request})[0])
    finally:
        generator.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

    assert_close(actual["video"], video.cpu(), rtol=0, atol=0)
    if actual["mask"] is not None:
        assert_close((actual["mask"] + 1) / 2, mask.cpu(), rtol=0, atol=0)
    for actual_ref, official_ref in zip(actual["references"], references[0], strict=True):
        assert_close(actual_ref, official_ref.cpu(), rtol=0, atol=0)
    assert_bf16_bitwise_equal(actual["prompt_embeds"], official_prompt.cpu(), "prompt_embeds")
    assert_bf16_bitwise_equal(actual["control"], official_control, "control_hidden_states")
    assert_close(actual["timesteps"], official_timesteps, rtol=0, atol=0)
    assert_close(actual["initial_latents"], latents, rtol=0, atol=0)
    print(f"VACE_PARITY size={size} mode={mode} conditioning=bitwise_equal", flush=True)

    assert len(actual["noise_preds"]) == len(official_calls["noise_preds"]) == NUM_INFERENCE_STEPS
    step0_stats = compute_parity_stats(actual["noise_preds"][0], official_calls["noise_preds"][0],
                                       "step0 prediction")
    print(f"VACE_PARITY size={size} mode={mode} {step0_stats.as_row()}", flush=True)
    assert_dit_parity(actual["noise_preds"][0], official_calls["noise_preds"][0], label="step0 prediction")

    final_official_latents = official_latents[:, :, num_refs:]
    final_stats = compute_parity_stats(actual["latents"], final_official_latents, "final latents")
    strict_matches = torch.isclose(actual["latents"].float(),
                                   final_official_latents.float(),
                                   rtol=1e-2,
                                   atol=1e-2)
    strict_mismatches = strict_matches.numel() - int(strict_matches.sum().item())
    print(f"VACE_PARITY size={size} mode={mode} {final_stats.as_row()} "
          f"strict_1e-2_mismatches={strict_mismatches}/{strict_matches.numel()}", flush=True)
    assert final_stats.abs_mean_drift_ratio < VACE_E2E_MAX_ABS_MEAN_DRIFT, (
        f"{final_stats.as_row()}; max abs={final_stats.max_abs:.6g}")
