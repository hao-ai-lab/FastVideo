# SPDX-License-Identifier: Apache-2.0
"""Deterministic Wan-VACE pipeline parity against Diffusers.

Set WAN_VACE_MODEL_DIR (1.3B) and WAN_VACE_14B_MODEL_DIR (14B) to local snapshots.
"""

from __future__ import annotations

import gc
from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

from tests.local_tests.wan.parity_stats import assert_dit_parity, compute_parity_stats
from tests.local_tests.wan.vace_parity_helpers import (
    assert_bf16_bitwise_equal,
    load_official_transformer,
    prepare_official_pipeline,
    prepare_vace_inputs,
    resolve_model_dir,
    run_fastvideo_stages,
    run_official_on_call,
    single_gpu_parity_runtime,
    video_generator_kwargs,
)

# VACE e2e gates (see tests/local_tests/wan/README.md):
# - conditioning inputs: bf16 bitwise (prompt/control/video/mask/ref/timesteps/initial latents)
# - every step, teacher-forced: FastVideo's prediction vs the Diffusers DiT run on FastVideo's exact
#   inputs for that step. Observed: max 0.03, abs-mean drift <= 0.2%.
# - final latent: free-running drift. The BF16 DiT itself turns a 1-ulp input change into ~2% output
#   drift (Diffusers vs Diffusers), so this gate bounds accumulation rather than implementation error.
VACE_STEP_ATOL = 0.05
VACE_STEP_RTOL = 0.05
VACE_STEP_MAX_ABS_MEAN_DRIFT = 0.01
VACE_E2E_MAX_ABS_MEAN_DRIFT = 0.04
NUM_INFERENCE_STEPS = 2

CASES = (
    ("1.3b", "t2v", "WAN_VACE_MODEL_DIR"),
    ("1.3b", "reference", "WAN_VACE_MODEL_DIR"),
    ("1.3b", "video_mask", "WAN_VACE_MODEL_DIR"),
    ("14b", "reference", "WAN_VACE_14B_MODEL_DIR"),
)


@pytest.fixture
def parity_runtime(monkeypatch):
    with single_gpu_parity_runtime(monkeypatch):
        yield


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan-VACE end-to-end parity requires CUDA")
@pytest.mark.parametrize("size,mode,env_name", CASES)
def test_wan_vace_pipeline_matches_diffusers(size, mode, env_name, tmp_path, parity_runtime):
    model_dir = resolve_model_dir(env_name)
    fastvideo_inputs, official_inputs, num_refs = prepare_vace_inputs(mode, tmp_path)
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
    official_transformer = load_official_transformer(model_dir, device)
    try:
        for step, call_args in enumerate(actual["call_args"]):
            expected = run_official_on_call(official_transformer, call_args, device)
            stats = assert_dit_parity(actual["noise_preds"][step],
                                      expected,
                                      label=f"step{step} teacher-forced prediction",
                                      atol=VACE_STEP_ATOL,
                                      rtol=VACE_STEP_RTOL,
                                      max_abs_mean_drift=VACE_STEP_MAX_ABS_MEAN_DRIFT)
            print(f"VACE_PARITY size={size} mode={mode} {stats.as_row()}", flush=True)
    finally:
        del official_transformer
        gc.collect()
        torch.cuda.empty_cache()

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
