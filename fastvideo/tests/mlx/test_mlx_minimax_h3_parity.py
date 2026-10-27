# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 parity: the MLX runtime vs the upstream PyTorch reference (#1674).

The H3 analogue of ``test_mlx_dit_parity.py`` — the trust gate for the Apple
Silicon H3 track. A tiny random-weight ``MiniMaxH3Transformer3DModel`` runs
end to end (packing -> token refiner -> AdaLN blocks -> dual heads) in
PyTorch and in ``fastvideo.mlx_runtime.minimax_h3.MLXMiniMaxH3DiT`` with
identical weights, and outputs must match within pinned fp32 tolerances.

Also gated: the packed-layout geometry vs the upstream packing builder, the
dual rectified-flow scheduler vs the upstream scheduler, the AdaLN precompute
cache vs the faithful forward, and int8 quantization SNR.

Runs anywhere MLX is installed (Metal on Apple Silicon, CPU elsewhere):

    pytest fastvideo/tests/mlx/test_mlx_minimax_h3_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 parity tests")
torch = pytest.importorskip("torch", reason="PyTorch supplies the independent H3 reference")

from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec  # noqa: E402
from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MINIMAX_H3_AUDIO_SHIFT,
    MINIMAX_H3_VIDEO_SHIFT,
    MiniMaxH3SchedulerState,
    MiniMaxH3StepCache,
    build_packed_layout,
    load_mlx_h3_checkpoint,
    minimax_h3_sigmas,
    save_mlx_h3_checkpoint,
)
from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402
    LATENT_HEIGHT,
    LATENT_WIDTH,
    NUM_AUDIO_LATENTS,
    NUM_LATENT_FRAMES,
    NUM_TEXT_TOKENS,
    TINY_ARCH,
    build_hf_config,
    build_inputs,
    build_tiny_h3_config,
    build_torch_model,
    mlx_cache_output,
    mlx_dit_from_torch_model,
    mlx_output,
    torch_reference_output,
)

# fp32 full-forward tolerances, matching the Wan parity gate's headroom logic.
FP32_ATOL = 2e-4
FP32_RTOL = 2e-4

# int8 (group 64) on tiny random weights lands far above this; a broken dequant
# path lands near 0 dB.
INT8_MIN_SNR_DB = 20.0

# The AdaLN cache reorders the same math, so it should be nearly exact.
CACHE_ATOL = 1e-4


def _snr_db(reference: np.ndarray, actual: np.ndarray) -> float:
    signal = float(np.mean(np.square(reference)))
    noise = float(np.mean(np.square(reference - actual)))
    return 10.0 * np.log10(signal / max(noise, 1e-20))


def test_packed_layout_matches_upstream_builder(distributed_setup) -> None:
    """The numpy packing geometry must equal the upstream torch builder's."""
    from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence

    layout = build_packed_layout(
        NUM_TEXT_TOKENS,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        patch_size=tuple(TINY_ARCH["patch_size"]),
    )
    reference = build_packed_sequence(
        torch.full((NUM_TEXT_TOKENS, ), 1, dtype=torch.long),
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        tuple(TINY_ARCH["patch_size"]),
    )
    assert layout.sequence_length == reference.sequence_length
    np.testing.assert_array_equal(layout.position_ids, reference.position_ids.numpy())
    np.testing.assert_array_equal(layout.token_tags, reference.token_tags.numpy())
    np.testing.assert_array_equal(layout.video_indices, reference.video_indices.numpy())
    np.testing.assert_array_equal(layout.audio_indices, reference.audio_indices.numpy())
    np.testing.assert_array_equal(layout.text_indices, reference.text_indices.numpy())


def test_packed_layout_with_keyframes(distributed_setup) -> None:
    """FL2VA anchors: 'first' pins time at n_text; 'last' at span - 5/3."""
    from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence
    layout = build_packed_layout(
        NUM_TEXT_TOKENS,
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        patch_size=tuple(TINY_ARCH["patch_size"]),
        keyframe_anchors=("first", "last"),
    )
    reference = build_packed_sequence(
        torch.full((NUM_TEXT_TOKENS, ), 1, dtype=torch.long),
        NUM_LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        NUM_AUDIO_LATENTS,
        tuple(TINY_ARCH["patch_size"]),
        keyframe_anchors=("first", "last"),
    )
    assert layout.sequence_length == reference.sequence_length
    np.testing.assert_array_equal(layout.position_ids, reference.position_ids.numpy())
    np.testing.assert_array_equal(layout.token_tags, reference.token_tags.numpy())
    np.testing.assert_array_equal(layout.video_indices, reference.video_indices.numpy())


def test_scheduler_matches_upstream(distributed_setup) -> None:
    """Sigma grids and data-ward Euler steps match the upstream scheduler."""
    import torch
    from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    num_steps = 6
    for shift in (MINIMAX_H3_VIDEO_SHIFT, MINIMAX_H3_AUDIO_SHIFT):
        reference = MiniMaxH3Scheduler(shift=shift)
        reference.set_timesteps(num_steps + 1)  # reference counts sigmas, not steps
        ours = MiniMaxH3SchedulerState.create(shift, num_steps)
        np.testing.assert_allclose(ours.sigmas, reference.sigmas.numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(ours.timesteps, reference.timesteps.numpy(), rtol=1e-6, atol=1e-7)

        generator = torch.Generator(device="cpu").manual_seed(7)
        sample = torch.randn(2, 8, generator=generator, dtype=torch.float32)
        model_output = torch.randn(2, 8, generator=generator, dtype=torch.float32)
        import mlx.core as mx

        mlx_sample = mx.array(sample.numpy())
        mlx_output_step = mx.array(model_output.numpy())
        for step_index in range(num_steps):
            sample = reference.step(model_output, reference.timesteps[step_index], sample).prev_sample
            mlx_sample = ours.step(mlx_output_step, step_index, mlx_sample)
            np.testing.assert_allclose(
                np.asarray(mlx_sample),
                sample.numpy(),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"scheduler step {step_index} diverged (shift={shift})",
            )


def test_full_dit_forward_matches_torch_reference(distributed_setup) -> None:
    model = build_torch_model()
    hf_config = build_hf_config(build_tiny_h3_config())
    layout, torch_inputs, mlx_inputs = build_inputs()
    reference_video, reference_audio = torch_reference_output(model, torch_inputs)

    dit = mlx_dit_from_torch_model(model, hf_config)
    video, audio = mlx_output(dit, layout, mlx_inputs)

    np.testing.assert_allclose(video, reference_video, atol=FP32_ATOL, rtol=FP32_RTOL)
    np.testing.assert_allclose(audio, reference_audio, atol=FP32_ATOL, rtol=FP32_RTOL)


def test_adaln_cache_matches_faithful_forward(distributed_setup) -> None:
    """The precompute-cache path (the 24 GB Mac enabler) must match the
    faithful temb path — it is the same math, served from precomputed rows."""
    model = build_torch_model()
    hf_config = build_hf_config(build_tiny_h3_config())
    layout, _, mlx_inputs = build_inputs()

    dit = mlx_dit_from_torch_model(model, hf_config)
    video_ref, audio_ref = mlx_output(dit, layout, mlx_inputs)
    video_cache, audio_cache = mlx_cache_output(dit, layout, mlx_inputs)

    np.testing.assert_allclose(video_cache, video_ref, atol=CACHE_ATOL, rtol=CACHE_ATOL)
    np.testing.assert_allclose(audio_cache, audio_ref, atol=CACHE_ATOL, rtol=CACHE_ATOL)
    # The cache must actually free the modulation projections.
    for block in dit.blocks:
        assert block["adaln_proj.linear.weight"] is None
        assert block["adaln_proj.linear.bias"] is None


def test_quantized_checkpoint_with_adaln_cache_round_trips(tmp_path, distributed_setup) -> None:
    """The converted checkpoint format must preserve quantization and cached inference."""
    import mlx.core as mx

    model = build_torch_model()
    hf_config = build_hf_config(build_tiny_h3_config())
    layout, _, mlx_inputs = build_inputs()
    dit = mlx_dit_from_torch_model(
        model,
        hf_config,
        quantization=MLXQuantizationSpec.from_name("int8"),
    )
    expected_video, expected_audio = mlx_cache_output(dit, layout, mlx_inputs)
    save_mlx_h3_checkpoint(dit, tmp_path)

    restored = load_mlx_h3_checkpoint(tmp_path)
    video, audio = restored.forward_with_cache(
        mx.array(mlx_inputs["video_rows"]),
        mx.array(mlx_inputs["audio_rows"]),
        mx.array(mlx_inputs["text_rows"]),
        layout=layout,
        step_timesteps=mlx_inputs["timesteps"],
        row_timestep_inverse=mlx_inputs["timestep_indices"],
    )
    mx.eval(video, audio)

    np.testing.assert_allclose(np.asarray(video), expected_video, atol=CACHE_ATOL, rtol=CACHE_ATOL)
    np.testing.assert_allclose(np.asarray(audio), expected_audio, atol=CACHE_ATOL, rtol=CACHE_ATOL)
    assert restored._adaln_cache is not None
    assert all("adaln_proj.linear.weight" not in block for block in restored.blocks)


def test_step_cache_reports_out_of_range_timestep() -> None:
    cache = MiniMaxH3StepCache(
        timesteps=np.asarray([0.25, 0.5], dtype=np.float32),
        block_tables=[],
        norm_out_shift=None,
        norm_out_scale=None,
    )

    with pytest.raises(ValueError, match="not in the cached schedule union"):
        cache.positions(np.asarray([0.75], dtype=np.float32))


def test_full_dit_forward_int8_stays_close_to_fp32(distributed_setup) -> None:
    model = build_torch_model()
    hf_config = build_hf_config(build_tiny_h3_config())
    layout, _, mlx_inputs = build_inputs()

    fp32_dit = mlx_dit_from_torch_model(model, hf_config)
    reference_video, reference_audio = mlx_output(fp32_dit, layout, mlx_inputs)

    int8_spec = MLXQuantizationSpec.from_name("int8")
    int8_dit = mlx_dit_from_torch_model(model, hf_config, quantization=int8_spec)
    video, audio = mlx_output(int8_dit, layout, mlx_inputs)

    assert _snr_db(reference_video, video) >= INT8_MIN_SNR_DB
    assert _snr_db(reference_audio, audio) >= INT8_MIN_SNR_DB


def test_h3_int6_uses_affine_group_64() -> None:
    spec = MLXQuantizationSpec.from_name("int6")

    assert spec is not None
    assert spec.mode == "affine"
    assert spec.bits == 6
    assert spec.group_size == 64


def test_sigma_grid_shape_and_monotonicity() -> None:
    for shift in (MINIMAX_H3_VIDEO_SHIFT, MINIMAX_H3_AUDIO_SHIFT, 1.0):
        sigmas = minimax_h3_sigmas(shift, 3)
        assert sigmas.shape == (4, )
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == 0.0
        assert np.all(np.diff(sigmas) < 0)
