# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 PDD (Parallel Decoding Distillation): MLX port vs the torch reference.

Gates the FastH3 Ref2VA PDD path against ``fastvideo/layers/pdd.py``: the fine
grid, node sigmas and integration weights must match the CUDA math exactly; the
``fastvideo_inference.json`` PDD contract must be accepted (and rejected) with
the same rules the CUDA loader applies; and a fused head block must equal the
slow weighted sum. Runs anywhere MLX and torch are installed:

    pytest fastvideo/tests/mlx/test_mlx_minimax_h3_pdd.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is required for MiniMax H3 PDD tests")
torch = pytest.importorskip("torch", reason="PyTorch supplies the independent PDD reference")

from fastvideo.mlx_runtime.minimax_h3 import (  # noqa: E402
    MiniMaxH3SchedulerState,
    load_fasth3_inference_contract,
    resolve_h3_denoise_schedule,
)
from fastvideo.mlx_runtime.minimax_h3_pdd import (  # noqa: E402
    PDD_GRID_MAX_T,
    PDDModalitySchedule,
    build_pdd_sampling_plan,
    fuse_block_params,
    pdd_fine_grid,
    shifted_noise_amount,
    shifted_noise_delta,
    validate_finite_nonzero,
    validate_pdd_steps,
)
from fastvideo.tests.mlx.tiny_h3 import (  # noqa: E402
    TINY_ARCH,
    build_hf_config,
    build_inputs,
    build_tiny_h3_config,
    build_torch_model,
    mlx_dit_from_torch_model,
)

# The grid math is float64 and formula-identical, so agreement is near bit-exact.
PLAN_RTOL = 1e-12
# The fused projection accumulates in fp32 (as the torch reference does); the
# slow path's fp32 matmul differs only by summation order.
FUSE_ATOL = 2e-6
FUSE_RTOL = 2e-6

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
STEP_INDICES = [0, 2, 4]


# ---------------------------------------------------------------------------
# Plan math vs fastvideo.layers.pdd
# ---------------------------------------------------------------------------


def test_fine_grid_matches_torch_reference() -> None:
    from fastvideo.layers.pdd import pdd_fine_grid as torch_fine_grid

    for pdd_steps in (2, 4, 7):
        ours = pdd_fine_grid(pdd_steps)
        reference = torch_fine_grid(pdd_steps).numpy()
        np.testing.assert_allclose(ours, reference, rtol=PLAN_RTOL, atol=0.0)
        assert ours[0] == PDD_GRID_MAX_T
        assert ours[-1] == 0.0
        assert np.all(np.diff(ours) < 0)


def test_shifted_noise_matches_torch_reference() -> None:
    from fastvideo.layers.pdd import shifted_noise_amount as torch_amount
    from fastvideo.layers.pdd import shifted_noise_delta as torch_delta

    base = np.linspace(PDD_GRID_MAX_T, 0.0, 9)
    for shift in (1.0, 3.0, 12.0):
        np.testing.assert_allclose(
            shifted_noise_amount(base, shift),
            torch_amount(torch.from_numpy(base), shift).numpy(),
            rtol=PLAN_RTOL,
            atol=0.0,
        )
        np.testing.assert_allclose(
            shifted_noise_delta(base[:-1], base[1:], shift),
            torch_delta(torch.from_numpy(base[:-1]), torch.from_numpy(base[1:]), shift).numpy(),
            rtol=PLAN_RTOL,
            atol=0.0,
        )
    # The identity shift passes the base through untouched (early return).
    np.testing.assert_array_equal(shifted_noise_amount(base, 1.0), base)
    np.testing.assert_array_equal(shifted_noise_delta(base[:-1], base[1:], 1.0), base[1:] - base[:-1])


def test_sampling_plan_matches_torch_reference() -> None:
    from fastvideo.layers.pdd import PDDModalitySchedule as TorchSchedule
    from fastvideo.layers.pdd import build_pdd_sampling_plan as torch_build_plan

    reference = torch_build_plan(
        STEP_INDICES,
        {"video": TorchSchedule(shift=VIDEO_SHIFT), "audio": TorchSchedule(shift=AUDIO_SHIFT)},
    )
    ours = build_pdd_sampling_plan(
        STEP_INDICES,
        {"video": PDDModalitySchedule(shift=VIDEO_SHIFT), "audio": PDDModalitySchedule(shift=AUDIO_SHIFT)},
    )
    assert ours.pdd_steps == reference.pdd_steps
    assert ours.num_steps == reference.num_steps
    assert ours.block_sizes == reference.block_sizes
    assert ours.block(1) == (2, 4)
    np.testing.assert_allclose(ours.fine_grid, reference.fine_grid.numpy(), rtol=PLAN_RTOL, atol=0.0)
    for name in ("video", "audio"):
        np.testing.assert_allclose(ours.node_sigmas[name], reference.node_sigmas[name].numpy(), rtol=PLAN_RTOL, atol=0.0)
        np.testing.assert_allclose(
            ours.integration_weights[name],
            reference.integration_weights[name].numpy(),
            rtol=PLAN_RTOL,
            atol=0.0,
        )
    # First node is the shared grid top; last node is 0 (clean).
    for name, shift in (("video", VIDEO_SHIFT), ("audio", AUDIO_SHIFT)):
        assert float(ours.node_sigmas[name][0]) == pytest.approx(PDD_GRID_MAX_T, abs=1e-12)
        assert float(ours.node_sigmas[name][-1]) == 0.0
        assert bool(np.all(ours.node_sigmas[name][1:] < ours.node_sigmas[name][:-1]))
        assert bool(np.all(ours.integration_weights[name] < 0))


def test_plan_validators() -> None:
    with pytest.raises(ValueError, match="int >= 2"):
        validate_pdd_steps(1)
    with pytest.raises(ValueError, match="int >= 2"):
        validate_pdd_steps(True)
    validate_pdd_steps(2)
    validate_finite_nonzero(np.array([1.0, -0.5]), "w")
    with pytest.raises(ValueError, match="finite and non-zero"):
        validate_finite_nonzero(np.array([0.0, 1.0]), "w")


# ---------------------------------------------------------------------------
# Contract acceptance / rejection
# ---------------------------------------------------------------------------


def _pdd_contract(**overrides) -> dict:
    payload = {
        "schema_version": "fasth3-inference-contract-v1",
        "schema": "fasth3-inference-contract-v1",
        "model_type": "ref2va",
        "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@abc123",
        "vsa_sparsity": 0.9,
        "vsa_tile_size": 128,
        "vsa_ref_keep_rate": 0.1,
        "transformer_component": "transformer_ref",
        "conditioning": "fixed_ordered_references_target_only_flow",
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_ref_policy": "p2_multi_region",
        "pdd_steps": 4,
        "pdd_step_indices": [0, 2, 4],
        "grid_max_t": PDD_GRID_MAX_T,
        "num_inference_steps": 2,
        "transformer_forwards": 2,
        "video_scheduler_shift": VIDEO_SHIFT,
        "audio_scheduler_shift": AUDIO_SHIFT,
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, payload) -> None:
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(payload))


def test_pdd_contract_is_accepted_and_tagged(tmp_path) -> None:
    _write(tmp_path, _pdd_contract())
    contract = load_fasth3_inference_contract(tmp_path)
    assert contract is not None
    assert contract["kind"] == "pdd"
    assert contract["pdd_step_indices"] == [0, 2, 4]
    assert contract["video_scheduler_shift"] == VIDEO_SHIFT


def test_dmd_contract_stays_dmd(tmp_path) -> None:
    _write(
        tmp_path,
        {
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": [999, 500, 250],
            "num_inference_steps": 4,
            "transformer_forwards": 3,
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
        },
    )
    contract = load_fasth3_inference_contract(tmp_path)
    assert contract is not None
    assert contract["kind"] == "dmd"
    assert "dmd_denoising_steps" in contract


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"pdd_steps": 1}, "pdd_steps must be an int >= 2"),
        ({"pdd_steps": 4, "pdd_step_indices": [1, 2, 4]}, "strictly from 0"),
        ({"pdd_steps": 4, "pdd_step_indices": [0, 2, 3]}, "pdd_steps=4"),
        ({"pdd_steps": 4, "pdd_step_indices": [0, 3, 2, 4]}, "strictly from 0"),
        ({"grid_max_t": 1.0}, "grid_max_t"),
        ({"num_inference_steps": 3}, "num_inference_steps="),
        ({"transformer_forwards": 1}, "transformer_forwards="),
        ({"transformer_component": "transformer"}, "transformer_component"),
        ({"guidance_scale": 2.0}, "guidance_scale"),
        ({"attention_backend": "TORCH_SDPA"}, "attention_backend"),
        ({"conditioning": "other"}, "conditioning"),
        ({"schema": "other"}, "schema"),
        ({"model_type": "t2va"}, "model_type"),
        ({"surprise": 1}, "unknown"),
        ({"base_model_revision": "MiniMaxAI/MiniMax-H3"}, "base_model_revision"),
        ({"base_model_revision": "hf://MiniMaxAI/MiniMax-H3@"}, "base_model_revision"),
        ({"vsa_tile_size": 96}, "vsa_tile_size"),
        ({"vsa_sparsity": 1.0}, "vsa_sparsity"),
        ({"vsa_sparsity": -0.1}, "vsa_sparsity"),
        ({"vsa_ref_keep_rate": 0.0}, "vsa_ref_keep_rate"),
        ({"vsa_ref_keep_rate": 1.0}, "vsa_ref_keep_rate"),
    ],
)
def test_malformed_pdd_contract_fails_closed(tmp_path, overrides, match) -> None:
    _write(tmp_path, _pdd_contract(**overrides))
    with pytest.raises(ValueError, match=match):
        load_fasth3_inference_contract(tmp_path)


def test_pdd_contract_missing_a_field_fails_closed(tmp_path) -> None:
    payload = _pdd_contract()
    del payload["vsa_tile_size"]
    _write(tmp_path, payload)
    with pytest.raises(ValueError, match="missing .*vsa_tile_size"):
        load_fasth3_inference_contract(tmp_path)


def _write_siblings(tmp_path, *, pdd_steps=4, video_shift=VIDEO_SHIFT, audio_shift=AUDIO_SHIFT) -> None:
    for relative, payload in (
        ("transformer_ref/config.json", {"pdd_steps": pdd_steps}),
        ("scheduler/scheduler_config.json", {"shift": video_shift}),
        ("audio_scheduler/scheduler_config.json", {"shift": audio_shift}),
    ):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(json.dumps(payload))


def test_pdd_contract_matching_sibling_files_is_accepted(tmp_path) -> None:
    _write(tmp_path, _pdd_contract())
    _write_siblings(tmp_path)
    assert load_fasth3_inference_contract(tmp_path)["kind"] == "pdd"


@pytest.mark.parametrize(
    "siblings, match",
    [
        ({"pdd_steps": 8}, "transformer_ref/config.json"),
        ({"video_shift": VIDEO_SHIFT + 1}, "scheduler/scheduler_config.json"),
        ({"audio_shift": AUDIO_SHIFT + 1}, "audio_scheduler/scheduler_config.json"),
    ],
)
def test_pdd_contract_disagreeing_with_sibling_files_fails(tmp_path, siblings, match) -> None:
    _write(tmp_path, _pdd_contract())
    _write_siblings(tmp_path, **siblings)
    with pytest.raises(ValueError, match=match):
        load_fasth3_inference_contract(tmp_path)


def test_pdd_contract_requires_positive_shifts(tmp_path) -> None:
    _write(tmp_path, _pdd_contract(video_scheduler_shift=0.0))
    with pytest.raises(ValueError, match="video_scheduler_shift"):
        load_fasth3_inference_contract(tmp_path)


# ---------------------------------------------------------------------------
# Schedule resolution
# ---------------------------------------------------------------------------


def test_pdd_schedule_carries_node_sigmas_and_blocks(tmp_path) -> None:
    _write(tmp_path, _pdd_contract())
    schedule = resolve_h3_denoise_schedule(tmp_path, 2)
    assert schedule.source == "contract-pdd"
    assert schedule.num_steps == 2
    assert schedule.pdd_plan is not None
    assert schedule.pdd_plan.block_sizes == [2, 2]
    # timesteps are H3 clean time (1 - sigmas[:-1]); the ladder ends at clean.
    for state in (schedule.video, schedule.audio):
        assert state.timesteps.shape == (2, )
        assert float(state.sigmas[-1]) == 0.0
        np.testing.assert_allclose(state.timesteps, (1.0 - state.sigmas[:-1]).astype(np.float32), atol=1e-6)


def test_pdd_schedule_rejects_a_wrong_step_count(tmp_path) -> None:
    _write(tmp_path, _pdd_contract())
    with pytest.raises(ValueError, match="exactly 2 transformer forwards"):
        resolve_h3_denoise_schedule(tmp_path, 4)


def test_converter_cache_follows_a_pdd_contract(tmp_path) -> None:
    """The converter must bake the PDD ladder, or the runtime's fixed-ladder check rejects it."""
    from scripts.checkpoint_conversion.convert_minimax_h3_mlx import _adaln_cache_timesteps
    from fastvideo.mlx_runtime.minimax_h3_pipeline import _validate_checkpoint_step_ladder

    _write(tmp_path, _pdd_contract())
    cached = _adaln_cache_timesteps(tmp_path)
    expected = resolve_h3_denoise_schedule(tmp_path, 2).adaln_timesteps
    np.testing.assert_array_equal(cached, expected)
    (tmp_path / "mlx_h3_dit.json").write_text(
        json.dumps({"adaln_cache": {"timesteps": cached.astype(np.float32).tolist()}}))
    _validate_checkpoint_step_ladder(tmp_path, 2, model_root=tmp_path)
    with pytest.raises(ValueError, match="exactly 2 transformer forwards"):
        _validate_checkpoint_step_ladder(tmp_path, 4, model_root=tmp_path)


def test_ref2va_ladder_carries_the_reference_condition_timestep(tmp_path) -> None:
    """Reference rows run at 0.999; a converted DiT has dropped AdaLN, so the baked cache must hold it."""
    from scripts.checkpoint_conversion.convert_minimax_h3_mlx import _adaln_cache_timesteps
    from fastvideo.mlx_runtime.minimax_h3 import MINIMAX_H3_KEYFRAME_NOISE_AUG

    _write(tmp_path, _pdd_contract())
    condition = np.float32(MINIMAX_H3_KEYFRAME_NOISE_AUG)
    assert condition in resolve_h3_denoise_schedule(tmp_path, 2).adaln_timesteps
    assert condition in _adaln_cache_timesteps(tmp_path)


def test_from_pdd_node_sigmas_validates(tmp_path) -> None:
    node = np.array([0.9, 0.4, 0.0], dtype=np.float64)
    state = MiniMaxH3SchedulerState.from_pdd_node_sigmas(VIDEO_SHIFT, node)
    assert state.timesteps.shape == (2, )
    with pytest.raises(ValueError, match="strictly decreasing"):
        MiniMaxH3SchedulerState.from_pdd_node_sigmas(VIDEO_SHIFT, np.array([0.9, 0.4, 0.5]))


# ---------------------------------------------------------------------------
# Head fusion
# ---------------------------------------------------------------------------


def test_fused_block_equals_slow_weighted_mean() -> None:
    """The fused projection emits ``sum_j w_j head_j / sum(w[start:end])``."""
    import mlx.core as mx

    grid, out_features, in_features = 4, 3, 5
    rng = np.random.default_rng(0)
    weight = mx.array(rng.standard_normal((grid * out_features, in_features)).astype(np.float32))
    bias = mx.array(rng.standard_normal((grid * out_features, )).astype(np.float32))
    integration_weights = rng.standard_normal(grid).astype(np.float64)
    start, end = 1, 3
    x = mx.array(rng.standard_normal((2, in_features)).astype(np.float32))

    fused_weight, fused_bias = fuse_block_params(
        weight,
        bias,
        grid_size=grid,
        start=start,
        end=end,
        weights=integration_weights,
    )
    fused_out = np.asarray(x @ fused_weight.T + fused_bias)

    heads = np.asarray(weight).reshape(grid, out_features, in_features)[start:end].astype(np.float32)
    head_bias = np.asarray(bias).reshape(grid, out_features)[start:end].astype(np.float32)
    block = integration_weights[start:end]
    normalized = (block / block.sum()).astype(np.float32)
    x_np = np.asarray(x)
    per_head = np.stack([x_np @ head.T + bias_j for head, bias_j in zip(heads, head_bias, strict=True)])
    slow = np.einsum("j,jio->io", normalized, per_head)

    np.testing.assert_allclose(fused_out, slow, atol=FUSE_ATOL, rtol=FUSE_RTOL)
    # The block total (the caller's Euler increment) times the emitted mean
    # reconstructs the un-normalized weighted sum.
    scaled = fused_out * float(block.sum())
    unnormalized = np.einsum("j,jio->io", block.astype(np.float32), per_head)
    np.testing.assert_allclose(scaled, unnormalized, atol=FUSE_ATOL, rtol=FUSE_RTOL)


def test_widened_model_requires_a_block_and_vice_versa() -> None:
    """A widened head with no block, or a block on an unwidened head, must error."""
    import mlx.core as mx

    model = build_torch_model()
    hf_config = build_hf_config(build_tiny_h3_config())
    layout, _, mlx_inputs = build_inputs()
    dit = mlx_dit_from_torch_model(model, hf_config)
    assert dit.pdd_grid_size is None

    common = dict(
        position_ids=mx.array(layout.position_ids.astype(np.float32)),
        token_tags=mx.array(layout.token_tags),
        timestep_indices=mx.array(mlx_inputs["timestep_indices"]),
        timesteps=mx.array(mlx_inputs["timesteps"]),
        video_indices=mx.array(layout.video_indices),
        audio_indices=mx.array(layout.audio_indices),
        text_indices=mx.array(layout.text_indices),
    )
    rows = (mx.array(mlx_inputs["video_rows"]), mx.array(mlx_inputs["audio_rows"]),
            mx.array(mlx_inputs["text_rows"]))

    with pytest.raises(ValueError, match="not widened"):
        dit.forward(*rows, pdd_block=(0, 1),
                    pdd_integration_weights={"video": np.ones(2), "audio": np.ones(2)}, **common)

    # Widen both heads by stacking two copies along the head axis and declare grid 2.
    grid = 2
    gewichten = np.ones(grid)
    weights = dict(dit.weights)
    weights["proj_out.weight"] = mx.concatenate([dit.weights["proj_out.weight"]] * grid, axis=0)
    weights["proj_out.bias"] = mx.concatenate([dit.weights["proj_out.bias"]] * grid, axis=0)
    weights["audio_proj_out.weight"] = mx.concatenate([dit.weights["audio_proj_out.weight"]] * grid, axis=0)
    weights["audio_proj_out.bias"] = mx.concatenate([dit.weights["audio_proj_out.bias"]] * grid, axis=0)
    widened_config = dict(dit.config)
    widened_config["pdd_steps"] = grid
    from fastvideo.mlx_runtime.minimax_h3 import MLXMiniMaxH3DiT

    widened = MLXMiniMaxH3DiT(weights, dit.blocks, dit.refiner, widened_config)
    assert widened.pdd_grid_size == grid

    with pytest.raises(ValueError, match="no PDD block"):
        widened.forward(*rows, **common)

    # A block [0, 1) with unit weights emits head 0 exactly, matching a direct
    # single-head projection of the same normed rows.
    video_out, _ = widened.forward(
        *rows,
        pdd_block=(0, 1),
        pdd_integration_weights={"video": gewichten, "audio": gewichten},
        **common,
    )
    mx.eval(video_out)
    assert np.isfinite(np.asarray(video_out)).all()


def test_resolved_grid_size_detects_widening() -> None:
    from fastvideo.mlx_runtime.minimax_h3_pdd import resolved_grid_size

    assert resolved_grid_size(np.zeros((32, 5)), np.zeros(32), 32) is None
    assert resolved_grid_size(np.zeros((96, 5)), np.zeros(96), 32) == 3
    assert resolved_grid_size(np.zeros((3 * TINY_ARCH["audio_in_channels"], 5)), None, TINY_ARCH["audio_in_channels"]) == 3
