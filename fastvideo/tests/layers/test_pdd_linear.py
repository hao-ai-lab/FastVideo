# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the PDD sampling primitives in ``fastvideo.layers.pdd``.

References are spelled out locally: the rational time shift on the
``[0, 0.999]`` base clock, its cancellation-free difference, the balanced and
explicit fine-grid partitions, and the fused-head forward as the
integration-weighted mean of the materialized heads.
"""

from __future__ import annotations

import pytest
import torch

from fastvideo.layers.pdd import (
    PDD_GRID_MAX_T,
    PDDModalitySchedule,
    PDDReplicatedLinear,
    build_pdd_sampling_plan,
    fuse_pdd_heads,
    pdd_fine_grid,
    pdd_step_indices,
    shifted_noise_amount,
    shifted_noise_delta,
)
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

# The OmniRef PDD-8 export: a 32-interval grid sampled in eight 4-head blocks.
GRID32_BLOCKS8 = list(range(0, 33, 4))


def _time_shift(value: torch.Tensor, shift: float, max_t: float = PDD_GRID_MAX_T) -> torch.Tensor:
    """``s * t * M / (t * (s - 1) + M)`` in float64."""
    value = value.to(torch.float64)
    return value * shift * max_t / (value * (shift - 1.0) + max_t)


@pytest.mark.parametrize("shift", [1.0, 0.25, 3.0, 12.0])
def test_shifted_noise_amount_is_the_rational_time_shift(shift: float) -> None:
    base = torch.linspace(0.0, PDD_GRID_MAX_T, 17, dtype=torch.float64)
    torch.testing.assert_close(shifted_noise_amount(base, shift), _time_shift(base, shift), rtol=0.0, atol=1e-15)
    # max_t is a fixed point of every shift, so the first node has the same
    # noise level in every modality.
    fixed_point = float(shifted_noise_amount(torch.tensor(PDD_GRID_MAX_T, dtype=torch.float64), shift))
    assert fixed_point == pytest.approx(PDD_GRID_MAX_T, abs=1e-15)


@pytest.mark.parametrize("shift", [1.0, 0.25, 12.0])
def test_shifted_noise_delta_matches_the_direct_difference(shift: float) -> None:
    grid = pdd_fine_grid(256)
    direct = shifted_noise_amount(grid[1:], shift) - shifted_noise_amount(grid[:-1], shift)
    delta = shifted_noise_delta(grid[:-1], grid[1:], shift)
    torch.testing.assert_close(delta, direct, rtol=1e-12, atol=1e-15)
    assert bool((delta < 0).all())
    if shift == 1.0:
        assert torch.equal(delta, grid[1:] - grid[:-1])


def test_modality_schedule_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="shift must be positive"):
        PDDModalitySchedule(shift=0.0)
    with pytest.raises(ValueError, match="max_t"):
        PDDModalitySchedule(shift=1.0, max_t=1.5)


def test_fine_grid_is_descending_float64_from_max_t_to_zero() -> None:
    grid = pdd_fine_grid(8)
    assert grid.dtype == torch.float64
    assert grid.shape == (9, )
    assert float(grid[0]) == PDD_GRID_MAX_T
    assert float(grid[-1]) == 0.0
    assert bool((grid[:-1] > grid[1:]).all())
    with pytest.raises(ValueError, match=">= 2"):
        pdd_fine_grid(1)


def test_step_indices_balanced_explicit_and_rejected() -> None:
    assert pdd_step_indices(256, 4).tolist() == [0, 64, 128, 192, 256]
    assert pdd_step_indices(10, 3).tolist() == [0, 3, 6, 10]
    assert pdd_step_indices(32, 8).tolist() == GRID32_BLOCKS8
    assert pdd_step_indices(128, 3, indices=[0, 32, 80, 128]).tolist() == [0, 32, 80, 128]
    with pytest.raises(ValueError, match="Cannot select"):
        pdd_step_indices(4, 8)
    with pytest.raises(ValueError, match="start at 0 and end at 128"):
        pdd_step_indices(128, 3, indices=[0, 32, 80, 120])
    with pytest.raises(ValueError, match="strictly increasing"):
        pdd_step_indices(128, 3, indices=[0, 80, 32, 128])
    with pytest.raises(ValueError, match="must have shape"):
        pdd_step_indices(128, 3, indices=[0, 128])
    with pytest.raises(TypeError, match="integer dtype"):
        pdd_step_indices(8, 2, indices=[0.0, 4.0, 8.0])


def test_sampling_plan_blocks_and_node_sigmas() -> None:
    schedules = {"video": PDDModalitySchedule(shift=12.0), "audio": PDDModalitySchedule(shift=3.0)}
    plan = build_pdd_sampling_plan(256, 4, schedules)

    assert plan.pdd_steps == 256
    assert plan.num_steps == 4
    assert plan.block_sizes == [64, 64, 64, 64]
    assert plan.block(1) == (64, 128)
    for name, schedule in schedules.items():
        node_sigmas = plan.node_sigmas[name]
        torch.testing.assert_close(node_sigmas, schedule.sigma(plan.fine_grid[plan.indices]), rtol=0.0, atol=0.0)
        assert float(node_sigmas[0]) == pytest.approx(PDD_GRID_MAX_T, abs=1e-15)
        assert float(node_sigmas[-1]) == 0.0
        # A fused block's total weight is exactly its node-sigma increment,
        # which is what makes one Euler step over the two nodes the block advance.
        for step in range(plan.num_steps):
            start, end = plan.block(step)
            total = plan.integration_weights[name][start:end].sum()
            torch.testing.assert_close(total, node_sigmas[step + 1] - node_sigmas[step], rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError, match="step must be in"):
        plan.block(4)
    assert build_pdd_sampling_plan(8, 3, schedules, indices=[0, 2, 5, 8]).indices.tolist() == [0, 2, 5, 8]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_pdd_sampling_plan(8, 3, schedules, indices=[0, 5, 2, 8])
    with pytest.raises(ValueError, match="share max_t"):
        build_pdd_sampling_plan(8, 2, {"a": PDDModalitySchedule(1.0), "b": PDDModalitySchedule(1.0, max_t=1.0)})


def test_grid32_eight_block_plan_feeds_the_h3_schedulers() -> None:
    """The exported contract (Grid32, blocks of 4, shifts 12/3) as the stage builds it."""
    plan = build_pdd_sampling_plan(32, 8, {
        "video": PDDModalitySchedule(shift=12.0),
        "audio": PDDModalitySchedule(shift=3.0)
    },
                                   indices=GRID32_BLOCKS8)
    assert plan.block_sizes == [4] * 8
    for name, shift in (("video", 12.0), ("audio", 3.0)):
        expected = _time_shift(pdd_fine_grid(32)[GRID32_BLOCKS8], shift)
        torch.testing.assert_close(plan.node_sigmas[name], expected, rtol=0.0, atol=1e-15)
        scheduler = MiniMaxH3Scheduler(shift=shift)
        scheduler.set_timesteps(sigmas=plan.node_sigmas[name].to(torch.float32))
        assert scheduler.num_inference_steps == 8
        torch.testing.assert_close(scheduler.timesteps, 1.0 - plan.node_sigmas[name][:-1].to(torch.float32))


def _randomized(linear: PDDReplicatedLinear) -> PDDReplicatedLinear:
    # ReplicatedLinear storage is uninitialized until a checkpoint loads.
    with torch.no_grad():
        linear.weight.normal_()
        linear.bias.normal_()
    return linear


def test_pdd_linear_is_head_major_and_state_dict_compatible() -> None:
    linear = PDDReplicatedLinear(5, 3, grid_size=4, params_dtype=torch.float32)
    assert linear.weight.shape == (12, 5) and linear.bias.shape == (12, )
    assert linear.head_output_size == 3 and linear.grid_size == 4
    # Fusion state is transient: the state dict is a plain linear's.
    assert set(linear.state_dict()) == {"weight", "bias"}
    with pytest.raises(ValueError, match="grid_size must be an int >= 2"):
        PDDReplicatedLinear(3, 2, grid_size=1)


def test_fused_params_use_normalized_integration_weights() -> None:
    torch.manual_seed(2)
    grid, channels, in_features = 5, 3, 7
    linear = _randomized(PDDReplicatedLinear(in_features, channels, grid_size=grid, params_dtype=torch.float32))
    weights = PDDModalitySchedule(shift=0.25).integration_weights(pdd_fine_grid(grid))
    start, end = 1, 5

    actual_weight, actual_bias = linear._fused_params(start, end, weights, torch.float32)
    alpha = weights[start:end]
    alpha = (alpha / alpha.sum()).float()
    expected_weight = torch.einsum("n,nci->ci", alpha, linear.weight.reshape(grid, channels, in_features)[start:end])
    expected_bias = torch.einsum("n,nc->c", alpha, linear.bias.reshape(grid, channels)[start:end])
    torch.testing.assert_close(actual_weight, expected_weight, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_bias, expected_bias, rtol=1e-6, atol=1e-6)


def test_fused_forward_is_the_weighted_mean_of_materialized_heads() -> None:
    torch.manual_seed(3)
    grid, out_features, in_features = 6, 4, 5
    linear = _randomized(PDDReplicatedLinear(in_features, out_features, grid_size=grid, params_dtype=torch.float32))
    x = torch.randn(2, 3, in_features)
    weights = PDDModalitySchedule(shift=3.0).integration_weights(pdd_fine_grid(grid))

    heads, extra_bias = linear(x)
    assert extra_bias is None
    heads = heads.unflatten(-1, (grid, out_features))
    for start, end in ((0, grid), (2, 5), (4, 5)):
        alpha = (weights[start:end] / weights[start:end].sum()).float()
        expected = torch.einsum("n,btnc->btc", alpha, heads[..., start:end, :])
        with linear.fuse(start, end, weights, torch.float32):
            fused, fused_extra = linear(x)
        assert fused_extra is None
        torch.testing.assert_close(fused, expected, rtol=1e-5, atol=1e-5)
    # Leaving the context restores the materialized-head forward.
    torch.testing.assert_close(linear(x)[0].unflatten(-1, (grid, out_features)), heads)


def test_fused_bf16_heads_accumulate_in_the_decoding_precision() -> None:
    torch.manual_seed(5)
    linear = _randomized(PDDReplicatedLinear(8, 3, grid_size=4, params_dtype=torch.bfloat16))
    weights = PDDModalitySchedule(shift=12.0).integration_weights(pdd_fine_grid(4))
    fused_weight, fused_bias = linear._fused_params(0, 4, weights, torch.float32)
    alpha = (weights / weights.sum()).float()
    expected = torch.einsum("n,nci->ci", alpha, linear.weight.float().reshape(4, 3, 8)).to(torch.bfloat16)
    assert fused_weight.dtype == torch.bfloat16 and fused_bias is not None and fused_bias.dtype == torch.bfloat16
    assert torch.equal(fused_weight, expected)


def test_fuse_pdd_heads_fuses_every_modality_head_at_once() -> None:
    torch.manual_seed(4)
    grid = 4
    linears = {
        "video": _randomized(PDDReplicatedLinear(3, 2, grid_size=grid, params_dtype=torch.float32)),
        "audio": _randomized(PDDReplicatedLinear(3, 1, grid_size=grid, params_dtype=torch.float32)),
    }
    weights = {
        "video": PDDModalitySchedule(shift=12.0).integration_weights(pdd_fine_grid(grid)),
        "audio": PDDModalitySchedule(shift=3.0).integration_weights(pdd_fine_grid(grid)),
    }
    x = torch.randn(2, 3)
    with fuse_pdd_heads(linears, 1, 3, weights, torch.float32):
        fused = {name: linear(x)[0] for name, linear in linears.items()}
    for name, linear in linears.items():
        with linear.fuse(1, 3, weights[name], torch.float32):
            torch.testing.assert_close(fused[name], linear(x)[0])
        assert linear._fusion_state is None
    with pytest.raises(ValueError, match="missing \\['audio'\\]"):
        with fuse_pdd_heads(linears, 1, 3, {"video": weights["video"]}, torch.float32):
            pass


def test_fusion_rejects_invalid_blocks_and_restores_nested_state() -> None:
    linear = _randomized(PDDReplicatedLinear(3, 2, grid_size=4, params_dtype=torch.float32))
    weights = PDDModalitySchedule(shift=1.0).integration_weights(pdd_fine_grid(4))
    x = torch.randn(1, 3)
    with pytest.raises(ValueError, match="0 <= start < end <= grid_size"):
        with linear.fuse(2, 2, weights, torch.float32):
            linear(x)
    with pytest.raises(ValueError, match="fusion weights must have shape"):
        with linear.fuse(0, 4, weights[:2], torch.float32):
            linear(x)
    with pytest.raises(TypeError, match="floating point"):
        with linear.fuse(0, 4, torch.ones(4, dtype=torch.long), torch.float32):
            linear(x)
    with pytest.raises(ValueError, match="finite and non-zero"):
        with linear.fuse(0, 2, torch.zeros(4, dtype=torch.float64), torch.float32):
            linear(x)

    with linear.fuse(0, 4, weights, torch.float32):
        outer = linear._fusion_state
        with linear.fuse(1, 3, weights, torch.float32):
            assert linear._fusion_state[:2] == (1, 3)
        assert linear._fusion_state is outer
    assert linear._fusion_state is None
