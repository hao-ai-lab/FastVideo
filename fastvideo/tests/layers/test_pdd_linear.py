# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the PDD sampling primitives in ``fastvideo.layers.pdd``.

References are spelled out locally: the rational time shift on the
``[0, 0.999]`` base clock, its cancellation-free difference, the fused blocks
of an explicit fine-grid partition, and the fused-head forward as the
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
def test_shifted_noise_amount_matches_rational_time_shift(shift: float) -> None:
    base = torch.linspace(0.0, PDD_GRID_MAX_T, 17, dtype=torch.float64)
    torch.testing.assert_close(shifted_noise_amount(base, shift), _time_shift(base, shift), rtol=0.0, atol=1e-15)
    # max_t is a fixed point of every shift, so the first node has the same
    # noise level in every modality.
    fixed_point = float(shifted_noise_amount(torch.tensor(PDD_GRID_MAX_T, dtype=torch.float64), shift))
    assert fixed_point == pytest.approx(PDD_GRID_MAX_T, abs=1e-15)


@pytest.mark.parametrize("shift", [1.0, 0.25, 12.0])
def test_shifted_noise_delta_matches_direct_difference(shift: float) -> None:
    grid = pdd_fine_grid(256)
    direct = shifted_noise_amount(grid[1:], shift) - shifted_noise_amount(grid[:-1], shift)
    delta = shifted_noise_delta(grid[:-1], grid[1:], shift)
    torch.testing.assert_close(delta, direct, rtol=1e-12, atol=1e-15)
    assert bool((delta < 0).all())
    if shift == 1.0:
        assert torch.equal(delta, grid[1:] - grid[:-1])


def test_pdd_fine_grid_descending_float64_from_max_t_to_zero() -> None:
    grid = pdd_fine_grid(8)
    assert grid.dtype == torch.float64
    assert grid.shape == (9, )
    assert float(grid[0]) == PDD_GRID_MAX_T
    assert float(grid[-1]) == 0.0
    assert bool((grid[:-1] > grid[1:]).all())


@pytest.mark.parametrize(("step_indices", "block_sizes"), [
    ([0, 64, 128, 192, 256], [64, 64, 64, 64]),
    ([0, 2, 5, 8], [2, 3, 3]),
], ids=["uniform", "uneven"])
def test_build_pdd_sampling_plan_blocks_and_node_sigmas(step_indices: list[int], block_sizes: list[int]) -> None:
    schedules = {"video": PDDModalitySchedule(shift=12.0), "audio": PDDModalitySchedule(shift=3.0)}
    plan = build_pdd_sampling_plan(step_indices, schedules)

    grid_size = step_indices[-1]
    assert plan.pdd_steps == grid_size
    assert plan.num_steps == len(block_sizes)
    assert plan.block_sizes == block_sizes
    assert plan.block(1) == (step_indices[1], step_indices[2])
    # Node i of an N-interval grid sits at base time max_t * (1 - i / N).
    node_base = PDD_GRID_MAX_T * (1.0 - torch.tensor(step_indices, dtype=torch.float64) / grid_size)
    for name, schedule in schedules.items():
        node_sigmas = plan.node_sigmas[name]
        torch.testing.assert_close(node_sigmas, _time_shift(node_base, schedule.shift), rtol=0.0, atol=1e-12)
        assert float(node_sigmas[0]) == pytest.approx(PDD_GRID_MAX_T, abs=1e-15)
        assert float(node_sigmas[-1]) == 0.0
        # A fused block's total weight is exactly its node-sigma increment,
        # which is what makes one Euler step over the two nodes the block advance.
        for step in range(plan.num_steps):
            start, end = plan.block(step)
            total = plan.integration_weights[name][start:end].sum()
            torch.testing.assert_close(total, node_sigmas[step + 1] - node_sigmas[step], rtol=1e-12, atol=1e-15)


def test_build_pdd_sampling_plan_mismatched_max_t() -> None:
    schedules = {"a": PDDModalitySchedule(1.0), "b": PDDModalitySchedule(1.0, max_t=1.0)}
    with pytest.raises(ValueError, match="share max_t"):
        build_pdd_sampling_plan([0, 4, 8], schedules)


def test_build_pdd_sampling_plan_grid32_eight_blocks_feed_h3_schedulers() -> None:
    """The exported contract (Grid32, blocks of 4, shifts 12/3) as the stage builds it."""
    plan = build_pdd_sampling_plan(GRID32_BLOCKS8, {
        "video": PDDModalitySchedule(shift=12.0),
        "audio": PDDModalitySchedule(shift=3.0)
    })
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


def test_pdd_replicated_linear_head_major_and_state_dict_compatible() -> None:
    linear = PDDReplicatedLinear(5, 3, grid_size=4, params_dtype=torch.float32)
    assert linear.weight.shape == (12, 5) and linear.bias.shape == (12, )
    assert linear.head_output_size == 3 and linear.grid_size == 4
    # Fusion state is transient: the state dict is a plain linear's.
    assert set(linear.state_dict()) == {"weight", "bias"}


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


def test_forward_fused_block_weighted_mean_of_materialized_heads() -> None:
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


def test_fused_params_bf16_heads_accumulate_in_decoding_precision() -> None:
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


def test_fuse_nested_contexts_restore_outer_state() -> None:
    linear = _randomized(PDDReplicatedLinear(3, 2, grid_size=4, params_dtype=torch.float32))
    weights = PDDModalitySchedule(shift=1.0).integration_weights(pdd_fine_grid(4))
    with linear.fuse(0, 4, weights, torch.float32):
        outer = linear._fusion_state
        with linear.fuse(1, 3, weights, torch.float32):
            assert linear._fusion_state[:2] == (1, 3)
        assert linear._fusion_state is outer
    assert linear._fusion_state is None
