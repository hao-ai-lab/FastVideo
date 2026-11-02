# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the packed-latent PDD core (``pdd_core.py``).

Ports the block-integration, Runge-Kutta teacher-target, x0-inversion, and
carry tests of FastGen's ``tests/test_pddmodel.py`` /
``test_pddmdmodel.py`` onto the column-slice representation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fastvideo.layers.pdd import PDDModalitySchedule, pdd_fine_grid
from fastvideo.train.methods.trajectory_carry import (
    TrajectoryCarry,
    parse_carry_slots,
    stagger_offset,
    stagger_rank_and_groups,
    validate_carry_coverage,
)
from fastvideo.train.methods.trajectory_matching.pdd_core import (
    PDDModalityBlock,
    advance_block,
    advance_fused,
    aligned_block_starts,
    block_mask,
    carried_step_count,
    head_losses,
    sample_supervision_index,
    teacher_block_target,
    validate_block_indices,
    x0_from_local_head,
    x0_from_points,
)

_VIDEO = slice(0, 6)
_AUDIO = slice(6, 9)
_NUMEL = 9


def _blocks(video_shift: float = 12.0, audio_shift: float = 3.0) -> tuple[PDDModalityBlock, ...]:
    return (
        PDDModalityBlock("video", _VIDEO, PDDModalitySchedule(shift=video_shift)),
        PDDModalityBlock("audio", _AUDIO, PDDModalitySchedule(shift=audio_shift)),
    )


def _forward_process(x0: torch.Tensor, eps: torch.Tensor, base: torch.Tensor, blocks) -> torch.Tensor:
    x_t = x0.clone()
    for block in blocks:
        sigma = block.schedule.sigma(base).to(x0.dtype)[:, None]
        x_t[:, block.columns] = (1.0 - sigma) * x0[:, block.columns] + sigma * eps[:, block.columns]
    return x_t


# ----------------------------------------------------------------------------
# Block integration
# ----------------------------------------------------------------------------


def test_advance_block_matches_fused_mean_reduction() -> None:
    torch.manual_seed(0)
    grid = 8
    fine_grid = pdd_fine_grid(grid)
    blocks = _blocks()
    state = torch.randn(2, _NUMEL, dtype=torch.float64)
    pred = torch.randn(2, grid, _NUMEL, dtype=torch.float64)
    step_idx = torch.tensor([1, 3])
    end_idx = torch.tensor([4, 8])

    advanced = advance_block(state, pred, step_idx, end_idx, fine_grid, blocks)

    expected = state.clone()
    for row in range(2):
        start, end = int(step_idx[row]), int(end_idx[row])
        for block in blocks:
            weights = block.schedule.integration_weights(fine_grid)[start:end]
            expected[row, block.columns] += torch.einsum("n,nk->k", weights, pred[row, start:end, block.columns])
    torch.testing.assert_close(advanced, expected, rtol=1e-12, atol=1e-12)

    # A shared block advanced from its normalized mean reproduces the same state.
    for row in range(2):
        start, end = int(step_idx[row]), int(end_idx[row])
        mean = torch.zeros(1, _NUMEL, dtype=torch.float64)
        for block in blocks:
            weights = block.schedule.integration_weights(fine_grid)[start:end]
            alpha = weights / weights.sum()
            mean[0, block.columns] = torch.einsum("n,nk->k", alpha, pred[row, start:end, block.columns])
        fused = advance_fused(state[row:row + 1], mean, start, end, fine_grid, blocks)
        torch.testing.assert_close(fused, advanced[row:row + 1], rtol=1e-12, atol=1e-12)


def test_advance_block_precision_scoping_and_empty_rows() -> None:
    grid = 4
    fine_grid = pdd_fine_grid(grid)
    blocks = _blocks()
    state = torch.randn(2, _NUMEL, dtype=torch.bfloat16)
    pred = torch.randn(2, grid, _NUMEL, dtype=torch.bfloat16)
    step_idx = torch.tensor([0, 2])

    advanced = advance_block(state, pred, step_idx, torch.tensor([2, 2]), fine_grid, blocks, allow_empty=True,
                             precision_decoding=torch.float32)
    assert advanced.dtype == torch.bfloat16
    # The empty second row is an identity update.
    assert torch.equal(advanced[1], state[1])
    reference = advance_block(state[:1].float(), pred[:1].float(), step_idx[:1], torch.tensor([2]), fine_grid, blocks)
    torch.testing.assert_close(advanced[:1].float(), reference, rtol=2e-2, atol=2e-2)

    with pytest.raises(ValueError, match="0 <= step_idx < end_idx"):
        advance_block(state, pred, step_idx, torch.tensor([2, 2]), fine_grid, blocks)
    with pytest.raises(TypeError, match="share a dtype"):
        advance_block(state.float(), pred, step_idx, torch.tensor([2, 4]), fine_grid, blocks)
    with pytest.raises(ValueError, match="\\[batch, heads, numel\\]"):
        advance_block(state, pred[:, :, :4], step_idx, torch.tensor([2, 4]), fine_grid, blocks)


def test_advance_block_accepts_a_head_window() -> None:
    torch.manual_seed(5)
    grid = 8
    fine_grid = pdd_fine_grid(grid)
    blocks = _blocks()
    state = torch.randn(2, _NUMEL, dtype=torch.float64)
    pred = torch.randn(2, grid, _NUMEL, dtype=torch.float64)
    step_idx = torch.tensor([2, 3])
    end_idx = torch.tensor([5, 6])
    full = advance_block(state, pred, step_idx, end_idx, fine_grid, blocks)
    windowed = advance_block(state, pred[:, 2:6], step_idx, end_idx, fine_grid, blocks, head_offset=2)
    torch.testing.assert_close(windowed, full, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="2 <= step_idx < end_idx <= 6"):
        advance_block(state, pred[:, 2:6], step_idx, torch.tensor([5, 7]), fine_grid, blocks, head_offset=2)
    with pytest.raises(ValueError, match="fall outside"):
        advance_block(state, pred[:, 2:6], step_idx, end_idx, fine_grid, blocks, head_offset=6)


def test_block_masks_and_index_validation() -> None:
    mask = block_mask(torch.tensor([0, 1]), torch.tensor([2, 4]), 4)
    assert mask.dtype == torch.bool and mask.shape == (2, 4)
    assert mask.tolist() == [[True, True, False, False], [False, True, True, True]]
    with pytest.raises(ValueError, match="must be a torch.long tensor of shape"):
        validate_block_indices(torch.tensor([0, 1]), torch.tensor([2]), 4)
    with pytest.raises(TypeError, match="torch.long"):
        validate_block_indices(torch.tensor([0.0]), torch.tensor([2]), 4)
    with pytest.raises(ValueError, match="<= 4"):
        validate_block_indices(torch.tensor([0]), torch.tensor([5]), 4)
    validate_block_indices(torch.tensor([2]), torch.tensor([2]), 4, allow_empty=True)


# ----------------------------------------------------------------------------
# Teacher targets
# ----------------------------------------------------------------------------


def _reference_sigma_velocity(x: torch.Tensor, base: torch.Tensor, blocks) -> torch.Tensor:
    """A linear toy teacher in native ``eps - x0`` space: ``0.125 x + sigma_m``."""
    out = 0.125 * x
    for block in blocks:
        out[:, block.columns] = out[:, block.columns] + block.schedule.sigma(base).to(x.dtype)[:, None]
    return out


@pytest.mark.parametrize("rk_type", ["euler", "midpoint", "heun", "kutta4"])
@pytest.mark.parametrize("shift", [1.0, 12.0])
def test_rk_stages_use_native_scaling_on_the_shared_clock(rk_type: str, shift: float) -> None:
    """Stages sit at primary-clock times with native per-modality increments.

    Samples whose interval ends on the clean endpoint blend to the solver's
    terminal fallback stage instead of supervising from a ``t = 0`` teacher
    evaluation.
    """
    torch.manual_seed(1)
    blocks = _blocks(video_shift=shift, audio_shift=3.0)
    primary = blocks[0].schedule
    fine_grid = torch.tensor([0.999, 0.72, 0.41, 0.18, 0.0], dtype=torch.float64)
    x = torch.randn(2, _NUMEL, dtype=torch.float64)
    step_idx = torch.tensor([1, 3])
    base = fine_grid[step_idx]
    base_next = fine_grid[step_idx + 1]
    delta_sigma = primary.sigma_delta(base, base_next)

    def teacher(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return _reference_sigma_velocity(state, time, blocks)

    def stage(k: torch.Tensor, fraction: float) -> torch.Tensor:
        if fraction == 1.0:
            stage_time = base_next
        else:
            stage_time = primary.unwarp(primary.sigma(base) + fraction * delta_sigma)
        stage_state = x.clone()
        for block in blocks:
            stage_state[:, block.columns] += k[:, block.columns] * block.schedule.sigma_delta(base, stage_time)[:, None]
        return teacher(stage_state, stage_time)

    terminal = (base_next <= fine_grid[-1])[:, None].expand_as(x)
    k1 = teacher(x, base)
    if rk_type == "euler":
        expected = k1
    elif rk_type == "midpoint":
        expected = stage(k1, 0.5)
    elif rk_type == "heun":
        k2 = stage(k1, 1.0)
        expected = torch.where(terminal, k1, 0.5 * k1 + 0.5 * k2)
    else:
        k2 = stage(k1, 0.5)
        k3 = stage(k2, 0.5)
        k4 = stage(k3, 1.0)
        expected = torch.where(terminal, k2, (k1 + 2 * k2 + 2 * k3 + k4) / 6)

    actual = teacher_block_target(x, step_idx, fine_grid, blocks, solver=rk_type, predict_teacher=teacher)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="RK type"):
        teacher_block_target(x, step_idx, fine_grid, blocks, solver="rk9", predict_teacher=teacher)


def test_euler_target_freezes_the_native_direction_for_one_interval() -> None:
    """One Euler call yields the direction that advances the state exactly."""
    torch.manual_seed(2)
    blocks = _blocks()
    fine_grid = pdd_fine_grid(3)
    x0 = torch.randn(2, _NUMEL, dtype=torch.float64)
    eps = torch.randn(2, _NUMEL, dtype=torch.float64)
    step_idx = torch.tensor([0, 1])
    x_t = _forward_process(x0, eps, fine_grid[step_idx], blocks)

    def exact_teacher(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del state, time
        return eps - x0

    target = teacher_block_target(x_t, step_idx, fine_grid, blocks, solver="euler", predict_teacher=exact_teacher)
    pred = target[:, None].expand(-1, 3, -1)
    advanced = advance_block(x_t, pred, step_idx, step_idx + 1, fine_grid, blocks)
    torch.testing.assert_close(advanced, _forward_process(x0, eps, fine_grid[step_idx + 1], blocks), rtol=1e-12,
                               atol=1e-12)


# ----------------------------------------------------------------------------
# x0 inversions and losses
# ----------------------------------------------------------------------------


def test_x0_inversions_recover_clean_data() -> None:
    torch.manual_seed(3)
    blocks = _blocks()
    fine_grid = pdd_fine_grid(4)
    x0 = torch.randn(2, _NUMEL, dtype=torch.float64)
    eps = torch.randn(2, _NUMEL, dtype=torch.float64)
    start_idx = torch.tensor([0, 1])
    end_idx = torch.tensor([2, 4])
    x_start = _forward_process(x0, eps, fine_grid[start_idx], blocks)
    x_end = _forward_process(x0, eps, fine_grid[end_idx], blocks)

    torch.testing.assert_close(x0_from_points(x_start, x_end, start_idx, end_idx, fine_grid, blocks), x0, rtol=1e-11,
                               atol=1e-11)
    torch.testing.assert_close(x0_from_local_head(x_start, start_idx, fine_grid, blocks, eps - x0), x0, rtol=1e-11,
                               atol=1e-11)
    # The clean endpoint is returned exactly.
    assert torch.equal(x0_from_points(x_start[1:], x_end[1:], start_idx[1:], end_idx[1:], fine_grid, blocks), x_end[1:])
    # Inversions preserve the caller's dtype boundary.
    assert x0_from_points(x_start.to(torch.bfloat16), x_end.to(torch.bfloat16), start_idx, end_idx, fine_grid,
                          blocks).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="0 <= step_idx < end_idx"):
        x0_from_points(x_start, x_start, start_idx, start_idx, fine_grid, blocks)


def test_head_losses_are_per_modality_and_promoted_to_fp32() -> None:
    blocks = _blocks()
    prediction = torch.ones(2, _NUMEL, dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros(2, _NUMEL, dtype=torch.float32)
    target[:, _AUDIO] = 2.0
    losses = head_losses(prediction, target, blocks)
    assert set(losses) == {"video", "audio"}
    assert losses["video"].dtype == torch.float32
    torch.testing.assert_close(losses["video"], torch.tensor(1.0))
    torch.testing.assert_close(losses["audio"], torch.tensor(1.0))
    sum(losses.values()).backward()
    assert prediction.grad is not None


# ----------------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------------


def test_supervision_index_stays_inside_its_window() -> None:
    generator = torch.Generator().manual_seed(0)
    step_idx = torch.full((64, ), 2, dtype=torch.long)
    sampled = sample_supervision_index(step_idx, pdd_steps=8, block_size_max=3, generator=generator)
    assert set(sampled.tolist()) <= {2, 3, 4}
    assert len(set(sampled.tolist())) > 1
    # An explicit segment end confines the head to that segment.
    sampled = sample_supervision_index(step_idx, pdd_steps=8, block_size_max=8, block_end_idx=torch.full_like(
        step_idx, 4), generator=generator)
    assert set(sampled.tolist()) <= {2, 3}
    # The window is clamped to the grid.
    sampled = sample_supervision_index(torch.tensor([7]), pdd_steps=8, block_size_max=4, generator=generator)
    assert sampled.tolist() == [7]


def test_aligned_block_starts_include_partial_final_stride() -> None:
    generator = torch.Generator().manual_seed(0)
    starts = aligned_block_starts(512, pdd_steps=10, block_size_min=4, device=torch.device("cpu"), generator=generator)
    assert set(starts.tolist()) == {0, 4, 8}
    assert carried_step_count(10, 4) == 3
    assert carried_step_count(256, 32) == 8


def test_trajectory_carry_slots_and_stagger() -> None:
    carry = TrajectoryCarry(2)
    assert [carry.next_slot() for _ in range(5)] == [0, 1, 0, 1, 0]
    carry.set(1, state=torch.ones(1, 2, requires_grad=True), rung=3, raw_batch={"prompt": "p"})
    assert carry.slots[1]["rung"] == 3
    assert not carry.slots[1]["state"].requires_grad
    carry.clear(1)
    assert carry.slots[1] is None
    assert [stagger_offset(stagger_rank=0, num_slots=2, slot=slot, num_steps=4) for slot in range(2)] == [0, 1]
    assert stagger_offset(stagger_rank=3, num_slots=2, slot=1, num_steps=4) == 3
    validate_carry_coverage(streams=1, num_steps=8, interval=5, owner="test")
    with pytest.raises(ValueError, match="test cannot cover every student rung"):
        validate_carry_coverage(streams=1, num_steps=8, interval=4, owner="test")
    with pytest.raises(ValueError, match="num_slots must be a positive int"):
        TrajectoryCarry(0)


def test_carry_slots_and_stagger_follow_the_trainer_cadence() -> None:
    loop = SimpleNamespace(gradient_accumulation_steps=4)
    assert parse_carry_slots({}, loop) == 4
    assert parse_carry_slots({"rollout_carry_slots": 4}, loop) == 4
    with pytest.raises(ValueError, match="rollout_carry_slots must equal"):
        parse_carry_slots({"rollout_carry_slots": 2}, loop)
    with pytest.raises(ValueError, match="pdd_carry_slots must be positive"):
        parse_carry_slots({"pdd_carry_slots": 0}, SimpleNamespace(gradient_accumulation_steps=0), key="pdd_carry_slots")
    # Native-shape data keeps only the slot staggering.
    exact = SimpleNamespace(distributed=SimpleNamespace(sp_size=1), data=SimpleNamespace(native_shape_bucketing=False))
    native = SimpleNamespace(distributed=SimpleNamespace(sp_size=1), data=SimpleNamespace(native_shape_bucketing=True))
    assert stagger_rank_and_groups(exact, (3, 4)) == (3, 4)
    assert stagger_rank_and_groups(native, (3, 4)) == (0, 1)
    assert stagger_rank_and_groups(exact) == (0, 1)
