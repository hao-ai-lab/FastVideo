# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 training-free token reduction (antirez/h3.c ``--token-reduction``).

CPU checks cover option parsing, the pair row maps, the pool/restore
arithmetic and the denoising-stage helpers. The GPU checks run a tiny real
transformer: a one-token-wide video has no pairs, so the reduced path must be
bit-identical to the plain forward, which exercises every reduced input
(positions, modulation rows, the attention metadata swap).
"""

from __future__ import annotations

import pytest
import torch

from fastvideo import envs
from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig, MiniMaxH3Config
from fastvideo.forward_context import get_forward_context, set_forward_context
from fastvideo.models.dits.minimax_h3_token_reduction import (
    MiniMaxH3TokenReductionSchedule,
    build_token_reduction,
    expand_token_pairs,
    pool_token_pairs,
)
from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence, build_row_timesteps
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
    _h3_token_reduced_segments,
    _h3_token_reduction_plan,
)

_PATCH = (1, 2, 2)


def _schedule(**options) -> MiniMaxH3TokenReductionSchedule:
    schedule = MiniMaxH3TokenReductionSchedule.from_options({"enabled": True, **options})
    assert schedule is not None
    return schedule


# ----------------------------------------------------------------------------
# Options
# ----------------------------------------------------------------------------


def test_schedule_is_off_unless_enabled() -> None:
    assert MiniMaxH3TokenReductionSchedule.from_options({}) is None
    assert MiniMaxH3TokenReductionSchedule.from_options({"enabled": False, "blocks": [1, 2]}) is None


def test_schedule_defaults_follow_h3c_share_of_eight_forwards() -> None:
    schedule = _schedule()
    assert (schedule.begin, schedule.end, schedule.early_steps, schedule.early_end) == (4, 30, 4, 40)
    assert [schedule.end_for_step(step) for step in range(8)] == [40, 40, 40, 40, 30, 30, 30, 30]


def test_schedule_skips_early_interval_when_base_reaches_block_40() -> None:
    schedule = _schedule(blocks="4:45")
    assert (schedule.early_steps, schedule.end_for_step(0)) == (0, 45)


def test_schedule_accepts_strings_and_disabling_early() -> None:
    schedule = _schedule(blocks="2:20", early="0:0", update_scale=0.5)
    assert (schedule.begin, schedule.end, schedule.end_for_step(0), schedule.update_scale) == (2, 20, 20, 0.5)


@pytest.mark.parametrize("options,match", [
    ({"enabled": "yes"}, "enabled must be a boolean"),
    ({"enabled": True, "blocks": [30, 4]}, "BEGIN < END"),
    ({"enabled": True, "blocks": "4"}, "two integers"),
    ({"enabled": True, "early": [2, 20]}, "must exceed blocks END"),
    ({"enabled": True, "update_scale": 3.0}, r"\[0, 2\]"),
    ({"enabled": True, "update_scale": True}, r"\[0, 2\]"),
    ({"enabled": True, "begin": 4}, "keys from"),
])
def test_schedule_rejects_bad_options(options, match) -> None:
    with pytest.raises(ValueError, match=match):
        MiniMaxH3TokenReductionSchedule.from_options(options)


# ----------------------------------------------------------------------------
# Row maps and arithmetic
# ----------------------------------------------------------------------------


def _positions(sequence_length: int) -> torch.Tensor:
    return torch.arange(sequence_length * 3, dtype=torch.float64).reshape(sequence_length, 3)


def test_pairs_cover_rows_and_leave_odd_column_alone() -> None:
    # Two prefix rows, then a 1x2x3 video: columns (0, 1) pair, column 2 is alone.
    plan = build_token_reduction(_positions(8), 2, (1, 2, 3), _schedule())
    assert plan.first.tolist() == [0, 1, 2, 4, 5, 7]
    assert plan.second.tolist() == [0, 1, 3, 4, 6, 7]
    assert plan.parents.tolist() == [0, 1, 2, 2, 3, 4, 4, 5]
    assert plan.pair_reduced_rows.tolist() == [2, 4]
    assert plan.pair_rows.tolist() == [2, 3, 5, 6]
    assert (plan.sequence_length, plan.reduced_length) == (8, 6)


def test_pair_positions_are_the_mean_and_unpaired_positions_are_exact() -> None:
    positions = _positions(8)
    plan = build_token_reduction(positions, 2, (1, 2, 3), _schedule())
    torch.testing.assert_close(plan.position_ids[2], (positions[2] + positions[3]) / 2, rtol=0, atol=0)
    for reduced_row, full_row in ((0, 0), (1, 1), (3, 4), (5, 7)):
        assert torch.equal(plan.position_ids[reduced_row], positions[full_row])


def test_plan_requires_the_video_to_end_the_sequence() -> None:
    with pytest.raises(ValueError, match="end the packed sequence"):
        build_token_reduction(_positions(9), 2, (1, 2, 3), _schedule())


def test_h3c_worked_example() -> None:
    # One prefix row (10), then a 1x1x4 video row 1, 3, 5, 9.
    plan = build_token_reduction(_positions(5), 1, (1, 1, 4), _schedule())
    hidden = torch.tensor([10.0, 1.0, 3.0, 5.0, 9.0]).view(1, 5, 1)
    pooled = pool_token_pairs(hidden, plan)
    assert pooled.flatten().tolist() == [10.0, 2.0, 7.0]
    # The middle blocks move the prefix to 11 and the pairs to 2.5 and 6.
    reduced = torch.tensor([11.0, 2.5, 6.0]).view(1, 3, 1)
    restored = expand_token_pairs(hidden, reduced, pooled, plan)
    assert restored.flatten().tolist() == [11.0, 1.5, 3.5, 4.0, 8.0]


def test_update_scale_scales_only_the_pair_update() -> None:
    plan = build_token_reduction(_positions(5), 1, (1, 1, 4), _schedule(update_scale=0.5))
    hidden = torch.tensor([10.0, 1.0, 3.0, 5.0, 9.0]).view(1, 5, 1)
    pooled = pool_token_pairs(hidden, plan)
    reduced = torch.tensor([11.0, 2.5, 6.0]).view(1, 3, 1)
    assert expand_token_pairs(hidden, reduced, pooled, plan).flatten().tolist() == [11.0, 1.25, 3.25, 4.5, 8.5]


def test_unchanged_reduced_stack_restores_bf16_input_exactly() -> None:
    plan = build_token_reduction(_positions(2 + 2 * 3 * 5), 2, (2, 3, 5), _schedule())
    hidden = torch.randn(1, plan.sequence_length, 8, generator=torch.Generator().manual_seed(0)).bfloat16()
    pooled = pool_token_pairs(hidden, plan)
    assert torch.equal(expand_token_pairs(hidden, pooled, pooled, plan), hidden)


def test_one_token_wide_video_is_an_identity() -> None:
    plan = build_token_reduction(_positions(2 + 3), 2, (3, 1, 1), _schedule())
    assert plan.reduced_length == plan.sequence_length
    assert plan.pair_rows.numel() == 0
    hidden = torch.randn(1, plan.sequence_length, 4).bfloat16()
    assert torch.equal(pool_token_pairs(hidden, plan), hidden)


# ----------------------------------------------------------------------------
# Denoising-stage helpers on a real packed layout
# ----------------------------------------------------------------------------


def _layout(latent_width: int, keyframes: tuple[str, ...] = ()):
    return build_packed_sequence(
        text_token_tags=torch.ones(4, dtype=torch.long),
        num_latent_frames=2,
        latent_height=4,
        latent_width=latent_width,
        num_audio_latents=3,
        patch_size=_PATCH,
        keyframe_anchors=keyframes,
    )


@pytest.mark.parametrize("keyframes", [(), ("first", )])
def test_stage_plan_pairs_only_the_generated_video(keyframes) -> None:
    layout = _layout(latent_width=6, keyframes=keyframes)
    plan = _h3_token_reduction_plan(layout, _PATCH, _schedule())
    video_start = layout.sequence_length - 2 * 2 * 3
    assert plan.sequence_length == layout.sequence_length
    assert plan.reduced_length == video_start + 2 * 2 * 2
    assert plan.first[:video_start].tolist() == list(range(video_start))
    assert int(plan.pair_rows.min()) >= video_start


def test_reduced_segments_halve_the_last_region_width() -> None:
    assert _h3_token_reduced_segments((4, 0, 6, (2, 4, 6)), _PATCH) == (4, 0, 6, (2, 4, 4))
    assert _h3_token_reduced_segments((4, (3, 4, 4), 6, (2, 4, 8)), _PATCH) == (4, (3, 4, 4), 6, (2, 4, 4))


# ----------------------------------------------------------------------------
# Tiny real transformer (GPU)
# ----------------------------------------------------------------------------


def _tiny_transformer(device: torch.device):
    from fastvideo.models.dits.minimax_h3 import MiniMaxH3Transformer3DModel

    arch = MiniMaxH3ArchConfig(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=3,
        num_refiner_layers=1,
        ffn_dim=64,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
    )
    transformer = MiniMaxH3Transformer3DModel(config=MiniMaxH3Config(arch_config=arch), hf_config={})
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for name, parameter in transformer.named_parameters():
            if parameter.ndim < 2 and "norm" in name:
                parameter.fill_(1.0)
            else:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return transformer.to(device).eval()


def _forward(transformer, layout, inputs, token_reduction=None):
    device = inputs["hidden_states"].device
    unique, inverse = build_row_timesteps(layout, 0.5, 0.5, 0.999, 1.0)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        outputs = transformer(
            timestep=unique.to(device),
            timestep_indices=inverse.to(device),
            token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device),
            video_indices=layout.video_indices.to(device),
            audio_indices=layout.audio_indices.to(device),
            text_indices=layout.text_indices.to(device),
            token_reduction=token_reduction,
            **inputs,
        )
        # The full-sequence metadata is back in place after the forward.
        assert get_forward_context().attn_metadata is None
    return outputs


def _inputs(transformer, layout, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1)

    def rows(count: int, width: int) -> torch.Tensor:
        return torch.randn(1, count, width, generator=generator).to(device)

    return {
        "hidden_states": rows(layout.video_indices.numel(), transformer.proj_in.weight.shape[1]),
        "audio_hidden_states": rows(layout.audio_indices.numel(), transformer.audio_proj_in.weight.shape[1]),
        "encoder_hidden_states": rows(layout.text_indices.numel(), transformer.context_embedder.weight.shape[1]),
    }


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="MiniMax H3 attention layers require a CUDA device for a real forward")
def test_one_token_wide_reduced_forward_is_bit_identical(env_overrides, distributed_setup) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    device = torch.device("cuda")
    transformer = _tiny_transformer(device)
    layout = _layout(latent_width=2)
    inputs = _inputs(transformer, layout, device)
    plan = _h3_token_reduction_plan(layout, _PATCH, _schedule(blocks="1:2", early="0:0")).to(device)
    plain = _forward(transformer, layout, inputs)
    reduced = _forward(transformer, layout, inputs, plan)
    for expected, actual in zip(plain, reduced, strict=True):
        assert torch.equal(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="MiniMax H3 attention layers require a CUDA device for a real forward")
@pytest.mark.parametrize("blocks", ["1:2", "0:3"])
def test_reduced_forward_keeps_shapes_and_changes_video(env_overrides, distributed_setup, blocks) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
    device = torch.device("cuda")
    transformer = _tiny_transformer(device)
    layout = _layout(latent_width=8)
    inputs = _inputs(transformer, layout, device)
    plan = _h3_token_reduction_plan(layout, _PATCH, _schedule(blocks=blocks, early="0:0")).to(device)
    plain_video, plain_audio = _forward(transformer, layout, inputs)
    video, audio = _forward(transformer, layout, inputs, plan)
    assert video.shape == plain_video.shape and audio.shape == plain_audio.shape
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert not torch.equal(video, plain_video)
