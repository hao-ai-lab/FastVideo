# SPDX-License-Identifier: Apache-2.0
"""PDD sampling and Ref2VA reference-video VSA in the MiniMax-H3 denoising stage.

A PDD-widened transformer samples fused blocks of its fine grid: the stage
partitions the grid (the checkpoint contract's ``pdd_step_indices``), hands
each scheduler its modality's node sigmas, starts the targets at the first
node's noise level, and fuses one block of heads per forward. The ordinary
Euler step over two node sigmas then equals the block advance. With
``vsa_ref_keep_rate`` set, every reference video is its own sparse VSA region.
All checks are CPU-only; the transformer is a tiny stand-in with real widened
heads.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from fastvideo.attention.backends.video_sparse_attn_h3 import MiniMaxH3VSAMetadataBuilder
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.forward_context import get_forward_context
from fastvideo.layers.pdd import (
    PDD_GRID_MAX_T,
    PDDModalitySchedule,
    PDDReplicatedLinear,
    fuse_pdd_heads,
    pdd_fine_grid,
    shifted_noise_amount,
)
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_TEXT_TAG,
    build_ref2va_packed_sequence,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_denoising as denoising
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

PATCH = (1, 2, 2)
VIDEO_CHANNELS = 96  # 24 latent channels x (1, 2, 2) patch
AUDIO_CHANNELS = 32
GRID32_BLOCKS8 = tuple(range(0, 33, 4))
# A 4-block partition of an 8-interval grid, for checks that need a short plan.
GRID8_BLOCKS4 = (0, 2, 4, 6, 8)


def _reference(kind: str, *, frames: int = 1, height: int = 4, width: int = 4, audio: int = 0):
    return MiniMaxH3PreparedReference(media_type=kind,
                                      has_audio=audio > 0,
                                      num_latent_frames=frames,
                                      latent_height=height,
                                      latent_width=width,
                                      num_audio_latents=audio)


def _ref2va_layout(references, *, text=5, frames=3, height=4, width=6, audio=3):
    tags = torch.full((text, ), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_ref2va_packed_sequence(tags, references, frames, height, width, audio, PATCH)


def _stage() -> denoising.MiniMaxH3DenoisingStage:
    return denoising.MiniMaxH3DenoisingStage(
        transformer=SimpleNamespace(),
        scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )


def _args(**pipeline_fields):
    return SimpleNamespace(pipeline_config=MiniMaxH3PipelineConfig(**pipeline_fields),
                           dit_cpu_offload=False,
                           dit_layerwise_offload=False,
                           use_fsdp_inference=False,
                           VSA_tile_size=128)


def _plan(stage, args):
    return stage._pdd_sampling_plan(args, torch.device("cpu"))


def test_pdd_sampling_plan_unset_step_indices():
    assert _plan(_stage(), _args()) is None


def test_pdd_sampling_plan_uses_contract_partition_and_modality_shifts():
    plan = _plan(_stage(), _args(pdd_step_indices=GRID32_BLOCKS8))
    assert plan.indices.tolist() == list(GRID32_BLOCKS8)
    base = pdd_fine_grid(32)[list(GRID32_BLOCKS8)]
    torch.testing.assert_close(plan.node_sigmas["video"], shifted_noise_amount(base, 12.0), rtol=0, atol=0)
    torch.testing.assert_close(plan.node_sigmas["audio"], shifted_noise_amount(base, 3.0), rtol=0, atol=0)


def test_step_pdd_block_equals_fused_block_advance():
    """x_end = x_start + sum_j w_j * dx/dsigma_j with the raw H3 output m = x0 - eps = -dx/dsigma."""
    torch.manual_seed(0)
    stage = _stage()
    plan = _plan(stage, _args(pdd_step_indices=GRID8_BLOCKS4))
    stage.scheduler.set_timesteps(sigmas=plan.node_sigmas["video"].to(torch.float32))
    stage.audio_scheduler.set_timesteps(sigmas=plan.node_sigmas["audio"].to(torch.float32))
    video, audio = torch.randn(5, 6), torch.randn(4, 3)
    for step in range(plan.num_steps):
        start, end = plan.block(step)
        fused_video, fused_audio = torch.randn(5, 6), torch.randn(4, 3)
        stepped_video = stage.scheduler.step(fused_video, stage.scheduler.timesteps[step], video, return_dict=False)[0]
        stepped_audio = stage.audio_scheduler.step(fused_audio,
                                                   stage.audio_scheduler.timesteps[step],
                                                   audio,
                                                   return_dict=False)[0]
        for name, state, fused, stepped in (("video", video, fused_video, stepped_video),
                                            ("audio", audio, fused_audio, stepped_audio)):
            total = plan.integration_weights[name][start:end].sum()
            expected = state.double() - total * fused.double()
            torch.testing.assert_close(stepped.double(), expected, rtol=1e-5, atol=1e-5)
        video, audio = stepped_video, stepped_audio


def test_scale_pdd_initial_noise_starts_targets_at_first_node_sigma():
    plan = _plan(_stage(), _args(pdd_step_indices=GRID8_BLOCKS4))
    layout = SimpleNamespace(num_condition_video_rows=2, num_condition_audio_rows=1)
    video, audio = torch.randn(5, 6), torch.randn(4, 3)
    expected_video, expected_audio = video.clone(), audio.clone()
    expected_video[2:] *= PDD_GRID_MAX_T
    expected_audio[1:] *= PDD_GRID_MAX_T

    denoising.scale_pdd_initial_noise(video, audio, layout, plan)

    assert float(plan.node_sigmas["video"][0]) == pytest.approx(PDD_GRID_MAX_T)
    assert float(plan.node_sigmas["audio"][0]) == pytest.approx(PDD_GRID_MAX_T)
    torch.testing.assert_close(video, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(audio, expected_audio, rtol=0, atol=0)


class _TinyPDDTransformer:
    """Stand-in DiT with real widened heads: each row's output depends on its own value and clock."""

    def __init__(self, pdd_steps: int, hidden: int = 8):
        self.pdd_steps = pdd_steps
        generator = torch.Generator().manual_seed(7)
        self.video_in = torch.randn(VIDEO_CHANNELS, hidden, generator=generator) / VIDEO_CHANNELS**0.5
        self.audio_in = torch.randn(AUDIO_CHANNELS, hidden, generator=generator) / AUDIO_CHANNELS**0.5
        self.proj_out = PDDReplicatedLinear(hidden, VIDEO_CHANNELS, grid_size=pdd_steps, params_dtype=torch.float32)
        self.audio_proj_out = PDDReplicatedLinear(hidden,
                                                  AUDIO_CHANNELS,
                                                  grid_size=pdd_steps,
                                                  params_dtype=torch.float32)
        with torch.no_grad():
            for linear in (self.proj_out, self.audio_proj_out):
                linear.weight.copy_(torch.randn(linear.weight.shape, generator=generator) * 0.1)
                linear.bias.copy_(torch.randn(linear.bias.shape, generator=generator) * 0.1)
        self.blocks: list[tuple[int, int]] = []
        self.calls: list[dict] = []

    @contextlib.contextmanager
    def fuse_pdd_block(self, start, end, integration_weights, precision_decoding):
        self.blocks.append((start, end))
        with fuse_pdd_heads({"video": self.proj_out, "audio": self.audio_proj_out}, start, end, integration_weights,
                            precision_decoding):
            yield

    def features(self, rows, weights, row_times):
        return torch.tanh(rows.float() @ weights + row_times[:, None])

    def __call__(self, **kwargs):
        row_times = kwargs["timestep"][kwargs["timestep_indices"]]
        self.calls.append({
            **{key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)},
            "attn_metadata": get_forward_context().attn_metadata,
        })
        video = self.features(kwargs["hidden_states"][0], self.video_in, row_times[kwargs["video_indices"]])
        audio = self.features(kwargs["audio_hidden_states"][0], self.audio_in, row_times[kwargs["audio_indices"]])
        return self.proj_out(video)[0][None], self.audio_proj_out(audio)[0][None]


def _run(monkeypatch, layout, transformer, *, args, num_inference_steps=8, sparsity=0.0, extra=None, builder=None):
    monkeypatch.setattr(denoising, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(denoising, "_h3_vsa_metadata_builder", lambda *_: builder)
    stage = denoising.MiniMaxH3DenoisingStage(transformer, MiniMaxH3Scheduler(shift=12.0),
                                              MiniMaxH3Scheduler(shift=3.0))
    generator = torch.Generator().manual_seed(3)
    latents = torch.randn(int(layout.video_indices.numel()), VIDEO_CHANNELS, generator=generator)
    audio_latents = torch.randn(int(layout.audio_indices.numel()), AUDIO_CHANNELS, generator=generator)
    batch = ForwardBatch(data_type="video",
                         prompt_embeds=[torch.zeros(1, int(layout.text_indices.numel()), 8)],
                         latents=latents.clone(),
                         audio_latents=audio_latents.clone(),
                         num_inference_steps=num_inference_steps,
                         VSA_sparsity=sparsity,
                         extra={
                             MINIMAX_H3_LAYOUT_KEY: layout,
                             **(extra or {})
                         })
    return stage, stage.forward(batch, args), latents, audio_latents


def _reference_video_and_audio_references():
    return [
        _reference("image", height=4, width=6),
        _reference("video", frames=5, height=4, width=6, audio=2),
        _reference("audio", audio=3),
    ]


def test_forward_pdd_runs_eight_fused_blocks_from_contract(monkeypatch):
    layout = _ref2va_layout(_reference_video_and_audio_references())
    transformer = _TinyPDDTransformer(pdd_steps=32)
    args = _args(pdd_step_indices=GRID32_BLOCKS8)
    stage, result, latents, audio_latents = _run(monkeypatch, layout, transformer, args=args)

    assert len(transformer.calls) == 8 and result.step_index == 7
    assert transformer.blocks == list(zip(GRID32_BLOCKS8[:-1], GRID32_BLOCKS8[1:], strict=True))
    for proj in (transformer.proj_out, transformer.audio_proj_out):
        assert proj._fusion_state is None

    # Independent recomputation: node sigmas on each modality's clock, targets
    # start at 0.999 * noise, conditions stay fixed, and every block advances
    # x by -(sigma_end - sigma_start) * (weighted mean of its raw heads).
    base = pdd_fine_grid(32)[list(GRID32_BLOCKS8)]
    sigmas = {"video": shifted_noise_amount(base, 12.0), "audio": shifted_noise_amount(base, 3.0)}
    torch.testing.assert_close(stage.scheduler.sigmas, sigmas["video"].float(), rtol=0, atol=0)
    torch.testing.assert_close(stage.audio_scheduler.sigmas, sigmas["audio"].float(), rtol=0, atol=0)
    weights = {name: PDDModalitySchedule(shift).integration_weights(pdd_fine_grid(32))
               for name, shift in (("video", 12.0), ("audio", 3.0))}
    video_start, audio_start = layout.num_condition_video_rows, layout.num_condition_audio_rows
    video, audio = latents.double().clone(), audio_latents.double().clone()
    video[video_start:] *= PDD_GRID_MAX_T
    audio[audio_start:] *= PDD_GRID_MAX_T
    for step, call in enumerate(transformer.calls):
        start, end = GRID32_BLOCKS8[step], GRID32_BLOCKS8[step + 1]
        torch.testing.assert_close(call["hidden_states"][0].double(), video, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(call["audio_hidden_states"][0].double(), audio, rtol=1e-5, atol=1e-5)
        row_times = call["timestep"][call["timestep_indices"]]
        video_time = 1.0 - float(sigmas["video"][step].float())
        audio_time = 1.0 - float(sigmas["audio"][step].float())
        video_rows, audio_rows = call["video_indices"], call["audio_indices"]
        for rows, value in ((video_rows[video_start:], video_time), (video_rows[:video_start],
                                                                     max(video_time, MINIMAX_H3_KEYFRAME_NOISE_AUG)),
                            (audio_rows[audio_start:], audio_time), (audio_rows[:audio_start], 1.0),
                            (call["text_indices"], video_time)):
            torch.testing.assert_close(row_times[rows], torch.full_like(row_times[rows], value), rtol=0, atol=1e-7)
        for name, state, rows, first, heads, features_in in (
            ("video", video, video_rows, video_start, transformer.proj_out, transformer.video_in),
            ("audio", audio, audio_rows, audio_start, transformer.audio_proj_out, transformer.audio_in),
        ):
            features = transformer.features(state, features_in, row_times[rows].double().float())
            materialized = heads(features.float())[0].double().unflatten(-1, (32, -1))
            alpha = weights[name][start:end]
            mean = torch.einsum("n,rnc->rc", alpha / alpha.sum(), materialized[:, start:end])
            state[first:] -= (sigmas[name][step + 1] - sigmas[name][step]) * mean[first:]
    torch.testing.assert_close(result.latents.double(), video, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(result.audio_latents.double(), audio, rtol=1e-4, atol=1e-4)
    # Reference conditions never move.
    torch.testing.assert_close(result.latents[:video_start], latents[:video_start], rtol=0, atol=0)
    torch.testing.assert_close(result.audio_latents[:audio_start], audio_latents[:audio_start], rtol=0, atol=0)


@pytest.mark.parametrize("dense_first_n", [0, 1])
def test_forward_ref_keep_rate_builds_one_sparse_region_per_reference_video(monkeypatch, dense_first_n):
    references = [
        _reference("video", frames=5, height=4, width=6),
        _reference("image", height=4, width=6),
        _reference("video", frames=2, height=8, width=4, audio=2),
    ]
    layout = _ref2va_layout(references)
    transformer = _TinyPDDTransformer(pdd_steps=32)
    args = _args(pdd_step_indices=GRID32_BLOCKS8, vsa_ref_keep_rate=0.25)
    _run(monkeypatch,
         layout,
         transformer,
         args=args,
         sparsity=0.9,
         extra={"vsa_dense_first_n_steps": dense_first_n},
         builder=MiniMaxH3VSAMetadataBuilder())

    assert len(transformer.calls) == 8
    for step, call in enumerate(transformer.calls):
        metadata = call["attn_metadata"]
        assert metadata.tile_elems == 128
        assert metadata.total_seq_length == layout.sequence_length
        assert len(metadata.video_tile_spans) == 3
        expected = (0.0, 0.0, 0.0) if step < dense_first_n else (0.75, 0.75, 0.9)
        assert metadata.span_sparsities == pytest.approx(expected)


@pytest.mark.parametrize("sparsity,effect", [
    (0.9, "each reference keeps 0.25 of its tiles and the target keeps 0.1"),
    (0.0, "VSA_sparsity=0 keeps every region dense"),
])
def test_forward_ref_keep_rate_logs_applied_sparsity(monkeypatch, sparsity, effect):
    infos = []
    monkeypatch.setattr(denoising.logger, "info", lambda message, *args: infos.append(message % args))
    layout = _ref2va_layout(_reference_video_and_audio_references())
    args = _args(pdd_step_indices=GRID32_BLOCKS8, vsa_ref_keep_rate=0.25)
    _run(monkeypatch,
         layout,
         _TinyPDDTransformer(pdd_steps=32),
         args=args,
         sparsity=sparsity,
         builder=MiniMaxH3VSAMetadataBuilder())
    assert f"MiniMax-H3 VSA-H3: 1 reference video region(s); {effect}; 128-token tiles." in infos


def test_forward_unset_ref_keep_rate_keeps_references_in_dense_prefix(monkeypatch):
    layout = _ref2va_layout(_reference_video_and_audio_references())
    transformer = _TinyPDDTransformer(pdd_steps=32)
    args = _args(pdd_step_indices=GRID32_BLOCKS8)
    _run(monkeypatch, layout, transformer, args=args, sparsity=0.9, builder=MiniMaxH3VSAMetadataBuilder())
    for call in transformer.calls:
        metadata = call["attn_metadata"]
        # The generated video is the only sparse region.
        assert len(metadata.video_tile_spans) == 1 and metadata.span_sparsities == (0.9, )


def test_forward_ref_keep_rate_requires_exempt_prefix_keys(monkeypatch):
    layout = _ref2va_layout(_reference_video_and_audio_references())
    args = _args(pdd_step_indices=GRID32_BLOCKS8, vsa_ref_keep_rate=0.1)
    with pytest.raises(ValueError, match="compete' supports only the generated-video region"):
        _run(monkeypatch,
             layout,
             _TinyPDDTransformer(pdd_steps=32),
             args=args,
             sparsity=0.9,
             extra={"vsa_mode": "compete"},
             builder=MiniMaxH3VSAMetadataBuilder())
