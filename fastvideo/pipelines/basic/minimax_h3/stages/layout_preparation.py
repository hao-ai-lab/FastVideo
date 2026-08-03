# SPDX-License-Identifier: Apache-2.0
"""Packed row-layout preparation for MiniMax H3 tasks."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.packing import build_packed_sequence
from fastvideo.pipelines.basic.minimax_h3.packing_ref2va import build_ref2va_packed_sequence
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult


def _patch_size(transformer: Any) -> tuple[int, int, int]:
    value = getattr(transformer, "patch_size", None)
    if value is None:
        config = getattr(transformer, "config", None)
        arch = getattr(config, "arch_config", config)
        value = getattr(arch, "patch_size", None)
    if value is None:
        raise ValueError(f"MiniMax-H3 component {type(transformer).__name__} does not expose `patch_size`.")
    patch_size = tuple(int(item) for item in value)
    if len(patch_size) != 3:
        raise ValueError(f"MiniMax-H3 patch_size must have three axes, got {patch_size}.")
    return patch_size[0], patch_size[1], patch_size[2]


class _MiniMaxH3LayoutPreparationStage(PipelineStage):

    def __init__(self, transformer: Any) -> None:
        super().__init__()
        self.transformer = transformer

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("text_token_tags", state.text_token_tags, V.with_dims(1))
        result.add_check("num_latent_frames", state.num_latent_frames, V.positive_int)
        result.add_check("latent_height", state.latent_height, V.positive_int)
        result.add_check("latent_width", state.latent_width, V.positive_int)
        result.add_check("num_audio_latents", state.num_audio_latents, V.positive_int)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        state = get_minimax_h3_state(batch)
        result = VerificationResult()
        result.add_check("layout", state.layout, V.not_none)
        return result

    @staticmethod
    def _inputs(batch: ForwardBatch) -> tuple[Any, int, int, int, int]:
        state = get_minimax_h3_state(batch)
        if state.text_token_tags is None:
            raise ValueError("MiniMax-H3 conditioning must run before layout preparation.")
        geometry = (state.num_latent_frames, state.latent_height, state.latent_width, state.num_audio_latents)
        if any(value is None for value in geometry):
            raise ValueError("MiniMax-H3 target latent geometry is incomplete.")
        return state, *(int(value) for value in geometry if value is not None)


class MiniMaxH3FL2VALayoutPreparationStage(_MiniMaxH3LayoutPreparationStage):
    """Build the T2VA/FL2VA packed row layout."""

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        state, num_frames, height, width, num_audio_latents = self._inputs(batch)
        state.layout = build_packed_sequence(
            state.text_token_tags,
            num_frames,
            height,
            width,
            num_audio_latents,
            _patch_size(self.transformer),
            state.keyframe_anchors,
        )
        return batch


class MiniMaxH3Ref2VALayoutPreparationStage(_MiniMaxH3LayoutPreparationStage):
    """Build the ordered-reference packed row layout."""

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_input(batch, fastvideo_args)
        state = get_minimax_h3_state(batch)
        result.add_check("prepared_references", state.prepared_references, V.list_not_empty)
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        del fastvideo_args
        state, num_frames, height, width, num_audio_latents = self._inputs(batch)
        if not state.prepared_references:
            raise ValueError("MiniMax-H3 Ref2VA references must be encoded before layout preparation.")
        state.layout = build_ref2va_packed_sequence(
            state.text_token_tags,
            state.prepared_references,
            num_frames,
            height,
            width,
            num_audio_latents,
            _patch_size(self.transformer),
        )
        return batch


__all__ = ["MiniMaxH3FL2VALayoutPreparationStage", "MiniMaxH3Ref2VALayoutPreparationStage"]
