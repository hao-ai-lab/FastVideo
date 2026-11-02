# SPDX-License-Identifier: Apache-2.0
"""Fixed-sigma Euler refine loop of Open-H3-Regenerate-2K."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.packing import MiniMaxH3PackedLayout, build_row_timesteps
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_denoising import (
    _h3_vsa_metadata_builder,
    _h3_vsa_single_region_segments,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sampling import refine_sigmas
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sharp_curve import SharpCurveBranches
from fastvideo.pipelines.lazy_module import is_lazy_module
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class Regenerate2KDenoisingStage(PipelineStage):
    """Euler steps over the refine sigmas with the video clock at ``1 - sigma`` and the silent audio held clean."""

    performance_component_metric = "dit_time_s"

    def __init__(self, transformer: Any, curve: SharpCurveBranches, start_sigma: float | None) -> None:
        super().__init__()
        self.transformer = transformer
        self.curve = curve
        self.start_sigma = start_sigma

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("layout", batch.extra.get(MINIMAX_H3_LAYOUT_KEY), V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(3))
        result.add_check("latents", batch.latents, V.with_dims(2))
        result.add_check("audio_latents", batch.audio_latents, V.with_dims(2))
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.with_dims(2))
        return result

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        transformer = self.transformer
        self.curve.attach(transformer.materialize() if is_lazy_module(transformer) else transformer)
        layout = batch.extra.get(MINIMAX_H3_LAYOUT_KEY)
        if not isinstance(layout, MiniMaxH3PackedLayout):
            raise ValueError("The MiniMax-H3 packed layout is missing before denoising.")
        device = get_local_torch_device()
        sigmas = refine_sigmas(batch.num_inference_steps, self.start_sigma)
        state = batch.latents.to(device)
        vsa = self._vsa_plan(batch, layout, transformer, fastvideo_args, device)
        layerwise = bool(getattr(fastvideo_args, "dit_layerwise_offload", False))
        full_cpu_offload = (bool(getattr(fastvideo_args, "dit_cpu_offload", False)) and not layerwise
                            and not bool(getattr(fastvideo_args, "use_fsdp_inference", False)))
        try:
            if full_cpu_offload:
                transformer.to(device)
            with self._attention_kernel(device):
                state = self._euler_loop(batch, layout, state, sigmas, device, transformer, vsa)
        finally:
            if layerwise:
                manager = getattr(transformer, "_layerwise_offload_manager", None)
                if manager is not None and getattr(manager, "enabled", False):
                    manager.release_all()
            if full_cpu_offload:
                transformer.to("cpu")
        batch.latents = state
        return batch

    @staticmethod
    def _vsa_plan(batch: ForwardBatch, layout: MiniMaxH3PackedLayout, transformer: Any, fastvideo_args: FastVideoArgs,
                  device: torch.device) -> tuple[Any, ...] | None:
        """Per-step VSA metadata inputs, or None for a dense backend.

        The builder and the packed segments come from MiniMax-H3's own denoising stage so the refine sees the same
        sparse layout its t2v runs do; the refine layout is a single generated-video region (no reference rows).
        """
        builder = _h3_vsa_metadata_builder(transformer, fastvideo_args)
        if builder is None:
            return None
        patch_size = fastvideo_args.pipeline_config.dit_config.patch_size
        sparsity = fastvideo_args.VSA_sparsity
        if sparsity is None:
            sparsity = getattr(batch, "VSA_sparsity", 0.0)
        tile_size = fastvideo_args.VSA_tile_size
        return (builder, patch_size, _h3_vsa_single_region_segments(layout,
                                                                    patch_size), float(sparsity), tile_size, device)

    @staticmethod
    def _attention_kernel(device: torch.device) -> contextlib.AbstractContextManager[None]:
        """cuDNN attention: at 133k tokens PyTorch's default SDPA dispatch picks its flash kernel, 3x slower on B200."""
        return sdpa_kernel(SDPBackend.CUDNN_ATTENTION) if device.type == "cuda" else contextlib.nullcontext()

    def _euler_loop(self, batch: ForwardBatch, layout: MiniMaxH3PackedLayout, state: torch.Tensor, sigmas: list[float],
                    device: torch.device, transformer: Any, vsa: Any) -> torch.Tensor:
        position_ids = layout.position_ids.to(device)
        token_tags = layout.token_tags.to(device)
        video_indices = layout.video_indices.to(device)
        audio_indices = layout.audio_indices.to(device)
        text_indices = layout.text_indices.to(device)
        prompt_embeds = batch.prompt_embeds[0].to(device)
        audio = batch.audio_latents.to(device=device, dtype=torch.bfloat16)[None]
        for index, (sigma, following) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            unique, inverse = build_row_timesteps(layout,
                                                  video_timestep=1.0 - sigma,
                                                  audio_timestep=1.0,
                                                  condition_video_timestep=1.0,
                                                  condition_audio_timestep=1.0)
            unique, inverse = unique.to(device), inverse.to(device)
            self.curve.set_timesteps(unique)
            attn_metadata = None
            if vsa is not None:
                builder, patch_size, segments, sparsity, tile_size, vsa_device = vsa
                attn_metadata = builder.build(current_timestep=index,
                                              patch_size=patch_size,
                                              VSA_sparsity=sparsity,
                                              packed_segments=segments,
                                              device=vsa_device,
                                              exempt=True,
                                              dense_layers=(),
                                              tile_size=tile_size)
            torch.compiler.cudagraph_mark_step_begin()
            with set_forward_context(current_timestep=index, attn_metadata=attn_metadata, forward_batch=batch):
                velocity, _ = transformer(hidden_states=state.to(torch.bfloat16)[None],
                                          audio_hidden_states=audio,
                                          encoder_hidden_states=prompt_embeds,
                                          timestep=unique,
                                          timestep_indices=inverse,
                                          token_tags=token_tags,
                                          position_ids=position_ids,
                                          video_indices=video_indices,
                                          audio_indices=audio_indices,
                                          text_indices=text_indices)
            state = state + (sigma - following) * velocity[0].float()
            batch.step_index = index
        return state


__all__ = ["Regenerate2KDenoisingStage"]
