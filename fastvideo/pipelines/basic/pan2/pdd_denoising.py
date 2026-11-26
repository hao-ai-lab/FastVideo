# SPDX-License-Identifier: Apache-2.0
"""Sampling a PAN2 Parallel Decoding Distillation (PDD) student: one transformer forward per fused block, no CFG.

A PDD export's ``fastvideo_inference.json`` (``PAN2PipelineConfig.resolve_checkpoint_settings``) names its trained
partition of the fine grid. Each runtime step fuses that block's heads inside the widened ``proj_out`` into their
weighted mean velocity and advances the fp32 state by the block's total sigma increment. This is the integration the
student trains and validates with (``Pan2PDDValidationCallback``): the same clock, the same fused heads, the same fp32
update, and the transformer input in the same memory layout, so an export samples here exactly what validation
showed.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.attention.selector import component_attention_backend, get_attn_backend
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.layers.pdd import PDD_GRID_MAX_T, PDDModalitySchedule, build_pdd_sampling_plan, shifted_noise_amount
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.pan2.stages import PAN2DenoisingStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.profiler import profiler_region
from fastvideo.utils import get_compute_dtype

logger = init_logger(__name__)

# Method timesteps are ``u * 1000`` on the base clock (``Pan2PDDModel.pdd_timestep``).
_TIMESTEP_SCALE = 1000


def pan2_pdd_transformer_timestep(base: torch.Tensor, shift: float) -> torch.Tensor:
    """The fp32 timestep the transformer reads at base-clock time ``u`` (``[1]`` float64): ``sigma(u) * 1000``.

    The arithmetic is the training plugin's, step for step (``Pan2PDDModel.pdd_timestep``, then
    ``Pan2DMDModel._noise_amount`` and ``_forward_raw``), so a student sees the bits it was trained on.
    """
    method_timestep = base.to(torch.float64).reshape(-1) * _TIMESTEP_SCALE
    unit = method_timestep.reshape(-1)[:1].to(torch.float64) / _TIMESTEP_SCALE
    sigma = shifted_noise_amount(unit.clamp(0.0, PDD_GRID_MAX_T), shift, max_t=PDD_GRID_MAX_T)
    return (sigma * _TIMESTEP_SCALE).to(torch.float32)


def _vsa_metadata_builder(transformer: Any, fastvideo_args: FastVideoArgs) -> Any:
    """A VSA-H3 metadata builder when the transformer resolved to VIDEO_SPARSE_ATTN_H3, else None."""
    dit_config = fastvideo_args.pipeline_config.dit_config
    backend = get_attn_backend(
        head_size=dit_config.arch_config.attention_head_dim,
        dtype=get_compute_dtype(),
        supported_attention_backends=dit_config.arch_config._supported_attention_backends,
        requested=component_attention_backend(transformer),
    )
    if backend.get_name() != "VIDEO_SPARSE_ATTN_H3":
        return None
    return backend.get_builder_cls()()


class PAN2PDDDenoisingStage(PAN2DenoisingStage):
    """Denoise with a PAN2 PDD student's trained partition: ``num_blocks`` conditional forwards, no guidance.

    The state starts at the first node's noise level and stays in fp32, frames first (the packed layout the student
    trains on); the transformer sees it as ``[1, C, T, H, W]`` beside the conditioning stack, both in bf16. A sparse
    student's VSA-H3 metadata describes the joint ``[video; text]`` sequence with the tile policy the student trained
    with, built once for the request as training builds it once per clip.
    """

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pipeline_config = fastvideo_args.pipeline_config
        indices = pipeline_config.pdd_step_indices
        if indices is None:
            raise ValueError("PAN2PDDDenoisingStage samples PDD checkpoints only (pdd_step_indices is unset)")
        if batch.do_classifier_free_guidance:
            raise ValueError("A PAN2 PDD student samples without classifier-free guidance; set guidance_scale=1")
        self.activate_transformer(self.transformer, None, fastvideo_args)
        transformer = self.transformer
        device = get_local_torch_device()
        if batch.latents is None or batch.image_latent is None or not batch.prompt_embeds:
            raise ValueError("PAN2 PDD denoising needs the initial latents, the conditioning stack and the prompt")

        shift = float(self.scheduler.shift)
        plan = build_pdd_sampling_plan(indices, {"video": PDDModalitySchedule(shift)}, device=device)
        if plan.pdd_steps != getattr(transformer, "pdd_steps", None):
            raise ValueError(f"The trained partition {list(indices)} spans a {plan.pdd_steps}-interval grid; the "
                             f"transformer has pdd_steps={getattr(transformer, 'pdd_steps', None)!r}")
        dtype = torch.bfloat16
        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=dtype)
        condition = batch.image_latent.to(device=device, dtype=dtype)
        # Frames first, as the student's packed state: [1, C, T, H, W] -> [1, T, C, H, W].
        state = batch.latents.to(device=device, dtype=torch.float32).permute(0, 2, 1, 3, 4).contiguous()
        state = state * plan.node_sigmas["video"][:1].to(device=device, dtype=torch.float32)

        attn_metadata = None
        builder = _vsa_metadata_builder(transformer, fastvideo_args)
        if builder is not None:
            _, num_latents, _, height, width = state.shape
            attn_metadata = builder.build(
                current_timestep=0,
                patch_size=pipeline_config.dit_config.patch_size,
                VSA_sparsity=float(batch.VSA_sparsity),
                # PAN2 joins the sequence as [video; text]: one sparse video region, then the dense text rows.
                packed_segments=((num_latents, height, width), int(prompt_embeds.shape[1])),
                device=device,
                tile_size=int(fastvideo_args.VSA_tile_size),
                video_tile_policy=tuple(pipeline_config.vsa_tile_policy),
                compress_prefix_keys=bool(pipeline_config.vsa_compress_text_keys),
                variant=pipeline_config.vsa_variant or "vsa",
                dense_layers=tuple(pipeline_config.vsa_dense_layers or ()),
                heads_first_tile=bool(getattr(pipeline_config, "vsa_heads_first_tile", False)),
            )
        logger.info("PAN2 PDD denoising: %d fused block(s) %s over a %d-interval fine grid%s.", plan.num_steps,
                    plan.block_sizes, plan.pdd_steps,
                    "" if builder is None else f", VSA-H3 ({pipeline_config.vsa_variant or 'vsa'})")

        weights = plan.integration_weights
        timesteps = []
        with profiler_region("inference_denoising"), self.progress_bar(total=plan.num_steps) as progress_bar:
            for step in range(plan.num_steps):
                start, end = plan.block(step)
                timestep = pan2_pdd_transformer_timestep(plan.fine_grid[start:start + 1], shift).to(device)
                timesteps.append(timestep)
                model_input = torch.cat([state.permute(0, 2, 1, 3, 4).to(dtype), condition], dim=1)
                with transformer.fuse_pdd_block(start, end, weights, torch.float32), torch.autocast(
                        device.type, dtype=dtype), set_forward_context(current_timestep=timestep,
                                                                       attn_metadata=attn_metadata):
                    output = transformer(hidden_states=model_input,
                                         encoder_hidden_states=prompt_embeds,
                                         timestep=timestep)
                velocity = output.permute(0, 2, 1, 3, 4).to(torch.float32)
                scale = weights["video"][start:end].sum().to(device=device, dtype=torch.float32)
                state = state + velocity * scale
                batch.step_index = step
                progress_bar.update()

        batch.timesteps = torch.cat(timesteps)
        batch.latents = state.permute(0, 2, 1, 3, 4).contiguous()
        if fastvideo_args.dit_layerwise_offload:
            manager = getattr(transformer, "_layerwise_offload_manager", None)
            if manager is not None and getattr(manager, "enabled", False):
                manager.release_all()
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result


__all__ = ["PAN2PDDDenoisingStage", "pan2_pdd_transformer_timestep"]
