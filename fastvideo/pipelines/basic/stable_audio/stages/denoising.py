# SPDX-License-Identifier: Apache-2.0
"""Stable Audio denoising — k-diffusion `dpmpp-3m-sde` over the native DiT.

The official upstream `generate_diffusion_cond` reduces to:

    denoiser = K.external.VDenoiser(model.model)         # v-prediction wrapper
    sigmas   = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho)
    x        = noise * sigmas[0]
    sampled  = K.sampling.sample_dpmpp_3m_sde(
        denoiser, x, sigmas, extra_args={... cross_attn_cond, global_cond, cfg_scale, ...},
    )

We do exactly the same here, using a thin `_DiTAdapter` to expose our
`StableAudioDiT` to `K.external.VDenoiser` with the kwargs it expects
(CFG batching is done inside the adapter, not by the DiT itself).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class _DiTAdapter(nn.Module):
    """Minimal wrapper exposing `StableAudioDiT` to `VDenoiser` with the
    same conditioning + CFG semantics as upstream `DiTWrapper.forward`.
    """

    def __init__(self, dit) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        cross_attn_cond: torch.Tensor,
        cross_attn_cond_mask: torch.Tensor | None = None,
        negative_cross_attn_cond: torch.Tensor | None = None,
        negative_cross_attn_mask: torch.Tensor | None = None,
        global_cond: torch.Tensor,
        negative_global_cond: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        **_unused,
    ) -> torch.Tensor:
        if cfg_scale == 1.0:
            return self.dit(x, t, cross_attn_cond=cross_attn_cond, global_embed=global_cond)

        # CFG batching — replicate upstream `DiffusionTransformer.forward`'s
        # batch-CFG branch.
        batch_inputs = torch.cat([x, x], dim=0)
        batch_t = torch.cat([t, t], dim=0)
        batch_global = torch.cat([global_cond, global_cond if negative_global_cond is None else negative_global_cond],
                                 dim=0)

        null_embed = torch.zeros_like(cross_attn_cond)
        if negative_cross_attn_cond is not None:
            if negative_cross_attn_mask is not None:
                neg_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                negative_cross_attn_cond = torch.where(neg_mask, negative_cross_attn_cond, null_embed)
            batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
        else:
            batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

        batch_out = self.dit(batch_inputs, batch_t, cross_attn_cond=batch_cond, global_embed=batch_global)
        cond_out, uncond_out = torch.chunk(batch_out, 2, dim=0)
        return uncond_out + (cond_out - uncond_out) * cfg_scale


class StableAudioDenoisingStage(PipelineStage):
    """k-diffusion `dpmpp-3m-sde` sampling loop."""

    # Defaults pulled from the official `generate_diffusion_cond` call site
    # used by the Stability AI demo and the official inference scripts.
    _SIGMA_MIN = 0.3
    _SIGMA_MAX = 500.0
    _RHO = 1.0

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def verify_input(self, batch, fastvideo_args):
        return VerificationResult()

    def verify_output(self, batch, fastvideo_args):
        return VerificationResult()

    @torch.inference_mode()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        pc = fastvideo_args.pipeline_config
        ext = batch.extra
        device = batch.latents.device
        guidance_scale = float(batch.guidance_scale or pc.guidance_scale)
        steps = int(batch.num_inference_steps)

        # k-diffusion is a sampling library, not a model class — using its
        # dpmpp-3m-sde sampler matches the official upstream byte-for-byte.
        import k_diffusion as K
        adapter = _DiTAdapter(self.transformer)
        denoiser = K.external.VDenoiser(adapter)

        sigmas = K.sampling.get_sigmas_polyexponential(steps,
                                                       self._SIGMA_MIN,
                                                       self._SIGMA_MAX,
                                                       self._RHO,
                                                       device=device)
        # Upstream scales the initial noise by sigmas[0] inside `sample_k`.
        x = batch.latents * sigmas[0]

        extra_args = {
            "cross_attn_cond": ext["cross_attn_cond"],
            "cross_attn_cond_mask": ext.get("cross_attn_mask"),
            "global_cond": ext["global_embed"],
            "cfg_scale": guidance_scale,
        }
        if ext.get("negative_cross_attn_cond") is not None:
            extra_args["negative_cross_attn_cond"] = ext["negative_cross_attn_cond"]
            extra_args["negative_cross_attn_mask"] = ext["negative_cross_attn_mask"]
        if ext.get("negative_global_embed") is not None:
            extra_args["negative_global_cond"] = ext["negative_global_embed"]

        # `LocalAttention` (used inside our native StableAudioDiT) needs an
        # active forward context to read `attn_metadata`. Wrap the whole
        # sampling loop — every DiT call sees `attn_metadata=None`, which is
        # what the SDPA / FlashAttn backends accept for unconditioned compute.
        with set_forward_context(current_timestep=0, attn_metadata=None):
            sampled = K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, extra_args=extra_args)
        batch.latents = sampled
        return batch
