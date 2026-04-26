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
`StableAudioDiT` to `K.external.VDenoiser` with the kwargs it expects.
CFG batching is precomputed in `forward()` so the adapter only has to
do the per-step `cat([x, x])` + DiT call + chunk + lerp; the constant
conditioning tensors are built once outside the sampler loop.
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
    """Wraps `StableAudioDiT` for `K.external.VDenoiser`.

    `batch_cond` and `batch_global` are the already-CFG-batched
    conditioning tensors (shape `[2, ...]` for CFG, `[1, ...]` for
    unconditional); building them once outside the sampler loop saves
    ~3 cats × 100 steps per call.
    """

    def __init__(self, dit, *, batch_cond: torch.Tensor, batch_global: torch.Tensor, cfg_scale: float) -> None:
        super().__init__()
        self.dit = dit
        self.batch_cond = batch_cond
        self.batch_global = batch_global
        self.cfg_scale = cfg_scale
        self.do_cfg = cfg_scale != 1.0

    def forward(self, x: torch.Tensor, t: torch.Tensor, **_unused) -> torch.Tensor:
        if not self.do_cfg:
            return self.dit(x, t, cross_attn_cond=self.batch_cond, global_embed=self.batch_global)
        batch_x = torch.cat([x, x], dim=0)
        batch_t = torch.cat([t, t], dim=0)
        out = self.dit(batch_x, batch_t, cross_attn_cond=self.batch_cond, global_embed=self.batch_global)
        cond_out, uncond_out = torch.chunk(out, 2, dim=0)
        return uncond_out + (cond_out - uncond_out) * self.cfg_scale


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

        import k_diffusion as K

        # If the user gave us an init latent (audio-to-audio variation),
        # the sampler should start at `sigma_max = init_noise_level` — see
        # upstream `generate_diffusion_cond`'s `sampler_kwargs` override.
        init_latent = ext.get("init_latent")
        sigma_max = (float(getattr(batch, "init_noise_level", None) or self._SIGMA_MAX)
                     if init_latent is not None else self._SIGMA_MAX)

        sigmas = K.sampling.get_sigmas_polyexponential(steps, self._SIGMA_MIN, sigma_max, self._RHO, device=device)
        # Upstream scales the initial noise by sigmas[0] inside `sample_k`,
        # then adds `init_data` if provided.
        x = batch.latents * sigmas[0]
        if init_latent is not None:
            x = x + init_latent

        # Precompute the CFG-batched conditioning once. The adapter then
        # only has to do `cat([x, x])` + DiT call per step.
        batch_cond, batch_global = _build_cfg_conditioning(
            cross_attn_cond=ext["cross_attn_cond"],
            global_embed=ext["global_embed"],
            negative_cross_attn_cond=ext.get("negative_cross_attn_cond"),
            negative_cross_attn_mask=ext.get("negative_cross_attn_mask"),
            negative_global_embed=ext.get("negative_global_embed"),
            do_cfg=guidance_scale != 1.0,
        )
        adapter = _DiTAdapter(self.transformer,
                              batch_cond=batch_cond,
                              batch_global=batch_global,
                              cfg_scale=guidance_scale)
        denoiser = K.external.VDenoiser(adapter)

        # RePaint-style inpainting hook (works on any v-prediction model;
        # SA Open 1.0 isn't inpaint-trained so this is the only path).
        inpaint_mask = ext.get("inpaint_mask_latent")
        inpaint_ref = ext.get("inpaint_reference_latent")
        callback = (_make_inpaint_callback(inpaint_ref, inpaint_mask, sigmas)
                    if inpaint_mask is not None and inpaint_ref is not None else None)

        # `LocalAttention` (used inside our native StableAudioDiT) reads
        # `get_forward_context()` for `attn_metadata`. Wrap the whole loop.
        with set_forward_context(current_timestep=0, attn_metadata=None):
            sampled = K.sampling.sample_dpmpp_3m_sde(denoiser,
                                                     x,
                                                     sigmas,
                                                     disable=False,
                                                     extra_args={},
                                                     callback=callback)

        # Final blend: kept region of the inpaint reference is exact-equal.
        if inpaint_mask is not None and inpaint_ref is not None:
            sampled = inpaint_ref * inpaint_mask + sampled * (1 - inpaint_mask)
        batch.latents = sampled
        return batch


def _build_cfg_conditioning(
    *,
    cross_attn_cond: torch.Tensor,
    global_embed: torch.Tensor,
    negative_cross_attn_cond: torch.Tensor | None,
    negative_cross_attn_mask: torch.Tensor | None,
    negative_global_embed: torch.Tensor | None,
    do_cfg: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the CFG-batched (cond, global) tensors once.

    Mirrors upstream `DiffusionTransformer.forward`'s batch-CFG branch:
      * cond ordering = [conditioned, unconditioned] (the adapter splits
        with the same convention)
      * masked negative cond is zero-filled where mask == 0 (matches
        `prepare negative cross_attn_cond` in upstream)
    """
    if not do_cfg:
        return cross_attn_cond, global_embed
    if negative_cross_attn_cond is not None:
        if negative_cross_attn_mask is not None:
            neg_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
            null_embed = torch.zeros_like(cross_attn_cond)
            negative_cross_attn_cond = torch.where(neg_mask, negative_cross_attn_cond, null_embed)
        batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)
    else:
        batch_cond = torch.cat([cross_attn_cond, torch.zeros_like(cross_attn_cond)], dim=0)
    other_global = global_embed if negative_global_embed is None else negative_global_embed
    batch_global = torch.cat([global_embed, other_global], dim=0)
    return batch_cond, batch_global


def _make_inpaint_callback(reference_latent: torch.Tensor, mask: torch.Tensor, sigmas: torch.Tensor):
    """Build a k-diffusion sampler callback that performs RePaint blending
    after every step.

    Pre-allocates the noise + working buffers so the ~100 sampler steps
    don't churn 25MB of fresh allocations per call.

    At step `i`, replaces the kept region (mask == 1) of the in-flight
    latent with the reference re-noised to `sigmas[i + 1]`. This pulls
    the kept region back onto the trajectory the model expects, which
    is what makes RePaint-style inpainting converge on non-inpaint-
    trained models.
    """
    noise_buf = torch.empty_like(reference_latent)
    inv_mask = 1 - mask

    def cb(info: dict) -> None:
        i = int(info["i"])
        next_i = min(i + 1, len(sigmas) - 1)
        sigma_next = float(sigmas[next_i])
        noise_buf.normal_()
        # Blend in-place — k_diffusion's dpmpp-3m-sde sampler reads
        # `state["x"]` between steps, so mutating it via copy_() carries
        # forward (verified against k_diffusion 0.1.1.post1).
        x = info["x"]
        x.copy_((reference_latent + noise_buf * sigma_next) * mask + x * inv_mask)

    return cb
