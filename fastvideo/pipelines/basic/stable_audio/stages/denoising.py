# SPDX-License-Identifier: Apache-2.0
"""Stable Audio denoising — k-diffusion `dpmpp-3m-sde` over the DiT.

CFG-batched conditioning is built once outside the sampler loop so the
adapter only does `cat([x, x])` + DiT call per step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import VerificationResult


class _DiTAdapter(nn.Module):
    """`StableAudioDiT` -> `K.external.VDenoiser` adapter.

    `batch_cond` / `batch_global` are precomputed CFG-batched tensors
    (`[2, ...]` for CFG, `[1, ...]` otherwise); building them once
    outside the sampler loop saves ~3 cats × 100 steps per call.
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

    # Sampler defaults from the published model card.
    _SIGMA_MIN = 0.3
    _SIGMA_MAX = 500.0
    _RHO = 1.0
    _LOG_SIGMA_MIN = math.log(_SIGMA_MIN)
    _LOG_SIGMA_MAX = math.log(_SIGMA_MAX)

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    def _resolve_sigma_max(self, batch) -> float:
        """Map A2A intent to `sigma_max`.

        Public knob is `init_audio_strength` (0..1, higher = closer to
        source), log-interpolated between SIGMA_MIN (= preservation) and
        SIGMA_MAX (= full T2A). Raw `init_noise_level` is the legacy
        sigma_max override; passing both is an error.
        """
        raw = getattr(batch, "init_noise_level", None)
        strength = getattr(batch, "init_audio_strength", None)
        if raw is not None and strength is not None:
            raise ValueError("Pass `init_audio_strength` (0..1) OR `init_noise_level` "
                             "(raw sigma_max), not both.")
        if raw is not None:
            return float(raw)
        s = max(0.0, min(1.0, float(strength) if strength is not None else 0.6))
        return float(math.exp(self._LOG_SIGMA_MAX - s * (self._LOG_SIGMA_MAX - self._LOG_SIGMA_MIN)))

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

        init_latent = ext.get("init_latent")
        sigma_max = self._resolve_sigma_max(batch) if init_latent is not None else self._SIGMA_MAX

        sigmas = K.sampling.get_sigmas_polyexponential(steps, self._SIGMA_MIN, sigma_max, self._RHO, device=device)
        # Cast noise + conditioning to the DiT's dtype before sampling
        # (matches `stable_audio_tools/inference/generation.py:185-187`).
        model_dtype = next(self.transformer.parameters()).dtype

        def _cast(t: torch.Tensor | None) -> torch.Tensor | None:
            return t.to(model_dtype) if t is not None else None

        x = (batch.latents * sigmas[0]).to(model_dtype)
        if init_latent is not None:
            x = x + init_latent.to(model_dtype)

        batch_cond, batch_global = _build_cfg_conditioning(
            cross_attn_cond=ext["cross_attn_cond"].to(model_dtype),
            global_embed=ext["global_embed"].to(model_dtype),
            negative_cross_attn_cond=_cast(ext.get("negative_cross_attn_cond")),
            negative_cross_attn_mask=ext.get("negative_cross_attn_mask"),
            negative_global_embed=_cast(ext.get("negative_global_embed")),
            do_cfg=guidance_scale != 1.0,
        )
        adapter = _DiTAdapter(self.transformer,
                              batch_cond=batch_cond,
                              batch_global=batch_global,
                              cfg_scale=guidance_scale)
        denoiser = K.external.VDenoiser(adapter)

        # RePaint blending hook — works on any v-prediction model, no
        # inpaint-trained checkpoint needed.
        inpaint_mask = ext.get("inpaint_mask_latent")
        inpaint_ref = ext.get("inpaint_reference_latent")
        if inpaint_mask is not None and inpaint_ref is not None:
            inpaint_mask = inpaint_mask.to(model_dtype)
            inpaint_ref = inpaint_ref.to(model_dtype)
            callback = _make_inpaint_callback(inpaint_ref, inpaint_mask, sigmas)
        else:
            callback = None

        # `LocalAttention` (in `StableAudioDiT`) reads `get_forward_context()`
        # for `attn_metadata`; wrap the whole loop.
        with set_forward_context(current_timestep=0, attn_metadata=None):
            sampled = K.sampling.sample_dpmpp_3m_sde(denoiser,
                                                     x,
                                                     sigmas,
                                                     disable=False,
                                                     extra_args={},
                                                     callback=callback)

        # Final blend so the kept region of the inpaint reference is exact.
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
    """Build the CFG-batched `(cond, global)` tensors once.

    Cond ordering is `[conditioned, unconditioned]` (the adapter splits
    with the same convention). Masked negative cond is zero-filled where
    `mask == 0`.
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
    """RePaint blending callback for the k-diffusion sampler.

    At every step, replaces the kept region (`mask == 1`) of the in-
    flight latent with the reference re-noised to the next sigma —
    pulls the kept region back onto the trajectory the model expects,
    so RePaint-style inpainting converges on non-inpaint-trained models.

    Pre-allocates the noise buffer so the ~100 sampler steps don't churn
    ~25 MB of fresh allocations per call.
    """
    noise_buf = torch.empty_like(reference_latent)
    inv_mask = 1 - mask

    def cb(info: dict) -> None:
        i = int(info["i"])
        next_i = min(i + 1, len(sigmas) - 1)
        sigma_next = float(sigmas[next_i])
        noise_buf.normal_()
        # `state["x"]` is the live latent; the dpmpp-3m-sde sampler picks
        # up our in-place mutation between steps (verified against
        # k_diffusion 0.1.1.post1).
        x = info["x"]
        x.copy_((reference_latent + noise_buf * sigma_next) * mask + x * inv_mask)

    return cb
