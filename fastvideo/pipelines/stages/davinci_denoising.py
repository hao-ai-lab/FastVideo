# SPDX-License-Identifier: Apache-2.0
"""Denoising stage for daVinci-MagiHuman.

daVinci is timestep-free: the DiT receives no timestep input.
Flow-matching scheduler (shift=5.0) runs externally in this stage.
Dynamic CFG: guidance_scale when t > 500, else 2.0.
Audio fallback: torch.randn when no audio is provided (T2V mode).
"""
from __future__ import annotations

import weakref
from typing import Optional

import torch
from tqdm.auto import tqdm

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

logger = init_logger(__name__)

# daVinci architecture constants (mirror DaVinciMagiHumanArchConfig defaults)
_PATCH_SIZE = (1, 2, 2)  # (temporal, height, width)
_VIDEO_IN_CHANNELS = 192  # z_dim=48 * patch 1×2×2
_AUDIO_IN_CHANNELS = 64
_TEXT_IN_CHANNELS = 3584  # T5Gemma-9B hidden dim
_Z_DIM = 48
# Audio token count for T2V fallback (no audio conditioning).
# Roughly 256 audio frames ≈ ~10s at typical audio latent fps.
_DEFAULT_AUDIO_TOKENS = 256
# Flow matching schedule
_FLOW_SHIFT = 5.0
# Dynamic CFG threshold (daVinci paper, Section 3.4)
_CFG_THRESHOLD = 500.0


# ---------------------------------------------------------------------------
# Patchify / unpatchify helpers
# ---------------------------------------------------------------------------

def _patchify(
    latents: torch.Tensor,
    patch_size: tuple[int, int, int] = _PATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Convert (B, C, T, H, W) latents to packed tokens + 9-dim coords.

    Returns:
        tokens:         (B*Tp*Hp*Wp, C*pt*ph*pw)
        coords_9:       (B*Tp*Hp*Wp, 9) — (t,h,w, Tp,Hp,Wp, Tp,Hp,Wp)
        shape:          (B, Tp, Hp, Wp) for unpatchify
    """
    B, C, T, H, W = latents.shape
    pt, ph, pw = patch_size
    Tp, Hp, Wp = T // pt, H // ph, W // pw

    x = latents.reshape(B, C, Tp, pt, Hp, ph, Wp, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, Tp, Hp, Wp, C, pt, ph, pw)
    tokens = x.flatten(4).reshape(B * Tp * Hp * Wp, -1)  # (N_v, 192)

    # Build (t, h, w) grid
    t_idx = torch.arange(Tp, device=latents.device, dtype=torch.float32)
    h_idx = torch.arange(Hp, device=latents.device, dtype=torch.float32)
    w_idx = torch.arange(Wp, device=latents.device, dtype=torch.float32)
    grid = torch.stack(
        torch.meshgrid(t_idx, h_idx, w_idx, indexing="ij"), dim=-1
    ).reshape(-1, 3)  # (Tp*Hp*Wp, 3)
    grid = grid.repeat(B, 1)  # (B*Tp*Hp*Wp, 3)

    # 9-dim: (t,h,w, Tp,Hp,Wp, ref_Tp,ref_Hp,ref_Wp) — refs = sizes for T2V
    sizes = torch.tensor(
        [Tp, Hp, Wp], device=latents.device, dtype=torch.float32
    ).unsqueeze(0).expand(B * Tp * Hp * Wp, -1)
    coords_9 = torch.cat([grid, sizes, sizes], dim=-1)

    return tokens, coords_9, (B, Tp, Hp, Wp)


def _unpatchify(
    video_velocity: torch.Tensor,
    shape: tuple,
    patch_size: tuple[int, int, int] = _PATCH_SIZE,
    z_dim: int = _Z_DIM,
) -> torch.Tensor:
    """Inverse of _patchify.  Returns (B, C, T, H, W)."""
    B, Tp, Hp, Wp = shape
    pt, ph, pw = patch_size
    x = video_velocity.reshape(B, Tp, Hp, Wp, z_dim, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(B, z_dim, Tp * pt, Hp * ph, Wp * pw)


def _build_text_coords(
    n_t: int, device: torch.device
) -> torch.Tensor:
    """9-dim coords for text tokens.

    Text uses negative t-offsets so positions never collide with video.
    coords: (i - n_t, 0, 0,  n_t, 1, 1,  1, 1, 1)
    """
    indices = torch.arange(n_t, device=device, dtype=torch.float32)
    xyz = torch.stack(
        [indices - n_t,
         torch.zeros(n_t, device=device),
         torch.zeros(n_t, device=device)], dim=-1
    )
    sizes = torch.tensor([n_t, 1, 1], device=device, dtype=torch.float32
                         ).unsqueeze(0).expand(n_t, -1)
    refs = torch.ones(n_t, 3, device=device, dtype=torch.float32)
    return torch.cat([xyz, sizes, refs], dim=-1)  # (n_t, 9)


def _build_audio_coords(
    n_a: int, device: torch.device
) -> torch.Tensor:
    """9-dim coords for audio tokens (v2-style ref)."""
    ref_t = float((n_a - 1) // 4 + 1)
    indices = torch.arange(n_a, device=device, dtype=torch.float32)
    xyz = torch.stack(
        [indices,
         torch.zeros(n_a, device=device),
         torch.zeros(n_a, device=device)], dim=-1
    )
    sizes = torch.tensor([n_a, 1, 1], device=device, dtype=torch.float32
                         ).unsqueeze(0).expand(n_a, -1)
    refs = torch.tensor([ref_t, 1.0, 1.0], device=device, dtype=torch.float32
                        ).unsqueeze(0).expand(n_a, -1)
    return torch.cat([xyz, sizes, refs], dim=-1)  # (n_a, 9)


def _pack_tokens(
    video_tokens: torch.Tensor,    # (N_v, 192)
    video_coords: torch.Tensor,    # (N_v, 9)
    audio_tokens: torch.Tensor,    # (N_a, 64)
    audio_coords: torch.Tensor,    # (N_a, 9)
    text_tokens: torch.Tensor,     # (N_t, 3584)
    text_coords: torch.Tensor,     # (N_t, 9)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Pack all modalities into the format expected by DaVinciMagiHuman.forward.

    Token order: video | audio | text  (matches MODALITY_VIDEO=0, AUDIO=1, TEXT=2)

    Returns:
        hidden_states: (S, max_in_channels)   — max=text_in=3584
        coords_mapping: (S, 9)
        modality_mapping: (S,) int
        n_v, n_a, n_t: token counts per modality
    """
    device = video_tokens.device
    n_v, n_a, n_t = (video_tokens.shape[0], audio_tokens.shape[0],
                     text_tokens.shape[0])
    S = n_v + n_a + n_t
    max_in = _TEXT_IN_CHANNELS  # 3584

    hidden = torch.zeros(S, max_in, device=device, dtype=video_tokens.dtype)
    hidden[:n_v, :_VIDEO_IN_CHANNELS] = video_tokens
    hidden[n_v:n_v + n_a, :_AUDIO_IN_CHANNELS] = audio_tokens
    hidden[n_v + n_a:, :_TEXT_IN_CHANNELS] = text_tokens.to(video_tokens.dtype)

    coords = torch.cat([video_coords, audio_coords, text_coords], dim=0)

    mod = torch.cat([
        torch.zeros(n_v, dtype=torch.long, device=device),   # video
        torch.ones(n_a, dtype=torch.long, device=device),    # audio
        torch.full((n_t,), 2, dtype=torch.long, device=device),  # text
    ])
    return hidden, coords, mod, n_v, n_a, n_t


# ---------------------------------------------------------------------------
# Denoising stage
# ---------------------------------------------------------------------------

class DaVinciDenoisingStage(PipelineStage):
    """Flow-matching denoising for daVinci-MagiHuman.

    Differences from standard DenoisingStage:
    - No timestep is passed to the DiT (timestep-free architecture).
    - Tokens are packed as video | audio | text before each forward pass.
    - Dynamic CFG: guidance_scale when t > CFG_THRESHOLD, else 2.0.
    - Audio tokens are random noise when no audio input is provided (T2V).
    """

    def __init__(
        self,
        transformer,
        scheduler: Optional[FlowMatchEulerDiscreteScheduler] = None,
        pipeline=None,
        n_audio_tokens: int = _DEFAULT_AUDIO_TOKENS,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler or FlowMatchEulerDiscreteScheduler(
            shift=_FLOW_SHIFT)
        self.n_audio_tokens = n_audio_tokens
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    # ------------------------------------------------------------------
    # PipelineStage.forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        latents = batch.latents
        if latents is None:
            raise ValueError("latents must be set before DaVinciDenoisingStage")

        # -- dtype --
        if hasattr(self.transformer, "module"):
            target_dtype = next(
                self.transformer.module.parameters()).dtype
        else:
            target_dtype = next(self.transformer.parameters()).dtype
        autocast_enabled = (target_dtype != torch.float32 and
                            not fastvideo_args.disable_autocast)

        device = latents.device
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # -- Text embeddings: (B, N_t, 3584) --
        # prompt_embeds is a list; index 0 = conditional embedding
        prompt_embeds = batch.prompt_embeds[0].to(device, dtype=target_dtype)
        B = latents.shape[0]
        # Flatten batch dim: (B*N_t, 3584). Inference typically B=1.
        text_tokens = prompt_embeds.reshape(-1, _TEXT_IN_CHANNELS)
        n_t = text_tokens.shape[0]
        text_coords = _build_text_coords(n_t, device)

        # Null text for CFG uncond pass: zeros of the same shape
        do_cfg = (batch.do_classifier_free_guidance and
                  guidance_scale > 1.0)
        null_text_tokens = torch.zeros_like(text_tokens)

        # -- Negative prompt embeds for CFG --
        if do_cfg and batch.negative_prompt_embeds:
            neg_embeds = batch.negative_prompt_embeds[0].to(
                device, dtype=target_dtype)
            null_text_tokens = neg_embeds.reshape(-1, _TEXT_IN_CHANNELS)

        # Build coords for null_text separately — negative prompt may have a
        # different token count than the positive prompt (longer neg prompts
        # are common), so reusing text_coords here causes coords/mod mismatch.
        null_text_coords = _build_text_coords(null_text_tokens.shape[0], device)

        # -- Audio latents (initialise as noise; denoised in parallel) --
        n_a = self.n_audio_tokens
        audio_latents = torch.randn(
            n_a, _AUDIO_IN_CHANNELS, device=device, dtype=target_dtype)
        audio_coords = _build_audio_coords(n_a, device)

        # -- Extra step kwargs for scheduler --
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": batch.generator})

        with self.progress_bar(total=num_inference_steps) as pbar:
            for i, t in enumerate(timesteps):
                # ---- Patchify current noisy video latents ----
                vid_tokens, vid_coords, patchify_shape = _patchify(
                    latents.to(target_dtype))

                # ---- Conditional forward ----
                with (
                    set_forward_context(current_timestep=int(t),
                                        attn_metadata=None,
                                        forward_batch=batch),
                    torch.autocast(device_type="cuda",
                                   dtype=target_dtype,
                                   enabled=autocast_enabled),
                ):
                    cond_out = self._forward_packed(
                        vid_tokens, vid_coords, audio_latents, audio_coords,
                        text_tokens, text_coords, target_dtype)

                    if do_cfg:
                        uncond_out = self._forward_packed(
                            vid_tokens, vid_coords, audio_latents,
                            audio_coords, null_text_tokens, null_text_coords,
                            target_dtype)

                        # Dynamic CFG threshold (daVinci paper §3.4)
                        eff_scale = (guidance_scale
                                     if float(t) > _CFG_THRESHOLD else 2.0)
                        n_v = vid_tokens.shape[0]
                        # Apply guidance only to video tokens
                        cond_vid = cond_out[:n_v, :_VIDEO_IN_CHANNELS]
                        uncond_vid = uncond_out[:n_v, :_VIDEO_IN_CHANNELS]
                        guided_vid = (uncond_vid +
                                      eff_scale * (cond_vid - uncond_vid))
                        # Reconstruct output with guided video portion
                        cond_out = cond_out.clone()
                        cond_out[:n_v, :_VIDEO_IN_CHANNELS] = guided_vid

                # ---- Extract video velocity and unpatchify ----
                n_v = vid_tokens.shape[0]
                vid_velocity = cond_out[:n_v, :_VIDEO_IN_CHANNELS]
                velocity = _unpatchify(vid_velocity, patchify_shape)

                # ---- Scheduler step ----
                # Capture dt BEFORE video step increments step_index.
                # Both video and audio share one scheduler; calling step()
                # twice per iteration would double-increment step_index and
                # exhaust sigmas after N/2 iterations instead of N.
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)
                s_idx = self.scheduler.step_index
                _sigmas = self.scheduler.sigmas
                _is_last = (s_idx + 1 >= len(_sigmas))
                _sigma_next = (
                    _sigmas[0].new_zeros(()) if _is_last
                    else _sigmas[s_idx + 1])
                _dt = _sigma_next - _sigmas[s_idx]

                latents = self.scheduler.step(
                    velocity, t, latents,
                    **extra_step_kwargs, return_dict=False)[0]

                # ---- Update audio latents in-place (manual Euler) ----
                # Avoids a second scheduler.step() call that would
                # double-increment step_index. Skip at terminal sigma.
                if not _is_last:
                    audio_velocity = cond_out[
                        n_v:n_v + n_a, :_AUDIO_IN_CHANNELS]
                    audio_latents = (
                        audio_latents + _dt * audio_velocity
                    ).to(target_dtype)

                pbar.update()

        batch.latents = latents
        return batch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward_packed(
        self,
        vid_tokens: torch.Tensor,
        vid_coords: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_coords: torch.Tensor,
        text_tokens: torch.Tensor,
        text_coords: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pack tokens and call the DiT once.  Returns the raw model output [S, 192]."""
        hidden, coords, mod, n_v, n_a, n_t = _pack_tokens(
            vid_tokens, vid_coords,
            audio_tokens, audio_coords,
            text_tokens, text_coords,
        )
        out = self.transformer(
            hidden_states=hidden.to(dtype),
            encoder_hidden_states=None,  # unused by daVinci
            timestep=None,               # timestep-free
            coords_mapping=coords,
            modality_mapping=mod,
        )
        return out

    def prepare_extra_func_kwargs(self, func, kwargs):
        import inspect
        accepted = set(inspect.signature(func).parameters)
        return {k: v for k, v in kwargs.items() if k in accepted}

    def progress_bar(self, total):
        return tqdm(total=total, desc="DaVinci denoising")
