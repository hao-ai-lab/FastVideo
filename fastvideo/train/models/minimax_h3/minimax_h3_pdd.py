# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Parallel Decoding Distillation student (packed dual-modality latents).

Port of FastGen's ``MiniMaxH3PDD`` (``networks/MiniMaxH3/network_pdd.py``) on
top of the packed DMD adapter. This role loads the H3 transformer with
``arch_config.pdd_steps`` set, so both output projections are
:class:`fastvideo.layers.pdd.PDDReplicatedLinear` heads initialized by
repeating the released single-head weights (H3's raw projections already emit
``eps - x0``, the native PDD head direction, so no per-node fold is needed).

Time convention: the PDD fine grid lives on FastGen's base clock
``u in [0, 0.999]``; video reaches ``sigma_v = f_12(u)`` and audio
``sigma_a = f_3(u)`` through the released scheduler shifts, exactly what
``MiniMaxH3DMDModel._noise_amounts`` computes for continuous method
timesteps ``u * 1000``.
"""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.layers.pdd import PDD_GRID_MAX_T, PDDModalitySchedule, pdd_fine_grid, validate_pdd_steps
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)
from fastvideo.train.methods.trajectory_matching.pdd_core import PDDModalityBlock
from fastvideo.train.models.minimax_h3.minimax_h3 import _AUDIO_LATENT_CHANNELS, _VIDEO_LATENT_CHANNELS
from fastvideo.train.models.minimax_h3.minimax_h3_dmd import _DMD_TIMESTEP_SCALE, _FASTGEN_MAX_T, MiniMaxH3DMDModel

logger = init_logger(__name__)

assert _FASTGEN_MAX_T == PDD_GRID_MAX_T

# The mixin requires the packed H3 host contract. At runtime it remains an
# object-only cooperative mixin so Ref2V subclasses retain their place in MRO.
if TYPE_CHECKING:

    class _PDDMixinHost(MiniMaxH3DMDModel):
        pass

else:
    _PDDMixinHost = object


class MiniMaxH3PDDMixin(_PDDMixinHost):
    """Cooperative PDD heads/schedules over a target-only packed H3 adapter."""

    training_config: Any
    # Capacity padding is independent of multi-document packing. PDD's widened
    # heads, modality schedules and carried base clock currently describe one
    # real document, so do not inherit DMD's multi-document opt-in.
    _supports_cross_sample_sequence_packing = False
    # Model-boundary dtype for student and teacher forwards.
    pdd_model_dtype = torch.bfloat16

    def __init__(self, *, pdd_steps: int, **kwargs: Any) -> None:
        self.pdd_steps = validate_pdd_steps(pdd_steps)
        super().__init__(**kwargs)
        # ``update_model_arch`` lets an exported checkpoint's ``config.json``
        # override the role's ``pdd_steps``; the method must see the real grid.
        loaded = getattr(self.transformer, "pdd_steps", None)
        if loaded != self.pdd_steps:
            raise RuntimeError(f"MiniMaxH3PDDModel asked for pdd_steps={self.pdd_steps} but loaded a transformer "
                               f"with pdd_steps={loaded!r} (init_from overrides the grid through its config.json)")

    def _dit_arch_overrides(self) -> dict[str, Any]:
        """Only this role widens its heads; the shared pipeline config stays single-head."""
        return {"pdd_steps": self.pdd_steps}

    # ------------------------------------------------------------------
    # PDD schedule contract
    # ------------------------------------------------------------------

    def pdd_fine_grid(self, device: torch.device) -> torch.Tensor:
        return pdd_fine_grid(self.pdd_steps, max_t=PDD_GRID_MAX_T, device=device)

    def pdd_modality_schedules(self, batch: TrainingBatch) -> tuple[PDDModalityBlock, ...]:
        """Video (the primary clock) and audio column blocks with their shifts."""
        layout = self._batch_latent_layout(batch)
        if getattr(layout, "num_documents", 1) != 1:
            raise ValueError("H3 PDD supports one real document plus capacity padding, not multi-document packing")
        columns = dict(layout.modality_slices())
        return (
            PDDModalityBlock("video", columns["video"], PDDModalitySchedule(self.video_scheduler_shift)),
            PDDModalityBlock("audio", columns["audio"], PDDModalitySchedule(self.audio_scheduler_shift)),
        )

    def pdd_timestep(self, base: torch.Tensor) -> torch.Tensor:
        """Continuous method timestep for base time ``u``: ``u * 1000`` in float64.

        The packed adapter applies one time to the whole (single-document)
        batch, so per-sample times must agree.
        """
        base = base.to(torch.float64).reshape(-1)
        if base.numel() > 1 and bool((base != base[0]).any()):
            raise ValueError("MiniMaxH3PDDModel packs one document per batch; every sample must share a base time")
        return base * _DMD_TIMESTEP_SCALE

    # ------------------------------------------------------------------
    # Forwards
    # ------------------------------------------------------------------

    def predict_pdd_heads(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool = True,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "vsa",
        head_window: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Materialize heads ``[start, end)`` as packed ``[1, end - start, numel]`` native ``eps - x0``.

        Both widened projections are ``channel_major``, so a head window is a
        contiguous slice of each output row and is cut before unpatchify. The
        heads are returned in the state's dtype, the bf16 model boundary, as
        FastGen's network casts its outputs to the input dtype; the training
        method promotes them to its decoding precision.
        """
        start, end = (0, self.pdd_steps) if head_window is None else head_window
        if not 0 <= start < end <= self.pdd_steps:
            raise ValueError(f"head_window must satisfy 0 <= start < end <= {self.pdd_steps}, got {head_window}")
        num_heads = end - start
        noisy_video = self._bind_packed_forward_inputs(state, timestep, batch)
        (
            video_rows,
            audio_rows,
            (num_video_latents, latent_height, latent_width),
            num_audio_latents,
        ) = self._joint_forward_rows(
            noisy_video,
            batch,
            conditional=conditional,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        video_head_dim = video_rows.shape[-1] // self.pdd_steps
        audio_head_dim = audio_rows.shape[-1] // self.pdd_steps
        # The widened channel axis reads (heads, C): unpatchify with
        # ``num_heads * C`` channels, then peel the head axis off the front.
        video = unpatchify_video_tokens(
            video_rows[..., start * video_head_dim:end * video_head_dim],
            num_video_latents,
            latent_height,
            latent_width,
            num_heads * _VIDEO_LATENT_CHANNELS,
            self.transformer.patch_size,
        ).view(num_heads, _VIDEO_LATENT_CHANNELS, num_video_latents, latent_height, latent_width)
        audio = unpack_audio_tokens(audio_rows[0, :, start * audio_head_dim:end * audio_head_dim], num_audio_latents)
        audio = audio.view(MINIMAX_H3_AUDIO_CHANNELS, num_heads, _AUDIO_LATENT_CHANNELS, num_audio_latents)

        columns = dict(self._batch_latent_layout(batch).modality_slices())
        # Optional scalar-layout padding is never a head target; initialize it
        # to zero rather than exposing uninitialized storage to carried walks.
        heads = video_rows.new_zeros((num_heads, state.shape[-1]), dtype=state.dtype)
        heads[:, columns["video"]].view(num_heads, num_video_latents, _VIDEO_LATENT_CHANNELS, latent_height,
                                        latent_width).copy_(video.permute(0, 2, 1, 3, 4))
        heads[:, columns["audio"]].view(num_heads, MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_LATENT_CHANNELS,
                                        num_audio_latents).copy_(audio.permute(1, 0, 2, 3))
        return heads.neg_()[None]

    def predict_pdd_fused(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        start: int,
        end: int,
        fine_grid: torch.Tensor,
        batch: TrainingBatch,
        *,
        precision_decoding: torch.dtype,
        conditional: bool = True,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "vsa",
    ) -> torch.Tensor:
        """One forward whose heads are fused over ``[start, end)``: packed mean ``eps - x0``."""
        integration_weights = {
            block.name: block.schedule.integration_weights(fine_grid)
            for block in self.pdd_modality_schedules(batch)
        }
        with self.transformer.fuse_pdd_block(start, end, integration_weights, precision_decoding):
            return self.predict_noise(
                state,
                timestep,
                batch,
                conditional=conditional,
                cfg_uncond=cfg_uncond,
                attn_kind=attn_kind,
            )


class MiniMaxH3PDDModel(MiniMaxH3PDDMixin, MiniMaxH3DMDModel):
    """PDD-widened T2VA role for PDD, PDDMD, and DMPDD."""


__all__ = ["MiniMaxH3PDDMixin", "MiniMaxH3PDDModel"]
