# SPDX-License-Identifier: Apache-2.0
"""Track-conditioned (MotionStream-style) bidirectional Wan DiT.

A thin subclass of ``WanTransformer3DModel``. Unlike MatrixGame (which inherits ``BaseDiT`` and
reimplements the Wan forward because actions modulate hidden states inside every block), track
conditioning is a channel-concat at the patch embed, so the transformer blocks are untouched.
The only additions are:
  - a ``TrackEncoder`` that turns sparse point tracks into a latent-aligned conditioning map, and
  - concatenating that map onto the latent before ``super().forward`` patchifies it.

Because the blocks are unchanged, the same ``TrackEncoder`` can later drop into a causal Wan
variant for Stage-2+ training.

Loading note: this expects the patch-embed input channels to be widened to
``num_channels_latents + track_channels``. Initialize from a bidirectional Wan teacher with a
small conversion step that zero-pads the new patch-embed input channels (see the trackwan config
and the eventual checkpoint-conversion script); ``TrackEncoder``'s final conv is zero-init so
step-0 behavior matches the teacher.
"""
from __future__ import annotations

from typing import Any

import torch

from fastvideo.configs.models.dits.trackwan import TrackWanVideoConfig
from fastvideo.models.dits.trackwan.track_encoder import TrackEncoder
from fastvideo.models.dits.wanvideo import WanTransformer3DModel


class TrackWanTransformer3DModel(WanTransformer3DModel):
    """Bidirectional Wan DiT with MotionStream point-track conditioning."""

    # Marker for track/motion input support (mirrors MatrixGame's ``supports_action_input``).
    supports_track_input = True

    def __init__(self, config: TrackWanVideoConfig, hf_config: dict[str, Any]) -> None:
        # Builds patch_embedding with in_chans = config.in_channels (already widened to
        # num_channels_latents + track_channels).
        super().__init__(config=config, hf_config=hf_config)

        # ``track_config`` is a FastVideo-only field. When the checkpoint loads
        # through a plain ``WanVideoArchConfig`` (which has no such field) it is
        # dropped from ``config``, so fall back to the raw ``hf_config``
        # (the diffusers config.json, which carries it verbatim).
        track_config = dict(getattr(config, "track_config", None) or {})
        if not track_config and isinstance(hf_config, dict):
            track_config = dict(hf_config.get("track_config", {}) or {})
        # Prefer an explicit track_channels (I2V widens the non-track input to
        # 36ch, so deriving from in_channels - num_channels_latents is wrong);
        # fall back to the T2V derivation when unset.
        self.track_channels = int(track_config.get("track_channels")
                                  or (config.in_channels - config.num_channels_latents))
        assert self.track_channels > 0, (
            f"TrackWan needs a positive track_channels; got {self.track_channels} "
            f"(in_channels={config.in_channels}, num_channels_latents={config.num_channels_latents}).")

        self.track_encoder = TrackEncoder(
            id_dim=track_config.get("id_dim", 128),
            track_channels=self.track_channels,
            vae_spatial_compression=track_config.get("vae_spatial_compression", 8),
            vae_temporal_compression=track_config.get("vae_temporal_compression", 4),
            max_track_id=track_config.get("max_track_id", 100_000),
            zero_init=track_config.get("zero_init_head", True),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                guidance=None,
                track_points: torch.Tensor | None = None,
                track_visibility: torch.Tensor | None = None,
                track_ids: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        """``track_points`` (B,T,N,2 normalized) + ``track_visibility`` (B,T,N).

        When tracks are absent (no-track inference, or motion-CFG unconditional branch), the track
        channels are zeros -- which is also exactly the p_mask temporal-masking augmentation.
        """
        batch_size, _, latent_t, latent_h, latent_w = hidden_states.shape
        if track_points is not None:
            cm = self.track_encoder(track_points, track_visibility, latent_t, latent_h, latent_w,
                                    track_ids=track_ids)
            cm = cm.to(hidden_states.dtype)
        else:
            cm = hidden_states.new_zeros((batch_size, self.track_channels, latent_t, latent_h, latent_w))

        hidden_states = torch.cat([hidden_states, cm], dim=1)
        return super().forward(hidden_states,
                               encoder_hidden_states,
                               timestep,
                               encoder_hidden_states_image=encoder_hidden_states_image,
                               guidance=guidance,
                               **kwargs)
