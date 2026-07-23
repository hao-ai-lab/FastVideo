# SPDX-License-Identifier: Apache-2.0
"""Bidirectional and causal track-conditioned Wan transformers."""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.configs.models.dits.trackwan import TrackWanVideoConfig
from fastvideo.models.dits.causal_wanvideo import CausalWanTransformer3DModel
from fastvideo.models.dits.trackwan.track_encoder import TrackEncoder
from fastvideo.models.dits.wanvideo import WanTransformer3DModel


class _TrackConditioningMixin:
    supports_track_input = True
    track_channels: int
    track_encoder: TrackEncoder

    def _init_track_conditioning(
        self,
        config: TrackWanVideoConfig,
        hf_config: dict[str, Any],
    ) -> None:
        track_config = dict(getattr(config, "track_config", None) or {})
        # Checkpoint config is authoritative for checkpoint-visible TrackEncoder
        # shapes and legacy bias tensors. Defaults only fill missing fields.
        if isinstance(hf_config, dict):
            track_config.update(dict(hf_config.get("track_config", {}) or {}))

        self.track_channels = int(
            track_config.get("track_channels", self.in_channels - 36))
        base_channels = self.in_channels - self.track_channels
        expected_base_channels = self.num_channels_latents + 20
        if self.track_channels <= 0:
            raise ValueError("TrackWan requires positive track_channels")
        if base_channels != expected_base_channels:
            raise ValueError(
                "TrackWan expects noisy latent + 20 I2V channels before "
                f"track conditioning; got {base_channels}, expected "
                f"{expected_base_channels}")

        self.track_encoder = TrackEncoder(
            id_dim=int(track_config.get("id_dim", 64)),
            track_channels=self.track_channels,
            vae_spatial_compression=int(
                track_config.get("vae_spatial_compression", 8)),
            vae_temporal_compression=int(
                track_config.get("vae_temporal_compression", 4)),
            max_track_id=int(track_config.get("max_track_id", 100_000)),
            zero_init=bool(track_config.get("zero_init_head", False)),
            use_bias=bool(track_config.get("use_bias", False)),
        )

    def _append_track_conditioning(
        self,
        hidden_states: torch.Tensor,
        *,
        track_points: torch.Tensor | None,
        track_visibility: torch.Tensor | None,
        track_ids: torch.Tensor | None,
        start_frame: int,
    ) -> torch.Tensor:
        expected_channels = self.in_channels - self.track_channels
        if hidden_states.ndim != 5:
            raise ValueError(
                "TrackWan hidden_states must be [B, C, T, H, W], got "
                f"{tuple(hidden_states.shape)}")
        if hidden_states.shape[1] != expected_channels:
            raise ValueError(
                "TrackWan received an unexpected pre-track channel count: "
                f"{hidden_states.shape[1]} (expected {expected_channels})")
        if (track_points is None) != (track_visibility is None):
            raise ValueError(
                "track_points and track_visibility must be provided together")

        batch_size, _, latent_t, latent_h, latent_w = hidden_states.shape
        if track_points is None:
            track_map = hidden_states.new_zeros(
                batch_size,
                self.track_channels,
                latent_t,
                latent_h,
                latent_w,
            )
        else:
            if track_points.shape[0] != batch_size:
                raise ValueError(
                    "track batch size must match hidden_states batch size")
            temporal_ratio = self.track_encoder.vae_temporal_compression
            full_latent_t = ((track_points.shape[1] - 1) //
                             temporal_ratio + 1)
            end_frame = int(start_frame) + latent_t
            if start_frame < 0 or end_frame > full_latent_t:
                raise ValueError(
                    "track sequence does not cover the requested latent "
                    f"window [{start_frame}, {end_frame}); available "
                    f"latent frames: {full_latent_t}")

            # Encode from the global beginning so the causal temporal conv is
            # left-padded exactly once, then take the requested latent chunk.
            full_track_map = self.track_encoder(
                track_points,
                track_visibility,
                full_latent_t,
                latent_h,
                latent_w,
                track_ids=track_ids,
            )
            track_map = full_track_map[:, :, start_frame:end_frame]
            track_map = track_map.to(dtype=hidden_states.dtype)

        return torch.cat([hidden_states, track_map], dim=1)


class TrackWanTransformer3DModel(
        _TrackConditioningMixin, WanTransformer3DModel):
    """Bidirectional Wan I2V transformer conditioned on sparse point tracks."""

    def __init__(
        self,
        config: TrackWanVideoConfig,
        hf_config: dict[str, Any],
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self._init_track_conditioning(config, hf_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        guidance: torch.Tensor | None = None,
        r_timestep: torch.Tensor | None = None,
        track_points: torch.Tensor | None = None,
        track_visibility: torch.Tensor | None = None,
        track_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self._append_track_conditioning(
            hidden_states,
            track_points=track_points,
            track_visibility=track_visibility,
            track_ids=track_ids,
            start_frame=0,
        )
        return super().forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            guidance=guidance,
            r_timestep=r_timestep,
            **kwargs,
        )


class CausalTrackWanTransformer3DModel(
        _TrackConditioningMixin, CausalWanTransformer3DModel):
    """Causal Wan I2V transformer with globally aligned track chunks."""

    def __init__(
        self,
        config: TrackWanVideoConfig,
        hf_config: dict[str, Any],
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        self._init_track_conditioning(config, hf_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        start_frame: int = 0,
        clean_x: torch.Tensor | None = None,
        aug_t: torch.Tensor | None = None,
        track_points: torch.Tensor | None = None,
        track_visibility: torch.Tensor | None = None,
        track_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = self._append_track_conditioning(
            hidden_states,
            track_points=track_points,
            track_visibility=track_visibility,
            track_ids=track_ids,
            start_frame=start_frame,
        )
        if clean_x is not None:
            clean_x = self._append_track_conditioning(
                clean_x,
                track_points=track_points,
                track_visibility=track_visibility,
                track_ids=track_ids,
                start_frame=start_frame,
            )
        return super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            start_frame=start_frame,
            clean_x=clean_x,
            aug_t=aug_t,
            **kwargs,
        )
