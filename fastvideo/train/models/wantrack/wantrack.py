# SPDX-License-Identifier: Apache-2.0
"""WanTrack training model: Wan2.2 + MotionStream point-track I2V finetuning.

Mirrors the Matrix-Game 2.0 bidirectional I2V training path, with two differences:
  - conditioning is point tracks (MotionStream) instead of mouse/keyboard, and
  - the original text prompt is KEPT (Wan's T5 cross-attention stays active).

DiT input channel layout (concatenated at the patch embed):
    16 (noisy latent) + 20 (I2V: 4 mask + 16 first-frame latent) + 16 (track) = 52

The I2V cond concat is built here (wrapper); the track map is built *inside* the
transformer (``TrackWanTransformer3DModel``) from the raw tracks so train and
inference share one code path. Loss is plain flow-matching MSE (FineTuneMethod).
"""
from __future__ import annotations

import copy
import os
from typing import Any, Literal

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_i2v_track
from fastvideo.distributed import get_sp_group, get_world_group
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import normalize_dit_input

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader
from fastvideo.train.utils.moduleloader import load_module_from_path


class WanTrackModel(WanModel):
    """Wan2.2 + point-track I2V model for finetuning in the new trainer."""

    _transformer_cls_name: str = "TrackWanTransformer3DModel"

    def init_preprocessors(self, training_config: Any) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()
        text_len = training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_len),
            parquet_schema=pyarrow_schema_i2v_track,
        )
        self.start_step = 0
        # Training uses precomputed conditional text embeds; CFG (incl. the
        # motion/text negative branches) is an inference-time concern.
        self._requires_negative_conditioning = False
        self._init_track_aug()

    def on_train_start(self) -> None:
        return

    # ------------------------------------------------------------------
    # MotionStream train-time augmentations (in-wrapper; no trainer changes).
    # Configured via env vars so the core training framework is untouched:
    #   WANTRACK_AUG=0          disable all (clean overfit)
    #   WANTRACK_MIN_POINTS / WANTRACK_MAX_POINTS   per-step point subsample range
    #                           (0 max => keep all points)
    #   WANTRACK_PMASK          per-frame temporal-mask probability (MotionStream 0.2)
    #   WANTRACK_MOTION_DROP    prob of dropping all motion (CFG ∅_motion)
    #   WANTRACK_TEXT_DROP      prob of zeroing the text embedding (CFG ∅_text)
    # ------------------------------------------------------------------
    def _init_track_aug(self) -> None:
        self._aug_enabled = os.getenv("WANTRACK_AUG", "1") not in ("0", "false", "False")
        self._aug_min_points = int(os.getenv("WANTRACK_MIN_POINTS", "1"))
        self._aug_max_points = int(os.getenv("WANTRACK_MAX_POINTS", "200"))
        self._aug_pmask = float(os.getenv("WANTRACK_PMASK", "0.2"))
        self._aug_motion_drop = float(os.getenv("WANTRACK_MOTION_DROP", "0.1"))
        self._aug_text_drop = float(os.getenv("WANTRACK_TEXT_DROP", "0.0"))

    def _augment_tracks(
        self,
        track_points: torch.Tensor | None,
        track_visibility: torch.Tensor | None,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if track_points is None or track_visibility is None or not self._aug_enabled:
            return track_points, track_visibility
        B, T, N, _ = track_points.shape
        device = track_visibility.device
        gdev = generator.device if generator is not None else device
        vis = track_visibility.clone()

        # 1) Point subsampling: keep K in [min, max] points per sample (others -> vis 0).
        if 0 < self._aug_max_points < N:
            for b in range(B):
                hi = min(self._aug_max_points, N)
                lo = min(self._aug_min_points, hi)
                k = int(torch.randint(lo, hi + 1, (1,), generator=generator, device=gdev).item())
                keep_idx = torch.randperm(N, generator=generator, device=gdev)[:k]
                keep = torch.zeros(N, device=device, dtype=vis.dtype)
                keep[keep_idx.to(device)] = 1.0
                vis[b] = vis[b] * keep.unsqueeze(0)

        # 2) Stochastic temporal masking: zero whole frames with prob pmask.
        if self._aug_pmask > 0.0:
            frame_keep = (torch.rand(B, T, generator=generator, device=gdev) >= self._aug_pmask)
            vis = vis * frame_keep.to(device=device, dtype=vis.dtype).unsqueeze(-1)

        # 3) Motion CFG dropout: drop ALL motion for this step (-> cm = zeros).
        if self._aug_motion_drop > 0.0:
            if float(torch.rand(1, generator=generator, device=gdev).item()) < self._aug_motion_drop:
                return None, None

        return track_points, vis

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()
        encoder_hidden_states = raw_batch["text_embedding"]
        encoder_attention_mask = raw_batch["text_attention_mask"]
        training_batch.infos = raw_batch.get("info_list")

        if latents_source == "data":
            latents = raw_batch["vae_latent"][:, :, :tc.data.num_latent_t].to(device, dtype=dtype)
        else:
            raise ValueError(f"WanTrack only supports latents_source='data', got {latents_source!r}")

        first_frame_latent = raw_batch["first_frame_latent"][:, :, :tc.data.num_latent_t].to(device, dtype=dtype)

        expected_frames = (tc.data.num_latent_t - 1) * self._temporal_compression_ratio() + 1
        track_points = raw_batch.get("track_points")
        track_visibility = raw_batch.get("track_visibility")
        if track_points is not None:
            track_points = track_points[:, :expected_frames].to(device, dtype=dtype)
        if track_visibility is not None:
            track_visibility = track_visibility[:, :expected_frames].to(device, dtype=dtype)

        # MotionStream train-time augments (subsample points / temporal mask /
        # motion-CFG drop). Env-gated; a no-op when WANTRACK_AUG=0.
        track_points, track_visibility = self._augment_tracks(track_points, track_visibility, generator)

        encoder_hidden_states = encoder_hidden_states.to(device, dtype=dtype)
        # Text CFG dropout: zero the text embedding for this step with prob.
        if self._aug_enabled and self._aug_text_drop > 0.0:
            gdev = generator.device if generator is not None else device
            if float(torch.rand(1, generator=generator, device=gdev).item()) < self._aug_text_drop:
                encoder_hidden_states = torch.zeros_like(encoder_hidden_states)

        training_batch.latents = latents
        training_batch.encoder_hidden_states = encoder_hidden_states
        training_batch.encoder_attention_mask = encoder_attention_mask.to(device, dtype=dtype)
        training_batch.image_latents = first_frame_latent
        training_batch.track_points = track_points          # stashed (extra attr)
        training_batch.track_visibility = track_visibility

        training_batch.latents = normalize_dit_input("wan", training_batch.latents, self.vae)
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]
        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        # Base builds noise/timesteps/noisy_model_input (16ch) + text conditional_dict.
        training_batch = super()._prepare_dit_inputs(training_batch, generator)

        image_latents = training_batch.image_latents
        if image_latents is None:
            raise RuntimeError("WanTrack requires first_frame_latent (image_latents)")
        cond_latents = self._build_i2v_cond_concat(image_latents)  # [B, 20, T, H, W]
        training_batch.image_latents = cond_latents
        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, cond_latents], dim=1)  # [B, 36, T, H, W]

        assert training_batch.conditional_dict is not None
        training_batch.conditional_dict["image_latents"] = cond_latents
        training_batch.conditional_dict["track_points"] = getattr(training_batch, "track_points", None)
        training_batch.conditional_dict["track_visibility"] = getattr(training_batch, "track_visibility", None)
        # No CFG dropout during finetuning: uncond mirrors cond (keeps text + tracks).
        training_batch.unconditional_dict = dict(training_batch.conditional_dict)
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for WanTrack")
        hidden_states = noise_input.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        if hidden_states.shape[1] == 16:  # fallback: concat I2V cond if caller passed raw 16ch
            cond_latents = text_dict.get("image_latents")
            if cond_latents is None:
                raise RuntimeError("WanTrack needs image_latents in conditional_dict for a 16ch input")
            hidden_states = torch.cat([hidden_states, cond_latents[:, :, :hidden_states.shape[2]]], dim=1)
        return {
            "hidden_states": hidden_states,                                   # [B, 36, T, H, W]
            "encoder_hidden_states": text_dict["encoder_hidden_states"],      # T5 text (kept)
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep.to(device=self.device, dtype=torch.bfloat16),
            "track_points": text_dict.get("track_points"),
            "track_visibility": text_dict.get("track_visibility"),
            "return_dict": False,
        }

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del cfg_uncond
        return batch.unconditional_dict or batch.conditional_dict

    # ------------------------------------------------------------------
    # Helpers (I2V mask+first-frame concat; mirrors MatrixGame2)
    # ------------------------------------------------------------------

    def _build_i2v_cond_concat(self, image_latents: torch.Tensor) -> torch.Tensor:
        """[B,16,T,H,W] first-frame latent -> [B,20,T,H,W] (4 mask + 16 latent)."""
        if image_latents.ndim != 5:
            raise ValueError(f"first_frame_latent must be [B,C,T,H,W], got {tuple(image_latents.shape)}")
        if image_latents.shape[1] == 20:
            return image_latents
        if image_latents.shape[1] != 16:
            raise ValueError(f"WanTrack expects first_frame_latent with 16 or 20 channels, got {image_latents.shape[1]}")

        ratio = self._temporal_compression_ratio()
        b, _, num_latent_t, h, w = image_latents.shape
        num_frames = (num_latent_t - 1) * ratio + 1

        mask = torch.ones(b, 1, num_frames, h, w, device=image_latents.device, dtype=image_latents.dtype)
        mask[:, :, 1:] = 0
        first = torch.repeat_interleave(mask[:, :, :1], dim=2, repeats=ratio)
        mask = torch.cat([first, mask[:, :, 1:]], dim=2)
        mask = mask.view(b, -1, ratio, h, w).transpose(1, 2)  # [B, ratio, num_latent_t, H, W]
        return torch.cat([mask, image_latents], dim=1)

    def _temporal_compression_ratio(self) -> int:
        assert self.training_config is not None
        return int(self.training_config.pipeline_config.vae_config.arch_config.temporal_compression_ratio)
