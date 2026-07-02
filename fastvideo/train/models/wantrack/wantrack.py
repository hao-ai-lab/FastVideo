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
        # Track sampling is ON by default, MotionStream-style: a random 1000-2500 subset
        # of the 2500-grid tracks each step (their range; also matches inference where you
        # never have a perfect full grid).
        self._aug_enabled = os.getenv("WANTRACK_AUG", "1") not in ("0", "false", "False")
        self._aug_min_points = int(os.getenv("WANTRACK_MIN_POINTS", "1000"))
        self._aug_max_points = int(os.getenv("WANTRACK_MAX_POINTS", "2500"))
        # MotionStream stochastic mid-frame masking: zero CONTIGUOUS frame chunks (not
        # independent frames) with prob pmask; WANTRACK_MASK_CHUNK = chunk length (frames).
        self._aug_pmask = float(os.getenv("WANTRACK_PMASK", "0.2"))
        self._aug_mask_chunk = int(os.getenv("WANTRACK_MASK_CHUNK", "8"))
        # Diversity: 0 = uniform (MotionStream). >0 up-weights moving/diverse tracks over
        # static/redundant ones (base weight 1 keeps SOME static tracks -- don't overdo it).
        self._aug_diversity = float(os.getenv("WANTRACK_DIVERSITY", "1.0"))
        # CFG dropout (ours, not MotionStream) -> default OFF; enables motion/text CFG when >0.
        self._aug_motion_drop = float(os.getenv("WANTRACK_MOTION_DROP", "0.0"))
        self._aug_text_drop = float(os.getenv("WANTRACK_TEXT_DROP", "0.0"))

    def _augment_tracks(
        self,
        track_points: torch.Tensor | None,
        track_visibility: torch.Tensor | None,
        generator: torch.Generator,
        object_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """MotionStream-style track augmentation.

        Sampling keeps K~U[min,max] tracks per step by zeroing the visibility of the rest,
        with two twists over uniform random: (a) OBJECT COVERAGE -- if ``object_ids``
        ([B,N] or [N] SAM segment labels, -1 = none) is given, guarantee >=1 track per
        segment; (b) DIVERSITY -- weight sampling by per-track motion so moving/unique
        tracks are favored while some static ones survive (base weight 1).
        """
        if track_points is None or track_visibility is None or not self._aug_enabled:
            return track_points, track_visibility
        B, T, N, _ = track_points.shape
        device = track_visibility.device
        gdev = generator.device if generator is not None else device
        vis = track_visibility.clone()

        # per-track motion magnitude (max displacement from frame 0, visible frames) -> diversity weight
        tp = track_points.float()
        disp = ((tp - tp[:, :1])**2).sum(-1).sqrt()  # [B,T,N]
        motion = (disp * (track_visibility > 0.5).float()).amax(dim=1)  # [B,N]

        if 0 < self._aug_max_points < N:
            oids = object_ids
            if oids is not None and oids.dim() == 1:
                oids = oids.unsqueeze(0).expand(B, -1)
            for b in range(B):
                hi = min(self._aug_max_points, N)
                lo = min(self._aug_min_points, hi)
                k = int(torch.randint(lo, hi + 1, (1, ), generator=generator, device=gdev).item())
                m = motion[b].to(gdev)
                w = 1.0 + self._aug_diversity * (m / (m.mean() + 1e-6))
                w = w * (track_visibility[b, 0].to(gdev) > 0.5).float()  # only frame-0-visible are valid queries
                if float(w.sum()) <= 0.0:
                    w = torch.ones(N, device=gdev)
                keep = torch.zeros(N, device=gdev)
                # (a) object coverage: one weighted pick per present segment
                if oids is not None:
                    ob = oids[b].to(gdev)
                    for oid in torch.unique(ob):
                        if int(oid) < 0:
                            continue
                        idxs = (ob == oid).nonzero(as_tuple=True)[0]
                        ww = w[idxs]
                        if float(ww.sum()) <= 0.0:
                            ww = torch.ones_like(ww)
                        keep[idxs[torch.multinomial(ww, 1, generator=generator)]] = 1.0
                # (b) fill remaining budget with weighted sampling over not-yet-kept points
                n_cov = int(keep.sum().item())
                wa = w * (keep < 0.5).float()
                rem = min(max(0, k - n_cov), int((wa > 0).sum().item()))
                if rem > 0:
                    keep[torch.multinomial(wa, rem, replacement=False, generator=generator)] = 1.0
                vis[b] = vis[b] * keep.to(device=device, dtype=vis.dtype).unsqueeze(0)

        # Stochastic CONTIGUOUS-chunk masking (MotionStream mid-frame chunks): zero L-frame
        # blocks with prob pmask -- simulates the user releasing control for a stretch.
        if self._aug_pmask > 0.0 and self._aug_mask_chunk > 0:
            L = max(1, self._aug_mask_chunk)
            n_chunks = (T + L - 1) // L
            drop = (torch.rand(B, n_chunks, generator=generator, device=gdev) < self._aug_pmask)
            frame_keep = torch.ones(B, T, device=gdev)
            for b in range(B):
                for c in range(n_chunks):
                    if bool(drop[b, c]):
                        frame_keep[b, c * L:(c + 1) * L] = 0.0
            vis = vis * frame_keep.to(device=device, dtype=vis.dtype).unsqueeze(-1)

        # Motion CFG dropout (ours; default off): drop ALL motion for this step (-> cm = zeros).
        if (self._aug_motion_drop > 0.0
                and float(torch.rand(1, generator=generator, device=gdev).item()) < self._aug_motion_drop):
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

        # CLIP image embedding of frame 0 (Wan2.1 I2V semantic cross-attention pathway).
        image_embeds = raw_batch.get("clip_feature")
        if image_embeds is None or (torch.is_tensor(image_embeds) and image_embeds.numel() == 0):
            raise ValueError("WanTrack (I2V) requires 'clip_feature'; re-run the i2v_track preprocess with an "
                             "image_encoder (e.g. the Wan2.1-Fun-1.3B-InP base).")
        image_embeds = image_embeds.to(device, dtype=dtype)

        expected_frames = (tc.data.num_latent_t - 1) * self._temporal_compression_ratio() + 1
        track_points = raw_batch.get("track_points")
        track_visibility = raw_batch.get("track_visibility")
        if track_points is not None:
            track_points = track_points[:, :expected_frames].to(device, dtype=dtype)
        if track_visibility is not None:
            track_visibility = track_visibility[:, :expected_frames].to(device, dtype=dtype)

        # Optional SAM object labels per track ([B,N]); enables object-coverage sampling
        # in _augment_tracks. Empty/absent on datasets preprocessed before segmentation.
        object_ids = raw_batch.get("object_ids")
        if (object_ids is not None and torch.is_tensor(object_ids) and object_ids.numel() > 0
                and object_ids.shape[-1] == (track_points.shape[2] if track_points is not None else -1)):
            object_ids = object_ids.to(device).long()
        else:
            object_ids = None

        # MotionStream train-time augments (1000-2500 track sampling w/ object coverage +
        # diversity, contiguous-chunk masking). Env-gated; a no-op when WANTRACK_AUG=0.
        track_points, track_visibility = self._augment_tracks(track_points,
                                                              track_visibility,
                                                              generator,
                                                              object_ids=object_ids)

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
        training_batch.image_embeds = image_embeds  # CLIP frame-0 features (cross-attn)
        training_batch.track_points = track_points  # stashed (extra attr)
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
        training_batch.noisy_model_input = torch.cat([training_batch.noisy_model_input, cond_latents],
                                                     dim=1)  # [B, 36, T, H, W]

        assert training_batch.conditional_dict is not None
        training_batch.conditional_dict["image_latents"] = cond_latents
        training_batch.conditional_dict["encoder_hidden_states_image"] = getattr(training_batch, "image_embeds", None)
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
            "hidden_states": hidden_states,  # [B, 36, T, H, W]
            "encoder_hidden_states": text_dict["encoder_hidden_states"],  # T5 text (kept)
            "encoder_attention_mask": text_dict["encoder_attention_mask"],
            "timestep": timestep.to(device=self.device, dtype=torch.bfloat16),
            "encoder_hidden_states_image": text_dict.get("encoder_hidden_states_image"),
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
            raise ValueError(
                f"WanTrack expects first_frame_latent with 16 or 20 channels, got {image_latents.shape[1]}")

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
