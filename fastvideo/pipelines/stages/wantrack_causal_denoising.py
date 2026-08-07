# SPDX-License-Identifier: Apache-2.0
"""Causal Self-Forcing denoising for TrackWan I2V."""

from __future__ import annotations

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.utils import pred_noise_to_pred_video
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.causal_denoising import CausalDMDDenosingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class WanTrackCausalDenoisingStage(CausalDMDDenosingStage):
    """Causal DMD loop with Wan I2V 20ch concat + sparse track conditioning.

    Differs from :class:`CausalDMDDenosingStage` in three ways:
    1. Latent clip geometry is ``1 + N * num_frames_per_block`` (e.g. 31 = 1+10*3).
    2. Each block cats ``[noise_16, image_latent_20]`` before the DiT forward.
    3. Passes ``track_points`` / ``track_visibility`` / ``track_ids`` and CLIP image embeds.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast

        assert batch.latents is not None, "latents must be provided"
        assert batch.image_latent is not None, "WanTrack requires image_latent (20ch I2V)"
        assert batch.track_points is not None and batch.track_visibility is not None, (
            "WanTrack requires track_points and track_visibility")

        latents = batch.latents  # [B, C, T, H, W]
        b, _c, t, h, w = latents.shape
        image_latent = batch.image_latent.to(device=latents.device, dtype=target_dtype)
        if image_latent.shape[1] != 20:
            raise ValueError(f"WanTrack image_latent must have 20 channels, got {image_latent.shape[1]}")
        if image_latent.shape[2] < t:
            raise ValueError(f"image_latent temporal length {image_latent.shape[2]} < latents {t}")

        latent_seq_length = h * w
        patch_ratio = (self.transformer.config.arch_config.patch_size[-1] *
                       self.transformer.config.arch_config.patch_size[-2])
        self.frame_seq_length = latent_seq_length // patch_ratio

        timesteps = torch.tensor(fastvideo_args.pipeline_config.dmd_denoising_steps, dtype=torch.long).cpu()
        if fastvideo_args.pipeline_config.warp_denoising_step:
            # Ensure the scheduler has a 1000-step grid for warping.
            self.scheduler.set_timesteps(1000, device="cpu")
            scheduler_timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(get_local_torch_device())

        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0

        image_embeds = batch.image_embeds
        if not image_embeds:
            raise ValueError("WanTrack requires CLIP image embeds")
        image_kwargs = {
            "encoder_hidden_states_image": [embed.to(target_dtype) for embed in image_embeds],
        }

        track_points = batch.track_points.to(device=latents.device, dtype=torch.float32)
        track_visibility = batch.track_visibility.to(device=latents.device, dtype=torch.float32)
        track_ids = batch.track_ids
        if track_ids is None:
            track_ids = self._sample_track_ids(track_points, batch)
        else:
            track_ids = track_ids.to(device=latents.device, dtype=torch.long)
        track_kwargs = {
            "track_points": track_points,
            "track_visibility": track_visibility,
            "track_ids": track_ids,
        }

        pos_cond_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {"encoder_attention_mask": batch.prompt_attention_mask},
        )

        kv_cache1 = self._initialize_kv_cache(batch_size=b, dtype=target_dtype, device=latents.device)
        crossattn_cache = self._initialize_crossattn_cache(
            batch_size=b,
            max_text_len=fastvideo_args.pipeline_config.text_encoder_configs[0].arch_config.text_len,
            dtype=target_dtype,
            device=latents.device,
        )

        block_sizes = self._block_sizes(t)
        start_index = 0
        pos_start_base = 0
        context_noise = int(getattr(fastvideo_args.pipeline_config, "context_noise", 0))

        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[:, :, start_index:start_index + current_num_frames, :, :]
                noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
                video_raw_latent_shape = noise_latents_btchw.shape
                image_block = image_latent[:, :, start_index:start_index + current_num_frames, :, :]

                for i, t_cur in enumerate(timesteps):
                    noise_latents = noise_latents_btchw.clone()
                    latent_model_input = torch.cat(
                        [current_latents.to(target_dtype), image_block],
                        dim=1,
                    )
                    t_expand = t_cur.repeat(b)

                    with torch.autocast(device_type="cuda", dtype=target_dtype,
                                        enabled=autocast_enabled), set_forward_context(
                                            current_timestep=i,
                                            attn_metadata=None,
                                            forward_batch=batch,
                                        ):
                        t_expanded_noise = t_cur * torch.ones(
                            (b, 1), device=latent_model_input.device, dtype=torch.long)
                        pred_noise_btchw = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expanded_noise,
                            kv_cache=kv_cache1,
                            crossattn_cache=crossattn_cache,
                            current_start=(pos_start_base + start_index) * self.frame_seq_length,
                            start_frame=start_index,
                            **image_kwargs,
                            **track_kwargs,
                            **pos_cond_kwargs,
                        ).permute(0, 2, 1, 3, 4)

                    pred_video_btchw = pred_noise_to_pred_video(
                        pred_noise=pred_noise_btchw.flatten(0, 1),
                        noise_input_latent=noise_latents.flatten(0, 1),
                        timestep=t_expand,
                        scheduler=self.scheduler,
                    ).unflatten(0, pred_noise_btchw.shape[:2])

                    if i < len(timesteps) - 1:
                        next_timestep = timesteps[i + 1] * torch.ones(
                            [1], dtype=torch.long, device=pred_video_btchw.device)
                        noise = torch.randn(
                            video_raw_latent_shape,
                            dtype=pred_video_btchw.dtype,
                            generator=(batch.generator[0]
                                       if isinstance(batch.generator, list) else batch.generator),
                        ).to(self.device)
                        noise_latents_btchw = self.scheduler.add_noise(
                            pred_video_btchw.flatten(0, 1),
                            noise.flatten(0, 1),
                            next_timestep,
                        ).unflatten(0, pred_video_btchw.shape[:2])
                        current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
                    else:
                        current_latents = pred_video_btchw.permute(0, 2, 1, 3, 4)

                    if progress_bar is not None:
                        progress_bar.update()

                latents[:, :, start_index:start_index + current_num_frames, :, :] = current_latents

                # Commit clean context into the KV cache for later blocks.
                t_context = torch.ones([b], device=latents.device, dtype=torch.long) * context_noise
                context_input = torch.cat(
                    [current_latents.to(target_dtype), image_block],
                    dim=1,
                )
                with torch.autocast(device_type="cuda", dtype=target_dtype,
                                    enabled=autocast_enabled), set_forward_context(
                                        current_timestep=0,
                                        attn_metadata=None,
                                        forward_batch=batch,
                                    ):
                    self.transformer(
                        context_input,
                        prompt_embeds,
                        t_context.unsqueeze(1),
                        kv_cache=kv_cache1,
                        crossattn_cache=crossattn_cache,
                        current_start=(pos_start_base + start_index) * self.frame_seq_length,
                        start_frame=start_index,
                        **image_kwargs,
                        **track_kwargs,
                        **pos_cond_kwargs,
                    )

                start_index += current_num_frames

        batch.latents = latents
        return batch

    def _block_sizes(self, latent_t: int) -> list[int]:
        chunk = int(self.num_frames_per_block)
        if chunk <= 0:
            raise ValueError("num_frames_per_block must be positive")
        if latent_t % chunk == 0:
            return [chunk] * (latent_t // chunk)
        if (latent_t - 1) % chunk == 0:
            return [1] + [chunk] * ((latent_t - 1) // chunk)
        raise ValueError("Causal WanTrack requires latent frames that form complete "
                         f"blocks (optionally after one leading I2V frame); got "
                         f"latent_t={latent_t}, num_frames_per_block={chunk}")

    def _sample_track_ids(
        self,
        track_points: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        max_track_id = int(getattr(getattr(self.transformer, "track_encoder", None), "max_track_id", 100_000))
        batch_size, _t, num_tracks = track_points.shape[:3]
        if num_tracks > max_track_id:
            raise ValueError(f"num_tracks ({num_tracks}) exceeds max_track_id ({max_track_id})")
        generator = batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        # randperm on CUDA does not accept a CPU generator; sample on CPU then move.
        ids = []
        for _ in range(batch_size):
            perm = torch.randperm(max_track_id, generator=generator)[:num_tracks]
            ids.append(perm)
        return torch.stack(ids, dim=0).to(device=track_points.device, dtype=torch.long)

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = super().verify_input(batch, fastvideo_args)
        result.add_check("track_points", batch.track_points, [V.is_tensor, V.with_dims(4)])
        result.add_check("track_visibility", batch.track_visibility, [V.is_tensor, V.with_dims(3)])
        result.add_check("image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)])
        return result
