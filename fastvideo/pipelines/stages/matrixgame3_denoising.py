# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import torch  # type: ignore

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.lingbotworld.cam_utils import compute_relative_poses
from fastvideo.models.dits.matrixgame3.utils import (
    build_extrinsics_from_actions,
    build_matrixgame3_action_preset,
    build_plucker_from_c2ws,
    build_plucker_from_pose,
    interpolate_camera_poses_handedness,
    select_memory_idx_fov,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising import DenoisingStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult

logger = init_logger(__name__)


class MatrixGame3DenoisingStage(DenoisingStage):

    def _infer_num_iterations(self, batch: ForwardBatch) -> int:
        if batch.num_iterations is not None:
            return batch.num_iterations
        if isinstance(batch.num_frames, int) and batch.num_frames > 57:
            return 1 + max(0, (batch.num_frames - 57 + 39) // 40)
        return 1

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check("image_latent", batch.image_latent, V.none_or_tensor_with_dims(5))
        result.add_check("num_inference_steps", batch.num_inference_steps, V.positive_int)
        result.add_check("guidance_scale", batch.guidance_scale, V.positive_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("do_classifier_free_guidance", batch.do_classifier_free_guidance, V.bool_value)
        result.add_check("negative_prompt_embeds", batch.negative_prompt_embeds,
                         lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x))
        return result

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        assert batch.latents is not None, "latents must be prepared before MatrixGame3 denoising"
        assert batch.image_latent is not None, "MatrixGame3 requires first-frame VAE latents"
        assert batch.prompt_embeds, "MatrixGame3 requires text embeddings"

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32) and not fastvideo_args.disable_autocast
        device = batch.latents.device

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta,
            },
        )

        prompt_embeds = batch.prompt_embeds
        negative_prompt_embeds = batch.negative_prompt_embeds or []
        use_base_model = batch.use_base_model

        latents = batch.latents
        img_cond = batch.image_latent.to(device=device, dtype=target_dtype)
        latent_h = latents.shape[-2]
        latent_w = latents.shape[-1]
        patch_h = int(getattr(self.transformer, "patch_size", (1, 2, 2))[1])
        patch_w = int(getattr(self.transformer, "patch_size", (1, 2, 2))[2])
        latent_h_aligned = (latent_h // patch_h) * patch_h
        latent_w_aligned = (latent_w // patch_w) * patch_w
        if latent_h_aligned != latent_h or latent_w_aligned != latent_w:
            logger.warning(
                "Cropping MatrixGame3 latents to patch-aligned size: (%d, %d) -> (%d, %d), patch=(%d,%d)",
                latent_h,
                latent_w,
                latent_h_aligned,
                latent_w_aligned,
                patch_h,
                patch_w,
            )
            latents = latents[:, :, :, :latent_h_aligned, :latent_w_aligned]
            img_cond = img_cond[:, :, :, :latent_h_aligned, :latent_w_aligned]
            latent_h = latent_h_aligned
            latent_w = latent_w_aligned
        spatial_ratio = fastvideo_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        target_h = latent_h * spatial_ratio
        target_w = latent_w * spatial_ratio
        num_iterations = self._infer_num_iterations(batch)
        clip_frame = 56  # hardcode for now
        first_clip_frame = clip_frame + 1
        past_frame = 16

        def align_frame_to_block(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 * 4 + 1 if frame_idx > 0 else 1

        def get_latent_idx(frame_idx: int) -> int:
            return (frame_idx - 1) // 4 + 1

        total_video_frames = first_clip_frame + max(0, num_iterations - 1) * (clip_frame - past_frame)
        if batch.keyboard_cond is None or batch.mouse_cond is None:
            keyboard_cond, mouse_cond = build_matrixgame3_action_preset(total_video_frames, seed=batch.seed)
            batch.keyboard_cond = keyboard_cond.unsqueeze(0).to(device=device, dtype=target_dtype)
            batch.mouse_cond = mouse_cond.unsqueeze(0).to(device=device, dtype=target_dtype)
        else:
            batch.keyboard_cond = batch.keyboard_cond.to(device=device, dtype=target_dtype)
            batch.mouse_cond = batch.mouse_cond.to(device=device, dtype=target_dtype)

        extrinsics_all = build_extrinsics_from_actions(batch.keyboard_cond[0], batch.mouse_cond[0]).to(device)
        all_latents: list[torch.Tensor] = []

        for clip_idx in range(num_iterations):
            try:
                self.scheduler.set_timesteps(
                    batch.num_inference_steps,
                    device=device,
                    shift=fastvideo_args.pipeline_config.flow_shift,
                )
            except TypeError:
                self.scheduler.set_timesteps(batch.num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            first_clip = clip_idx == 0
            current_end_frame_idx = first_clip_frame if first_clip else first_clip_frame + clip_idx * (clip_frame -
                                                                                                       past_frame)
            current_start_frame_idx = 0 if first_clip else current_end_frame_idx - clip_frame
            # 15 for first clip, 14 for later clips
            current_latent_frames = (first_clip_frame - 1) // 4 + 1 if first_clip else clip_frame // 4
            cond_frames = 1 if first_clip else 4
            latent_start_idx = get_latent_idx(current_start_frame_idx)
            latent_end_idx = get_latent_idx(current_end_frame_idx)

            clip_keyboard = batch.keyboard_cond[:, current_start_frame_idx:current_end_frame_idx]
            clip_mouse = batch.mouse_cond[:, current_start_frame_idx:current_end_frame_idx]

            cond_frames = min(cond_frames, img_cond.shape[2])
            current_latents = torch.randn(
                (latents.shape[0], latents.shape[1], latent_end_idx - latent_start_idx, latent_h, latent_w),
                generator=batch.generator if isinstance(batch.generator, torch.Generator) else None,
                device=device,
                dtype=target_dtype)
            current_latents[:, :, :cond_frames] = img_cond[:, :, :cond_frames]

            c2ws_chunk = extrinsics_all[current_start_frame_idx:current_end_frame_idx]
            src_indices = np.linspace(current_start_frame_idx, current_end_frame_idx - 1,
                                      current_end_frame_idx - current_start_frame_idx)
            tgt_indices = np.linspace(0 if first_clip else current_start_frame_idx + 3, current_end_frame_idx - 1,
                                      current_latent_frames)

            if batch.c2ws_plucker_emb is not None:
                plucker_no_mem = batch.c2ws_plucker_emb[:, :, current_start_frame_idx:current_end_frame_idx]
                plucker_no_mem = plucker_no_mem.to(device=device, dtype=target_dtype)
            else:
                plucker_no_mem = build_plucker_from_c2ws(
                    c2ws_chunk,
                    src_indices,
                    tgt_indices,
                    target_h=target_h,
                    target_w=target_w,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    framewise=True,
                ).to(device=device, dtype=target_dtype)

            x_memory = None
            timestep_memory = None
            mouse_cond_memory = None
            keyboard_cond_memory = None
            memory_latent_idx = None
            c2ws_plucker_emb = plucker_no_mem
            if all_latents:
                # t-1, t-9, t-17, t-25, t-33
                selected_index_base = [current_end_frame_idx - offset for offset in range(1, 34, 8)]
                selected_index = select_memory_idx_fov(
                    extrinsics_all,
                    current_start_frame_idx,
                    selected_index_base,
                    height=target_h,
                    width=target_w,
                )
                if len(selected_index) > 0:
                    selected_index[-1] = 4

                memory_pluckers: list[torch.Tensor] = []
                memory_latent_idx = []
                for mem_idx, reference_idx in zip(selected_index, selected_index_base, strict=False):
                    memory_latent_idx.append(get_latent_idx(mem_idx))

                    mem_idx_aligned = align_frame_to_block(mem_idx)
                    mem_block = extrinsics_all[mem_idx_aligned:mem_idx_aligned + 4]
                    mem_src = np.linspace(mem_idx_aligned, mem_idx_aligned + mem_block.shape[0] - 1, mem_block.shape[0])
                    mem_tgt = np.array([mem_idx_aligned + 3], dtype=np.float32)
                    mem_pose = interpolate_camera_poses_handedness(
                        src_indices=mem_src,
                        src_rot_mat=mem_block[:, :3, :3].detach().cpu().numpy(),
                        src_trans_vec=mem_block[:, :3, 3].detach().cpu().numpy(),
                        tgt_indices=mem_tgt,
                    ).to(device=device, dtype=target_dtype)
                    reference_pose = extrinsics_all[reference_idx:reference_idx + 1].to(device=device,
                                                                                        dtype=target_dtype)
                    rel_pair = torch.cat([reference_pose, mem_pose], dim=0)
                    rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                    memory_pluckers.append(
                        build_plucker_from_pose(
                            rel_pose,
                            target_h=target_h,
                            target_w=target_w,
                            latent_h=latent_h,
                            latent_w=latent_w,
                        ).to(device=device, dtype=target_dtype))

                if memory_pluckers:
                    c2ws_plucker_emb = torch.cat(memory_pluckers + [plucker_no_mem], dim=2)
                    history = torch.cat(all_latents, dim=2)
                    x_memory = history[:, :, memory_latent_idx]
                    mouse_cond_memory = torch.ones(
                        (clip_keyboard.shape[0], len(memory_latent_idx), clip_mouse.shape[-1]),
                        device=device,
                        dtype=target_dtype)
                    keyboard_cond_memory = -torch.ones(
                        (clip_keyboard.shape[0], len(memory_latent_idx), clip_keyboard.shape[-1]),
                        device=device,
                        dtype=target_dtype)
                    timestep_memory = x_memory.new_zeros(
                        (x_memory.shape[0], x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))
            with self.progress_bar(total=len(timesteps)) as progress_bar:
                for timestep in timesteps:
                    latent_model_input = current_latents
                    if hasattr(self.scheduler, "scale_model_input"):
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

                    timestep_tokens = latent_model_input.new_full(
                        (
                            latent_model_input.shape[2],
                            latent_model_input.shape[3] * latent_model_input.shape[4] // 4,
                        ),
                        timestep,
                    )
                    timestep_tokens[:cond_frames].zero_()
                    timestep_tokens = timestep_tokens.flatten().unsqueeze(0)
                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled), \
                        set_forward_context(current_timestep=timestep,
                                            attn_metadata=None,
                                            forward_batch=batch):
                        noise_pred = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            timestep_tokens,
                            mouse_cond=clip_mouse,
                            keyboard_cond=clip_keyboard,
                            x_memory=x_memory,
                            timestep_memory=timestep_memory,
                            mouse_cond_memory=mouse_cond_memory,
                            keyboard_cond_memory=keyboard_cond_memory,
                            c2ws_plucker_emb=c2ws_plucker_emb,
                            memory_latent_idx=memory_latent_idx,
                            predict_latent_idx=(latent_start_idx, latent_end_idx),
                        )

                        if use_base_model and batch.do_classifier_free_guidance and negative_prompt_embeds:
                            null_mouse_cond = torch.ones_like(clip_mouse)
                            null_keyboard_cond = -torch.ones_like(clip_keyboard)
                            noise_pred_uncond = self.transformer(
                                latent_model_input,
                                negative_prompt_embeds,
                                timestep_tokens,
                                mouse_cond=null_mouse_cond,
                                keyboard_cond=null_keyboard_cond,
                                x_memory=None,
                                timestep_memory=None,
                                mouse_cond_memory=None,
                                keyboard_cond_memory=None,
                                c2ws_plucker_emb=plucker_no_mem,
                                memory_latent_idx=None,
                                predict_latent_idx=(latent_start_idx, latent_end_idx),
                            )
                            noise_pred = noise_pred_uncond + batch.guidance_scale * (noise_pred - noise_pred_uncond)

                    if noise_pred.shape != current_latents.shape:
                        aligned_t = min(noise_pred.shape[2], current_latents.shape[2])
                        aligned_h = min(noise_pred.shape[3], current_latents.shape[3])
                        aligned_w = min(noise_pred.shape[4], current_latents.shape[4])
                        logger.warning(
                            "Aligning noise/sample shapes before scheduler.step: noise=%s sample=%s -> (*,*,%d,%d,%d)",
                            tuple(noise_pred.shape),
                            tuple(current_latents.shape),
                            aligned_t,
                            aligned_h,
                            aligned_w,
                        )
                        noise_pred = noise_pred[:, :, :aligned_t, :aligned_h, :aligned_w]
                        current_latents = current_latents[:, :, :aligned_t, :aligned_h, :aligned_w]

                    current_latents = self.scheduler.step(noise_pred,
                                                          timestep,
                                                          current_latents,
                                                          **extra_step_kwargs,
                                                          return_dict=False)[0]
                    current_latents[:, :, :cond_frames] = img_cond[:, :, :cond_frames]
                    progress_bar.update()

            img_cond = current_latents[:, :, -4:].detach()
            denoised_pred = current_latents if first_clip else current_latents[:, :, -10:]
            all_latents.append(denoised_pred.detach())

        batch.latents = torch.cat(all_latents, dim=2)
        return batch
