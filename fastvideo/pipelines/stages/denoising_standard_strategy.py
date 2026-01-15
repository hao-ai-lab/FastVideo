# SPDX-License-Identifier: Apache-2.0
"""
Standard denoising strategy backed by the legacy DenoisingStage utilities.
"""

from __future__ import annotations

from typing import Any

import torch

from fastvideo.configs.pipelines.base import STA_Mode
from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.denoising_strategies import (
    DenoisingStrategy,
    ModelInputs,
    StrategyState,
)
from fastvideo.utils import dict_to_3d_list, masks_like

try:
    from fastvideo.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)
    st_attn_available = True
except ImportError:
    st_attn_available = False
    SlidingTileAttentionBackend = None  # type: ignore

try:
    from fastvideo.attention.backends.vmoba import VMOBAAttentionBackend
    from fastvideo.utils import is_vmoba_available
    vmoba_attn_available = is_vmoba_available()
except ImportError:
    vmoba_attn_available = False
    VMOBAAttentionBackend = None  # type: ignore

try:
    from fastvideo.attention.backends.video_sparse_attn import (
        VideoSparseAttentionBackend)
    vsa_available = True
except ImportError:
    vsa_available = False
    VideoSparseAttentionBackend = None  # type: ignore

logger = init_logger(__name__)


class StandardStrategy(DenoisingStrategy):

    def __init__(self, stage: Any) -> None:
        self.stage = stage

    def prepare(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> StrategyState:
        pipeline = self.stage.pipeline() if self.stage.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.stage.transformer = loader.load(
                fastvideo_args.model_paths["transformer"], fastvideo_args)
            if pipeline:
                pipeline.add_module("transformer", self.stage.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        extra_step_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta
            },
        )

        target_dtype = torch.bfloat16
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.stage.scheduler.order

        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert not torch.isnan(
                image_embeds[0]).any(), "image_embeds contains nan"
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(
                    None, t_max=50, l_max=60, h_max=24)
            },
        )

        pos_cond_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

        neg_cond_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "encoder_hidden_states_2": batch.clip_embedding_neg,
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )

        action_kwargs = self.stage.prepare_extra_func_kwargs(
            self.stage.transformer.forward,
            {
                "mouse_cond": batch.mouse_cond,
                "keyboard_cond": batch.keyboard_cond,
            },
        )

        if (st_attn_available
                and self.stage.attn_backend == SlidingTileAttentionBackend):
            self.stage.prepare_sta_param(batch, fastvideo_args)

        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        assert not torch.isnan(
            prompt_embeds[0]).any(), "prompt_embeds contains nan"

        neg_prompt_embeds = None
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            assert not torch.isnan(
                neg_prompt_embeds[0]).any(), "neg_prompt_embeds contains nan"

        boundary_ratio = (
            fastvideo_args.pipeline_config.dit_config.boundary_ratio)
        if batch.boundary_ratio is not None:
            logger.info("Overriding boundary ratio from %s to %s",
                        boundary_ratio, batch.boundary_ratio)
            boundary_ratio = batch.boundary_ratio
        if boundary_ratio is not None:
            boundary_timestep = (boundary_ratio *
                                 self.stage.scheduler.num_train_timesteps)
        else:
            boundary_timestep = None

        latent_model_input = latents.to(target_dtype)
        if latent_model_input.ndim == 5:
            assert latent_model_input.shape[0] == 1, (
                "only support batch size 1")

        ti2v_mask = None
        ti2v_z = None
        ti2v_seq_len = None
        if (fastvideo_args.pipeline_config.ti2v_task
                and batch.pil_image is not None):
            assert batch.image_latent is None, (
                "TI2V task should not have image latents")
            assert self.stage.vae is not None, (
                "VAE is not provided for TI2V task")
            z = self.stage.vae.encode(batch.pil_image).mean.float()
            if (hasattr(self.stage.vae, "shift_factor")
                    and self.stage.vae.shift_factor is not None):
                if isinstance(self.stage.vae.shift_factor, torch.Tensor):
                    z -= self.stage.vae.shift_factor.to(z.device, z.dtype)
                else:
                    z -= self.stage.vae.shift_factor

            if isinstance(self.stage.vae.scaling_factor, torch.Tensor):
                z = z * self.stage.vae.scaling_factor.to(z.device, z.dtype)
            else:
                z = z * self.stage.vae.scaling_factor

            latent_model_input = latents.to(target_dtype).squeeze(0)
            _, mask2 = masks_like([latent_model_input], zero=True)
            latent_model_input = ((1. - mask2[0]) * z +
                                  mask2[0] * latent_model_input)
            latent_model_input = latent_model_input.to(get_local_torch_device())
            latents = latent_model_input
            F = batch.num_frames
            temporal_scale = (fastvideo_args.pipeline_config.vae_config.
                              arch_config.scale_factor_temporal)
            spatial_scale = (fastvideo_args.pipeline_config.vae_config.
                             arch_config.scale_factor_spatial)
            patch_size = (fastvideo_args.pipeline_config.dit_config.arch_config.
                          patch_size)
            seq_len = ((F - 1) // temporal_scale +
                       1) * (batch.height // spatial_scale) * (
                           batch.width // spatial_scale) // (patch_size[1] *
                                                             patch_size[2])
            ti2v_mask = mask2[0]
            ti2v_z = z
            ti2v_seq_len = seq_len

        trajectory_timesteps: list[torch.Tensor] | None = None
        trajectory_latents: list[torch.Tensor] | None = None
        if batch.return_trajectory_latents:
            trajectory_timesteps = []
            trajectory_latents = []

        progress_bar = self.stage.progress_bar(total=num_inference_steps)

        extra: dict[str, Any] = {
            "batch": batch,
            "fastvideo_args": fastvideo_args,
            "extra_step_kwargs": extra_step_kwargs,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "num_warmup_steps": num_warmup_steps,
            "image_kwargs": image_kwargs,
            "pos_cond_kwargs": pos_cond_kwargs,
            "neg_cond_kwargs": neg_cond_kwargs,
            "action_kwargs": action_kwargs,
            "boundary_timestep": boundary_timestep,
            "progress_bar": progress_bar,
            "trajectory_timesteps": trajectory_timesteps,
            "trajectory_latents": trajectory_latents,
            "ti2v_mask": ti2v_mask,
            "ti2v_z": ti2v_z,
            "ti2v_seq_len": ti2v_seq_len,
        }

        return StrategyState(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            prompt_attention_mask=batch.prompt_attention_mask,
            negative_attention_mask=batch.negative_attention_mask,
            image_embeds=image_embeds,
            guidance_scale=batch.guidance_scale,
            guidance_scale_2=batch.guidance_scale_2,
            guidance_rescale=batch.guidance_rescale,
            do_cfg=batch.do_classifier_free_guidance,
            extra=extra,
        )

    def make_model_inputs(self, state: StrategyState, t: torch.Tensor,
                          step_idx: int) -> ModelInputs:
        batch = state.extra["batch"]
        fastvideo_args = state.extra["fastvideo_args"]
        target_dtype = state.extra["target_dtype"]
        boundary_timestep = state.extra["boundary_timestep"]

        if getattr(self.stage, "interrupt", False):
            state.extra["skip_step"] = True
        else:
            state.extra["skip_step"] = False

        if boundary_timestep is None or t >= boundary_timestep:
            if (fastvideo_args.dit_cpu_offload
                    and not fastvideo_args.dit_layerwise_offload
                    and self.stage.transformer_2 is not None
                    and next(self.stage.transformer_2.parameters()).device.type
                    == 'cuda'):
                self.stage.transformer_2.to('cpu')
            current_model = self.stage.transformer
            if (fastvideo_args.dit_cpu_offload
                    and not fastvideo_args.dit_layerwise_offload
                    and not fastvideo_args.use_fsdp_inference
                    and current_model is not None):
                transformer_device = next(
                    current_model.parameters()).device.type
                if transformer_device == 'cpu':
                    current_model.to(get_local_torch_device())
            current_guidance_scale = batch.guidance_scale
        else:
            if (fastvideo_args.dit_cpu_offload
                    and not fastvideo_args.dit_layerwise_offload
                    and next(self.stage.transformer.parameters()).device.type
                    == 'cuda'):
                self.stage.transformer.to('cpu')
            current_model = self.stage.transformer_2
            if (fastvideo_args.dit_cpu_offload
                    and not fastvideo_args.dit_layerwise_offload
                    and not fastvideo_args.use_fsdp_inference
                    and current_model is not None):
                transformer_2_device = next(
                    current_model.parameters()).device.type
                if transformer_2_device == 'cpu':
                    current_model.to(get_local_torch_device())
            current_guidance_scale = batch.guidance_scale_2

        assert current_model is not None, "current_model is None"
        state.extra["current_model"] = current_model
        state.extra["current_guidance_scale"] = current_guidance_scale
        state.extra["step_idx"] = step_idx

        latent_model_input = state.latents.to(target_dtype)
        if batch.video_latent is not None:
            latent_model_input = torch.cat([
                latent_model_input, batch.video_latent,
                torch.zeros_like(state.latents)
            ],
                                           dim=1).to(target_dtype)
        elif batch.image_latent is not None:
            assert not fastvideo_args.pipeline_config.ti2v_task, (
                "image latents should not be provided for TI2V task")
            latent_model_input = torch.cat(
                [latent_model_input, batch.image_latent],
                dim=1).to(target_dtype)

        if (fastvideo_args.pipeline_config.ti2v_task
                and batch.pil_image is not None):
            timestep = torch.stack([t]).to(get_local_torch_device())
            mask2 = state.extra["ti2v_mask"]
            seq_len = state.extra["ti2v_seq_len"]
            temp_ts = (mask2[0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat([
                temp_ts,
                temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
            ])
            timestep = temp_ts.unsqueeze(0)
            t_expand = timestep.repeat(latent_model_input.shape[0], 1)
        else:
            t_expand = t.repeat(latent_model_input.shape[0])

        latent_model_input = self.stage.scheduler.scale_model_input(
            latent_model_input, t)

        guidance_expand = None
        if fastvideo_args.pipeline_config.embedded_cfg_scale is not None:
            guidance_expand = (torch.tensor(
                [fastvideo_args.pipeline_config.embedded_cfg_scale] *
                latent_model_input.shape[0],
                dtype=torch.float32,
                device=get_local_torch_device(),
            ).to(target_dtype) * 1000.0)

        state.extra["guidance_expand"] = guidance_expand
        return ModelInputs(
            latent_model_input=latent_model_input,
            timestep=t_expand,
            prompt_embeds=state.prompt_embeds,
            prompt_attention_mask=state.prompt_attention_mask,
        )

    def forward(self, state: StrategyState,
                model_inputs: ModelInputs) -> torch.Tensor:
        if state.extra.get("skip_step", False):
            return state.latents

        batch = state.extra["batch"]
        fastvideo_args = state.extra["fastvideo_args"]
        target_dtype = state.extra["target_dtype"]
        autocast_enabled = state.extra["autocast_enabled"]
        current_model = state.extra["current_model"]
        current_guidance_scale = state.extra["current_guidance_scale"]
        step_idx = state.extra["step_idx"]
        guidance_expand = state.extra["guidance_expand"]

        image_kwargs = state.extra["image_kwargs"]
        pos_cond_kwargs = state.extra["pos_cond_kwargs"]
        neg_cond_kwargs = state.extra["neg_cond_kwargs"]
        action_kwargs = state.extra["action_kwargs"]

        if ((st_attn_available
             and self.stage.attn_backend == SlidingTileAttentionBackend) or
            (vsa_available
             and self.stage.attn_backend == VideoSparseAttentionBackend)):
            self.attn_metadata_builder_cls = (
                self.stage.attn_backend.get_builder_cls())

            if self.attn_metadata_builder_cls is not None:
                self.attn_metadata_builder = self.attn_metadata_builder_cls()
                attn_metadata = self.attn_metadata_builder.build(  # type: ignore
                    current_timestep=step_idx,  # type: ignore
                    raw_latent_shape=batch.
                    raw_latent_shape[2:5],  # type: ignore
                    patch_size=fastvideo_args.pipeline_config.dit_config.
                    patch_size,  # type: ignore
                    STA_param=batch.STA_param,  # type: ignore
                    VSA_sparsity=fastvideo_args.VSA_sparsity,  # type: ignore
                    device=get_local_torch_device(),
                )
                assert attn_metadata is not None, (
                    "attn_metadata cannot be None")
            else:
                attn_metadata = None
        elif (vmoba_attn_available
              and self.stage.attn_backend == VMOBAAttentionBackend):
            self.attn_metadata_builder_cls = (
                self.stage.attn_backend.get_builder_cls())
            if self.attn_metadata_builder_cls is not None:
                self.attn_metadata_builder = self.attn_metadata_builder_cls()
                moba_params = fastvideo_args.moba_config.copy()
                moba_params.update({
                    "current_timestep":
                    step_idx,
                    "raw_latent_shape":
                    batch.raw_latent_shape[2:5],
                    "patch_size":
                    fastvideo_args.pipeline_config.dit_config.patch_size,
                    "device":
                    get_local_torch_device(),
                })
                attn_metadata = self.attn_metadata_builder.build(**moba_params)
                assert attn_metadata is not None, (
                    "attn_metadata cannot be None")
            else:
                attn_metadata = None
        else:
            attn_metadata = None

        with torch.autocast(device_type="cuda",
                            dtype=target_dtype,
                            enabled=autocast_enabled):
            batch.is_cfg_negative = False
            with set_forward_context(
                    current_timestep=step_idx,
                    attn_metadata=attn_metadata,
                    forward_batch=batch,
            ):
                noise_pred = current_model(
                    model_inputs.latent_model_input,
                    state.prompt_embeds,
                    model_inputs.timestep,
                    guidance=guidance_expand,
                    **image_kwargs,
                    **pos_cond_kwargs,
                    **action_kwargs,
                )

            if state.do_cfg:
                batch.is_cfg_negative = True
                with set_forward_context(
                        current_timestep=step_idx,
                        attn_metadata=attn_metadata,
                        forward_batch=batch,
                ):
                    noise_pred_uncond = current_model(
                        model_inputs.latent_model_input,
                        state.negative_prompt_embeds,
                        model_inputs.timestep,
                        guidance=guidance_expand,
                        **image_kwargs,
                        **neg_cond_kwargs,
                        **action_kwargs,
                    )

                noise_pred_text = noise_pred
                noise_pred = noise_pred_uncond + current_guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                if state.guidance_rescale > 0.0:
                    noise_pred = self.stage.rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=state.guidance_rescale,
                    )

        return noise_pred

    def cfg_combine(self, state: StrategyState,
                    noise_pred: torch.Tensor) -> torch.Tensor:
        return noise_pred

    def scheduler_step(self, state: StrategyState, noise_pred: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        if state.extra.get("skip_step", False):
            return state.latents

        batch = state.extra["batch"]
        extra_step_kwargs = state.extra["extra_step_kwargs"]
        latents = self.stage.scheduler.step(
            noise_pred,
            t,
            state.latents,
            **extra_step_kwargs,
            return_dict=False,
        )[0]

        if (state.extra["ti2v_mask"] is not None
                and batch.pil_image is not None):
            mask2 = state.extra["ti2v_mask"]
            z = state.extra["ti2v_z"]
            latents = latents.squeeze(0)
            latents = (1. - mask2) * z + mask2 * latents

        if state.extra["trajectory_latents"] is not None:
            state.extra["trajectory_timesteps"].append(t)
            state.extra["trajectory_latents"].append(latents)

        progress_bar = state.extra["progress_bar"]
        step_idx = state.extra["step_idx"]
        num_warmup_steps = state.extra["num_warmup_steps"]
        timesteps = state.timesteps
        if step_idx == len(timesteps) - 1 or (
            (step_idx + 1) > num_warmup_steps and
            (step_idx + 1) % self.stage.scheduler.order == 0
                and progress_bar is not None):
            progress_bar.update()

        return latents

    def postprocess(self, state: StrategyState) -> ForwardBatch:
        batch = state.extra["batch"]
        fastvideo_args = state.extra["fastvideo_args"]
        progress_bar = state.extra.get("progress_bar")

        if state.extra["trajectory_latents"]:
            trajectory_tensor = torch.stack(state.extra["trajectory_latents"],
                                            dim=1)
            trajectory_timesteps_tensor = torch.stack(
                state.extra["trajectory_timesteps"], dim=0)
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        batch.latents = state.latents

        if fastvideo_args.dit_layerwise_offload:
            mgr = getattr(self.stage.transformer, "_layerwise_offload_manager",
                          None)
            if mgr is not None and getattr(mgr, "enabled", False):
                mgr.release_all()
            if self.stage.transformer_2 is not None:
                mgr2 = getattr(self.stage.transformer_2,
                               "_layerwise_offload_manager", None)
                if mgr2 is not None and getattr(mgr2, "enabled", False):
                    mgr2.release_all()

        if (st_attn_available
                and self.stage.attn_backend == SlidingTileAttentionBackend
                and fastvideo_args.STA_mode == STA_Mode.STA_SEARCHING):
            self.stage.save_sta_search_results(batch)

        pipeline = self.stage.pipeline() if self.stage.pipeline else None
        if torch.backends.mps.is_available():
            logger.info("Memory before deallocating transformer: %s",
                        torch.mps.current_allocated_memory())
            del self.stage.transformer
            if pipeline is not None and "transformer" in pipeline.modules:
                del pipeline.modules["transformer"]
            fastvideo_args.model_loaded["transformer"] = False
            logger.info("Memory after deallocating transformer: %s",
                        torch.mps.current_allocated_memory())

        if progress_bar is not None:
            progress_bar.close()

        return batch
