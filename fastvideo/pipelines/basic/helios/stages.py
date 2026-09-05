# SPDX-License-Identifier: Apache-2.0
"""Model-specific stages for Helios-Distilled T2V inference."""

import math
import weakref

import torch
import torch.nn.functional as F

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.hooks.activation_trace import trace_step
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import TransformerLoader, VAELoader
from fastvideo.pipelines.basic.helios.pipeline_utils import (
    _randn_tensor,
    build_helios_frame_indices,
    calculate_shift,
    downsample_to_pyramid_base,
    get_generated_pixel_frames,
    get_num_latent_chunks,
    sample_block_noise,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.decoding import DecodingStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.validators import StageValidators as V
from fastvideo.pipelines.stages.validators import VerificationResult
from fastvideo.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class HeliosInputValidationStage(InputValidationStage):
    """Validate the intentionally narrow first Helios contribution: T2V."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = super().forward(batch, fastvideo_args)
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise ValueError("Helios T2V expects scalar height and width")
        if batch.height % 64 or batch.width % 64:
            raise ValueError("Helios height and width must be divisible by 64 for two pyramid downsamples")
        if not isinstance(batch.num_frames, int) or batch.num_frames <= 0:
            raise ValueError(f"Helios num_frames must be a positive integer, got {batch.num_frames}")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("Helios currently supports num_videos_per_prompt=1")
        if isinstance(batch.prompt, list) and len(batch.prompt) != 1:
            raise ValueError("Helios currently accepts one prompt per FastVideo request")
        if batch.image_path is not None or batch.video_path is not None or batch.pil_image is not None:
            raise ValueError("This initial Helios contribution supports T2V only")
        if batch.latents is not None:
            raise ValueError("Pre-generated Helios chunk latents are not supported yet")

        steps = batch.pyramid_num_inference_steps_list
        if steps is None:
            steps = [batch.num_inference_steps] * 3
        if len(steps) != 3 or any(not isinstance(value, int) or value <= 0 for value in steps):
            raise ValueError(f"Helios requires three positive pyramid step counts, got {steps}")
        batch.pyramid_num_inference_steps_list = list(steps)

        history_sizes = batch.history_sizes or [16, 2, 1]
        if len(history_sizes) != 3 or any(not isinstance(value, int) or value <= 0 for value in history_sizes):
            raise ValueError(f"Helios requires three positive history sizes, got {history_sizes}")
        batch.history_sizes = sorted(history_sizes, reverse=True)
        if batch.num_latent_frames_per_chunk != 9:
            raise ValueError("Helios-Distilled requires num_latent_frames_per_chunk=9")
        if not batch.keep_first_frame:
            raise ValueError("The initial Helios T2V port requires keep_first_frame=True")
        if batch.is_skip_first_chunk:
            raise ValueError("is_skip_first_chunk is only meaningful for conditioned Helios modes")
        if batch.zero_steps < 0:
            raise ValueError("Helios zero_steps must be non-negative")
        return batch


class HeliosPyramidDenoisingStage(PipelineStage):
    """Generate autoregressive latent chunks with the three-stage DMD sampler."""

    performance_component_metric = "dit_time_s"

    def __init__(self, transformer, scheduler, pipeline=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.pipeline = weakref.ref(pipeline) if pipeline else None

    @staticmethod
    def _scheduler_value(scheduler, name: str, default):
        value = getattr(scheduler.config, name, None)
        if value is None and hasattr(scheduler.config, "get"):
            value = scheduler.config.get(name, default)
        return default if value is None else value

    def _load_or_move_transformer(self, fastvideo_args: FastVideoArgs):
        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["transformer"]:
            self.transformer = TransformerLoader().load(
                fastvideo_args.model_paths["transformer"],
                fastvideo_args,
            )
            if pipeline is not None:
                pipeline.add_module("transformer", self.transformer)
            fastvideo_args.model_loaded["transformer"] = True

        if (fastvideo_args.dit_cpu_offload and not fastvideo_args.dit_layerwise_offload
                and not fastvideo_args.use_fsdp_inference and next(self.transformer.parameters()).device.type == "cpu"):
            self.transformer.to(get_local_torch_device())

    def _transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        histories: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch: ForwardBatch,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        history_short, history_mid, history_long = histories
        current_indices, short_indices, mid_indices, long_indices = indices
        with set_forward_context(
                current_timestep=int(timestep[0].item()),
                attn_metadata=None,
                forward_batch=batch,
        ):
            return self.transformer(
                hidden_states=latents.to(target_dtype),
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=current_indices,
                indices_latents_history_short=short_indices,
                indices_latents_history_mid=mid_indices,
                indices_latents_history_long=long_indices,
                latents_history_short=history_short.to(target_dtype),
                latents_history_mid=history_mid.to(target_dtype),
                latents_history_long=history_long.to(target_dtype),
            )

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        self._load_or_move_transformer(fastvideo_args)
        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.dit_precision]
        prompt_embeds = batch.prompt_embeds[0].to(device=device, dtype=target_dtype)
        negative_prompt_embeds = None
        if batch.do_classifier_free_guidance:
            if not batch.negative_prompt_embeds:
                raise ValueError("Helios CFG requires negative prompt embeddings")
            negative_prompt_embeds = batch.negative_prompt_embeds[0].to(
                device=device,
                dtype=target_dtype,
            )

        batch_size = prompt_embeds.shape[0]
        assert isinstance(batch.height, int) and isinstance(batch.width, int)
        assert isinstance(batch.num_frames, int)
        assert batch.generator is not None
        assert batch.history_sizes is not None
        assert batch.pyramid_num_inference_steps_list is not None

        vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
        spatial_scale = vae_arch.scale_factor_spatial
        temporal_scale = vae_arch.scale_factor_temporal
        latent_height = batch.height // spatial_scale
        latent_width = batch.width // spatial_scale
        num_channels = self.transformer.in_channels
        chunk_size = batch.num_latent_frames_per_chunk
        num_chunks = get_num_latent_chunks(
            batch.num_frames,
            chunk_size,
            temporal_scale,
        )
        history_sizes = list(batch.history_sizes)
        history_frame_count = sum(history_sizes)
        history_latents = torch.zeros(
            batch_size,
            num_channels,
            history_frame_count,
            latent_height,
            latent_width,
            device=device,
            dtype=torch.float32,
        )
        indices = build_helios_frame_indices(
            history_sizes,
            chunk_size,
            batch.keep_first_frame,
            device,
        )
        latent_chunks: list[torch.Tensor] = []
        first_frame_latent: torch.Tensor | None = None
        patch_size = tuple(self.transformer.patch_size)
        num_stages = len(batch.pyramid_num_inference_steps_list)
        global_step_index = 0

        logger.info(
            "Helios sampling %d chunk(s), %d latent frames each, pyramid steps=%s",
            num_chunks,
            chunk_size,
            batch.pyramid_num_inference_steps_list,
        )
        for chunk_index in range(num_chunks):
            history_long, history_mid, history_one = history_latents[:, :, -history_frame_count:].split(history_sizes,
                                                                                                        dim=2)
            if first_frame_latent is None:
                prefix = torch.zeros(
                    batch_size,
                    num_channels,
                    1,
                    latent_height,
                    latent_width,
                    device=device,
                    dtype=history_one.dtype,
                )
            else:
                prefix = first_frame_latent
            history_short = torch.cat([prefix, history_one], dim=2)
            histories = (history_short, history_mid, history_long)

            latents = _randn_tensor(
                (
                    batch_size,
                    num_channels,
                    chunk_size,
                    latent_height,
                    latent_width,
                ),
                generator=batch.generator,
                device=device,
            )
            latents = downsample_to_pyramid_base(latents, num_stages)
            start_points = [latents]

            for stage_index, stage_steps in enumerate(batch.pyramid_num_inference_steps_list):
                image_seq_len = math.prod(latents.shape[-3:]) // math.prod(patch_size)
                mu = calculate_shift(
                    image_seq_len,
                    self._scheduler_value(self.scheduler, "base_image_seq_len", 256),
                    self._scheduler_value(self.scheduler, "max_image_seq_len", 4096),
                    self._scheduler_value(self.scheduler, "base_shift", 0.5),
                    self._scheduler_value(self.scheduler, "max_shift", 1.15),
                )
                self.scheduler.set_timesteps(
                    stage_steps,
                    stage_index,
                    device=device,
                    mu=mu,
                    is_amplify_first_chunk=(batch.is_amplify_first_chunk and chunk_index == 0),
                )
                timesteps = self.scheduler.timesteps

                if stage_index > 0:
                    batch_count, channels, frames, height, width = latents.shape
                    flattened = latents.permute(0, 2, 1, 3, 4).reshape(
                        batch_count * frames,
                        channels,
                        height,
                        width,
                    )
                    flattened = F.interpolate(
                        flattened,
                        size=(height * 2, width * 2),
                        mode="nearest",
                    )
                    latents = flattened.reshape(
                        batch_count,
                        frames,
                        channels,
                        height * 2,
                        width * 2,
                    ).permute(0, 2, 1, 3, 4)

                    original_signal = 1 - self.scheduler.ori_start_sigmas[stage_index]
                    gamma = self.scheduler.config.gamma
                    alpha = 1 / (math.sqrt(1 + 1 / gamma) * (1 - original_signal) + original_signal)
                    beta = alpha * (1 - original_signal) / math.sqrt(gamma)
                    noise = sample_block_noise(
                        self.scheduler,
                        tuple(latents.shape),
                        patch_size,
                        device,
                        batch.generator,
                    ).to(dtype=target_dtype)
                    latents = alpha * latents + beta * noise
                    start_points.append(latents)

                for step_index, timestep_value in enumerate(timesteps):
                    timestep = timestep_value.expand(batch_size).to(torch.int64)
                    with trace_step(global_step_index):
                        noise_pred = self._transformer_forward(
                            latents,
                            timestep,
                            prompt_embeds,
                            histories,
                            indices,
                            batch,
                            target_dtype,
                        )
                        if batch.do_classifier_free_guidance:
                            assert negative_prompt_embeds is not None
                            noise_uncond = self._transformer_forward(
                                latents,
                                timestep,
                                negative_prompt_embeds,
                                histories,
                                indices,
                                batch,
                                target_dtype,
                            )
                            noise_pred = noise_uncond + batch.guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(
                        noise_pred,
                        timestep_value,
                        latents,
                        generator=(batch.generator[0] if isinstance(batch.generator, list) else batch.generator),
                        return_dict=False,
                        cur_sampling_step=step_index,
                        dmd_noisy_tensor=start_points[stage_index],
                        dmd_sigmas=self.scheduler.sigmas,
                        dmd_timesteps=self.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]
                    global_step_index += 1

            if first_frame_latent is None:
                first_frame_latent = latents[:, :, :1]
            history_latents = torch.cat([history_latents, latents], dim=2)
            latent_chunks.append(latents)
            logger.info("Helios completed latent chunk %d/%d", chunk_index + 1, num_chunks)

        batch.helios_latent_chunks = latent_chunks
        batch.latents = torch.cat(latent_chunks, dim=2)
        batch.timesteps = self.scheduler.timesteps

        if fastvideo_args.dit_layerwise_offload:
            manager = getattr(self.transformer, "_layerwise_offload_manager", None)
            if manager is not None and getattr(manager, "enabled", False):
                manager.release_all()
        elif (fastvideo_args.dit_cpu_offload and not fastvideo_args.use_fsdp_inference
              and next(self.transformer.parameters()).device.type == "cuda"):
            self.transformer.to("cpu")
        return batch

    def verify_input(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result

    def verify_output(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result


class HeliosChunkDecodingStage(DecodingStage):
    """Decode each 9-latent chunk independently, matching official Helios."""

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if fastvideo_args.output_type == "latent":
            assert batch.latents is not None
            batch.output = batch.latents.detach().to(dtype=torch.float32, device="cpu")
            batch.latents = None
            batch.helios_latent_chunks = None
            return batch

        pipeline = self.pipeline() if self.pipeline else None
        if not fastvideo_args.model_loaded["vae"]:
            self.vae = VAELoader().load(fastvideo_args.model_paths["vae"], fastvideo_args)
            if pipeline is not None:
                pipeline.add_module("vae", self.vae)
            fastvideo_args.model_loaded["vae"] = True

        latent_chunks = batch.helios_latent_chunks
        if not isinstance(latent_chunks, list) or not latent_chunks:
            raise ValueError("Helios decoding requires non-empty helios_latent_chunks")
        decoded_chunks = [self.decode(chunk, fastvideo_args) for chunk in latent_chunks]
        frames = torch.cat(decoded_chunks, dim=2)

        temporal_scale = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
        assert isinstance(batch.num_frames, int)
        generated_frames = min(
            batch.num_frames,
            get_generated_pixel_frames(frames.shape[2], temporal_scale),
        )
        batch.output = frames[:, :, :generated_frames].detach().to(dtype=torch.float32, device="cpu")
        batch.latents = None
        batch.helios_latent_chunks = None

        if fastvideo_args.vae_cpu_offload:
            self.vae.to("cpu")
        return batch


__all__ = [
    "HeliosChunkDecodingStage",
    "HeliosInputValidationStage",
    "HeliosPyramidDenoisingStage",
    "build_helios_frame_indices",
    "calculate_shift",
    "downsample_to_pyramid_base",
    "get_num_latent_chunks",
    "get_generated_pixel_frames",
    "sample_block_noise",
]
