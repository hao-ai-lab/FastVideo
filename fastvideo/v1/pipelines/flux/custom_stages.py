import importlib

import numpy as np
import torch
from einops import rearrange

from fastvideo.v1.attention import get_attn_backend
from fastvideo.v1.distributed import (get_sequence_model_parallel_rank,
                                      get_sequence_model_parallel_world_size)
from fastvideo.v1.distributed.communication_op import (
    sequence_model_parallel_all_gather)
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.denoising import DenoisingStage
from fastvideo.v1.platforms import _Backend
from fastvideo.v1.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TimestepsPreparationPreStage(PipelineStage):

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def calculate_shift(self,
                        image_seq_len,
                        base_seq_len: int = 256,
                        max_seq_len: int = 4096,
                        base_shift: float = 0.5,
                        max_shift: float = 1.15):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.height is not None
        assert batch.width is not None

        batch.sigmas = np.linspace(
            1.0, 1 / batch.num_inference_steps,
            batch.num_inference_steps) if batch.sigmas is None else batch.sigmas
        spatial_compression_ratio = fastvideo_args.vae_config.arch_config.spatial_compression_ratio
        batch.extra_set_timesteps_kwargs["mu"] = (
            batch.extra_set_timesteps_kwargs.get("mu", None)
            or self.calculate_shift(
                (batch.height // spatial_compression_ratio // 2) *
                (batch.width // spatial_compression_ratio // 2),
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            ))
        return batch


class DenoisingPreprocessingStage(PipelineStage):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        # [B, in_channels // 4, 1, H, W] -> [B, H // 2, W // 2, in_channels]
        assert batch.latents is not None
        b, c, _, h, w = batch.latents.shape
        latents = batch.latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
        batch.latents = latents
        return batch


class DenoisingPostprocessingStage(PipelineStage):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        assert batch.latents is not None
        assert batch.height_latents is not None
        assert batch.width_latents is not None

        latents = batch.latents
        # Skip decoding if output type is latent
        if fastvideo_args.output_type == "latent":
            latents = latents
        else:
            # [B, (H // 2) * (W // 2), in_channels] -> [B, in_channels // 4, 1, H, W]
            b, _, c = latents.shape
            h, w = batch.height_latents, batch.width_latents
            latents = latents.view(b, h // 2, w // 2, c // 4, 2, 2)
            latents = latents.permute(0, 3, 1, 4, 2, 5)
            latents = latents.reshape(b, c // 4, h, w)
            # latents = latents.squeeze(2)

        batch.latents = latents
        return batch


st_attn_available = False
spec = importlib.util.find_spec("st_attn")
if spec is not None:
    st_attn_available = True

    from fastvideo.v1.attention.backends.sliding_tile_attn import (
        SlidingTileAttentionBackend)


class FluxDenoisingStage(DenoisingStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Run the denoising loop.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The batch with denoised latents.
        """
        # Prepare extra step kwargs for scheduler
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {
                "generator": batch.generator,
                "eta": batch.eta
            },
        )

        # Setup precision and autocast settings
        target_dtype = PRECISION_TO_TYPE[fastvideo_args.precision]
        autocast_enabled = (target_dtype != torch.float32
                            ) and not fastvideo_args.disable_autocast

        # Handle sequence parallelism if enabled
        world_size, rank = get_sequence_model_parallel_world_size(
        ), get_sequence_model_parallel_rank()
        sp_group = world_size > 1
        if sp_group:
            latents = rearrange(batch.latents,
                                "b (n s) c -> b n s c",
                                n=world_size).contiguous()
            latents = latents[:, rank, :, :]
            batch.latents = latents
            if batch.image_latent is not None:
                image_latent = rearrange(batch.image_latent,
                                         "b (n s) c -> b n s c",
                                         n=world_size).contiguous()
                image_latent = image_latent[:, rank, :, :]
                batch.image_latent = image_latent

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        # TODO(will): remove this once we add input/output validation for stages
        if timesteps is None:
            raise ValueError("Timesteps must be provided")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order

        # Create 3D list for mask strategy
        def dict_to_3d_list(mask_strategy, t_max=50, l_max=60, h_max=24):
            result = [[[None for _ in range(h_max)] for _ in range(l_max)]
                      for _ in range(t_max)]
            if mask_strategy is None:
                return result
            for key, value in mask_strategy.items():
                t, layer, h = map(int, key.split('_'))
                result[t][layer][h] = value
            return result

        # Prepare image latents and embeddings for I2V generation
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            assert torch.isnan(image_embeds[0]).sum() == 0
            image_embeds = [
                image_embed.to(target_dtype) for image_embed in image_embeds
            ]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "height_latents": batch.height_latents,
                "width_latents": batch.width_latents,
            },
        )

        # Get latents and embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0
        if batch.do_classifier_free_guidance:
            neg_prompt_embeds = batch.negative_prompt_embeds
            assert neg_prompt_embeds is not None
            assert torch.isnan(neg_prompt_embeds[0]).sum() == 0

        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                # Expand latents for I2V
                latent_model_input = latents.to(target_dtype)
                if batch.image_latent is not None:
                    latent_model_input = torch.cat(
                        [latent_model_input, batch.image_latent],
                        dim=1).to(target_dtype)
                assert torch.isnan(latent_model_input).sum() == 0
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # Prepare inputs for transformer
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (torch.tensor(
                    [fastvideo_args.embedded_cfg_scale] *
                    latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=fastvideo_args.device,
                ).to(target_dtype) * 1000.0 if fastvideo_args.embedded_cfg_scale
                                   is not None else None)

                # Predict noise residual
                with torch.autocast(device_type="cuda",
                                    dtype=target_dtype,
                                    enabled=autocast_enabled):

                    # TODO(will-refactor): all of this should be in the stage's init
                    attn_head_size = self.transformer.hidden_size // self.transformer.num_attention_heads
                    self.attn_backend = get_attn_backend(
                        head_size=attn_head_size,
                        dtype=torch.float16,  # TODO(will): hack
                        supported_attention_backends=(
                            _Backend.SLIDING_TILE_ATTN, _Backend.FLASH_ATTN,
                            _Backend.TORCH_SDPA)  # hack
                    )
                    if st_attn_available and self.attn_backend == SlidingTileAttentionBackend:
                        self.attn_metadata_builder_cls = self.attn_backend.get_builder_cls(
                        )
                        if self.attn_metadata_builder_cls is not None:
                            self.attn_metadata_builder = self.attn_metadata_builder_cls(
                            )
                            # TODO(will): clean this up
                            attn_metadata = self.attn_metadata_builder.build(
                                current_timestep=i,
                                forward_batch=batch,
                                fastvideo_args=fastvideo_args,
                            )
                            assert attn_metadata is not None, "attn_metadata cannot be None"
                        else:
                            attn_metadata = None
                    else:
                        attn_metadata = None
                    # TODO(will): finalize the interface. vLLM uses this to
                    # support torch dynamo compilation. They pass in
                    # attn_metadata, vllm_config, and num_tokens. We can pass in
                    # fastvideo_args or training_args, and attn_metadata.
                    with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            # fastvideo_args=fastvideo_args
                    ):
                        # Run transformer
                        noise_pred = self.transformer(
                            latent_model_input,
                            prompt_embeds,
                            t_expand,
                            guidance=guidance_expand,
                            **image_kwargs,
                        )

                    # Apply guidance
                    if batch.do_classifier_free_guidance:
                        with set_forward_context(
                                current_timestep=i,
                                attn_metadata=attn_metadata,
                                # fastvideo_args=fastvideo_args
                        ):
                            # Run transformer
                            noise_pred_uncond = self.transformer(
                                latent_model_input,
                                neg_prompt_embeds,
                                t_expand,
                                guidance=guidance_expand,
                                **image_kwargs,
                            )
                        noise_pred_text = noise_pred
                        noise_pred = noise_pred_uncond + batch.guidance_scale * (
                            noise_pred_text - noise_pred_uncond)

                        # Apply guidance rescale if needed
                        if batch.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = self.rescale_noise_cfg(
                                noise_pred,
                                noise_pred_text,
                                guidance_rescale=batch.guidance_rescale,
                            )

                    # Compute the previous noisy sample
                    latents = self.scheduler.step(noise_pred,
                                                  t,
                                                  latents,
                                                  **extra_step_kwargs,
                                                  return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and
                    (i + 1) % self.scheduler.order == 0
                        and progress_bar is not None):
                    progress_bar.update()

        # Gather results if using sequence parallelism
        if sp_group:
            latents = sequence_model_parallel_all_gather(latents, dim=1)

        # Update batch with final latents
        batch.latents = latents

        if fastvideo_args.use_cpu_offload:
            self.transformer.to('cpu')
            torch.cuda.empty_cache()

        return batch


class ImageOutputStage(PipelineStage):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, batch: ForwardBatch,
                fastvideo_args: FastVideoArgs) -> ForwardBatch:
        output = batch.output
        output = output.unsqueeze(2)
        batch.output = output
        return batch
