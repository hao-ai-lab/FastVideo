"""
Denoising stage for diffusion pipelines.
"""

import inspect
import torch
from typing import Optional, Dict, Any, List
from tqdm.auto import tqdm
from einops import rearrange

from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.models.hunyuan.constants import PRECISION_TO_TYPE
from fastvideo.utils.parallel_states import nccl_info, get_sequence_parallel_state
from fastvideo.utils.communications import all_gather
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class DenoisingStage(PipelineStage):
    """
    Stage for running the denoising loop in diffusion pipelines.
    
    This stage handles the iterative denoising process that transforms
    the initial noise into the final output.
    """
    
    def _call_implementation(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Run the denoising loop.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
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
        target_dtype = PRECISION_TO_TYPE[inference_args.precision]
        autocast_enabled = (target_dtype != torch.float32) and not inference_args.disable_autocast

        # Handle sequence parallelism if enabled
        world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
        sp_group = True
        if sp_group:
            latents = rearrange(batch.latents, "b t (n s) h w -> b t n s h w", n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]
            batch.latents = latents

        # Get timesteps and calculate warmup steps
        timesteps = batch.timesteps
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Create 3D list for mask strategy
        def dict_to_3d_list(mask_strategy, t_max=50, l_max=60, h_max=24):
            result = [[[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)]
            if mask_strategy is None:
                return result
            for key, value in mask_strategy.items():
                t, l, h = map(int, key.split('_'))
                result[t][l][h] = value
            return result

        mask_strategy = dict_to_3d_list(batch.mask_strategy)
        
        # Get latents and embeddings
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        prompt_embeds_2 = batch.prompt_embeds_2
        prompt_mask = batch.attention_mask
        prompt_mask_2 = batch.attention_mask_2
        
        # Run denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, 'interrupt') and self.interrupt:
                    continue

                # Expand latents for classifier-free guidance
                latent_model_input = (torch.cat([latents] * 2) if batch.do_classifier_free_guidance else latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare inputs for transformer
                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (torch.tensor(
                    [inference_args.embedded_cfg_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=batch.device,
                ).to(target_dtype) * 1000.0 if inference_args.embedded_cfg_scale is not None else None)
                
                # Predict noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    # Prepare encoder hidden states
                    if prompt_embeds_2 is not None and prompt_embeds_2.shape[-1] != prompt_embeds.shape[-1]:
                        prompt_embeds_2 = torch.nn.functional.pad(
                            prompt_embeds_2,
                            (0, prompt_embeds.shape[2] - prompt_embeds_2.shape[1]),
                            value=0,
                        ).unsqueeze(1)
                    total_length = prompt_mask.sum()
                    # TODO(PY): move no padding logic to text encoder
                    prompt_embeds = prompt_embeds[:, :total_length, :]
                    encoder_hidden_states = torch.cat([prompt_embeds_2, prompt_embeds], dim=1) if prompt_embeds_2 is not None else prompt_embeds
                    
                    # Run transformer
                    noise_pred = self.transformer(
                        latent_model_input,
                        encoder_hidden_states,
                        t_expand,
                        # prompt_mask,
                        # mask_strategy=mask_strategy[i],
                        guidance=guidance_expand,
                        # return_dict=False,
                    )

                # Apply guidance
                if batch.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + batch.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Apply guidance rescale if needed
                    if batch.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = self.rescale_noise_cfg(
                            noise_pred,
                            noise_pred_text,
                            guidance_rescale=batch.guidance_rescale,
                        )

                # Compute the previous noisy sample
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        # Gather results if using sequence parallelism
        if get_sequence_parallel_state():
            latents = all_gather(latents, dim=2)
            
        # Update batch with final latents
        batch.latents = latents
        
        return batch
    
    def prepare_extra_func_kwargs(self, func, kwargs):
        """
        Prepare extra kwargs for the scheduler step.
        
        Args:
            func: The function to prepare kwargs for.
            kwargs: The kwargs to prepare.
            
        Returns:
            The prepared kwargs.
        """
        extra_step_kwargs = {}
        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs
    
    def progress_bar(self, iterable=None, total=None):
        """
        Create a progress bar for the denoising process.
        
        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.
            
        Returns:
            A tqdm progress bar.
        """
        return tqdm(iterable=iterable, total=total)
    
    def rescale_noise_cfg(self, noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale noise prediction according to guidance_rescale.
        
        Based on findings of "Common Diffusion Noise Schedules and Sample Steps are Flawed"
        (https://arxiv.org/pdf/2305.08891.pdf), Section 3.4.
        
        Args:
            noise_cfg: The noise prediction with guidance.
            noise_pred_text: The text-conditioned noise prediction.
            guidance_rescale: The guidance rescale factor.
            
        Returns:
            The rescaled noise prediction.
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # Rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # Mix with the original results from guidance by factor guidance_rescale
        noise_cfg = (guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg)
        return noise_cfg 