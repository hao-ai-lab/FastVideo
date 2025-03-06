import inspect
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm   

from diffusers.utils.torch_utils import randn_tensor

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.inference_args import InferenceArgs
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from fastvideo.models.hunyuan.constants import PRECISION_TO_TYPE
from fastvideo.distributed.parallel_state import get_sp_group
from fastvideo.logger import init_logger
from einops import rearrange
from diffusers.utils import BaseOutput
import numpy as np

logger = init_logger(__name__)


@dataclass
class DiffusionPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]

class DiffusionPipelineBase(ABC):
    """
    Abstract base class for diffusion pipelines that provides a common structure.
    This class assumes that all abstract components are implemented by child classes.
    
    The pipeline implements a general diffusion process with the following steps:
    1. Input validation and preparation
    2. Encoding the input prompt(s)
    3. Preparing timesteps for the diffusion process
    4. Preparing initial latent variables
    5. Running the denoising loop
    6. Decoding the results
    
    Child classes should implement the abstract methods to provide the specific
    functionality needed for their particular use case.
    """
    
    # Define configuration properties
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    model_cpu_offload_seq = None  # Should be defined by child classes if needed
    is_video_pipeline: bool = False  # To be overridden by video pipelines
    
    def __init__(self):
        pass
    
    @property
    def device(self) -> torch.device:
        """Returns the device on which the pipeline is running."""
        for module in self.components.values():
            if isinstance(module, torch.nn.Module):
                return next(module.parameters()).device
        return torch.device("cpu")
    
    @property
    def components(self) -> Dict[str, Any]:
        """Returns all the components of the pipeline."""
        components = {}
        for name, value in self.__dict__.items():
            if not name.startswith("_"):
                components[name] = value
        return components
    
    @property
    def num_timesteps(self):
        """The number of timesteps in the current generation process."""
        return self._num_timesteps
    
    @property
    def attention_kwargs(self):
        """Additional keyword arguments for attention modules."""
        return self._attention_kwargs
    
    @property
    def current_timestep(self):
        """The current timestep in the generation process."""
        return self._current_timestep
    
    @property
    def interrupt(self):
        """Whether the generation has been interrupted."""
        # TODO(will): add interrupt and see which models use it from diffuers
        # and why
        return False
    
    @property
    def _execution_device(self):
        """The device used for execution."""
        return self.device
    
    def register_modules(self, **kwargs):
        """
        Register components of the pipeline.
        
        Args:
            **kwargs: Components to register, as name=component pairs.
        """
        for name, module in kwargs.items():
            setattr(self, name, module)
    
    @abstractmethod
    def check_inputs(self, batch: ForwardBatch, inference_args: InferenceArgs):
        """
        Validate inputs to ensure they meet the requirements for the pipeline.
        Raise appropriate errors for invalid inputs.
        
        Args:
            **kwargs: Input parameters to validate.
        """
        raise NotImplementedError

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs
    
    def encode_prompt(
        self,
        batch: ForwardBatch,
        text_encoder: Optional[TextEncoder] = None,
        is_secondary: bool = False,
    ) -> ForwardBatch:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            batch (ForwardBatch):
                The batch containing text data to encode
            text_encoder (TextEncoder, *optional*):
                The text encoder to use. If None, uses self.text_encoder
            is_secondary (bool, *optional*):
                Whether this is the secondary text encoder. Determines which attributes
                to set on the batch. Default: False
                
        Returns:
            ForwardBatch: The batch with encoded prompt embeddings
        """
        if text_encoder is None:
            text_encoder = self.text_encoder
        
        prompt: Union[str, List[str]] = batch.prompt
        device: torch.device = batch.device
        num_videos_per_prompt: int = batch.num_videos_per_prompt
        data_type: str = batch.data_type
        
        # Get the right prompt embeds and attention masks based on whether this is primary or secondary
        if is_secondary:
            prompt_embeds = batch.prompt_embeds_2
            attention_mask = batch.attention_mask_2
        else:
            prompt_embeds = batch.prompt_embeds
            attention_mask = batch.attention_mask

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            # if isinstance(self, TextualInversionLoaderMixin):
            #     prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
            prompt_embeds = prompt_outputs.hidden_state
            # TODO(will): support clip_skip

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # Set the appropriate attributes based on whether this is primary or secondary
        if is_secondary:
            batch.prompt_embeds_2 = prompt_embeds
            batch.attention_mask_2 = attention_mask
        else:
            batch.prompt_embeds = prompt_embeds
            batch.attention_mask = attention_mask

        return batch
    
    def maybe_apply_classifier_free_guidance(self, batch: ForwardBatch):
        """Concatenate negative and positive prompt embeddings for classifier-free guidance."""
        if not batch.do_classifier_free_guidance:
            return batch
        
        logger.info(f"batch.negative_prompt_embeds: {batch.negative_prompt_embeds}")
        logger.info(f"do_classifier_free_guidance: {batch.do_classifier_free_guidance}")
        logger.info(f"cfg_scale: {batch.guidance_scale}")
        assert batch.negative_prompt_embeds is not None, (
            "Negative prompt embeddings are required for classifier-free guidance"
        )
        
        # Concatenate primary embeddings and masks
        batch.prompt_embeds = torch.cat(
            [batch.negative_prompt_embeds, batch.prompt_embeds]
        )
        if batch.attention_mask is not None:
            batch.attention_mask = torch.cat(
                [batch.negative_attention_mask, batch.attention_mask]
            )
        
        # Concatenate secondary embeddings and masks if present
        if batch.prompt_embeds_2 is not None:
            batch.prompt_embeds_2 = torch.cat(
                [batch.negative_prompt_embeds_2, batch.prompt_embeds_2]
            )
        if batch.attention_mask_2 is not None:
            batch.attention_mask_2 = torch.cat(
                [batch.negative_attention_mask_2, batch.attention_mask_2]
            )
        
        return batch
    
    def prepare_latents(self, batch: ForwardBatch) -> ForwardBatch:
        """
        Prepare the initial latent variables for the diffusion process.
        
        Args:
            **kwargs: Input parameters for latent preparation.
            
        Returns:
            Initial latent variables.
        """

        batch_size = self._get_batch_size(batch)

        batch_size *= batch.num_videos_per_prompt

        dtype                    = batch.prompt_embeds.dtype  # torch.dtype
        device: torch.device     = batch.device
        generator: torch.Generator = batch.generator
        latents: Optional[torch.Tensor]      = batch.latents
        num_channels_latents: int            = batch.num_channels_latents
        num_frames: int                    = batch.num_frames
        height: int                          = batch.height
        width: int                           = batch.width

        num_channels_latents = self.transformer.config.in_channels
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        logger.info(f"shape: {shape}")
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators.")

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            
        batch.latents = latents
        return batch

    def retrieve_timesteps(
        self,
        batch: ForwardBatch,
        **kwargs,
    ):
        """
        Prepares the timestep sequence for the diffusion process based on the scheduler.
        
        Args:
            scheduler: The scheduler to get timesteps from.
            num_inference_steps: The number of diffusion steps.
            device: The device to move timesteps to.
            timesteps: Custom timesteps to override the scheduler's default.
            sigmas: Custom sigmas to override the scheduler's default.
            **kwargs: Additional arguments for the scheduler.
            
        Returns:
            Tuple of timesteps tensor and number of inference steps.
        """
        scheduler = self.scheduler
        device = batch.device
        num_inference_steps = batch.num_inference_steps
        timesteps = batch.timesteps
        sigmas = batch.sigmas

        if batch.timesteps is not None and batch.sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if batch.timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler.")
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler.")
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps
    
    def progress_bar(self, iterable=None, total=None):
        """
        Provides a progress bar for the denoising process.
        
        Args:
            iterable: The iterable to iterate over.
            total: The total number of items.
            
        Returns:
            A tqdm progress bar.
        """
        return tqdm(iterable=iterable, total=total)
    
    def set_progress_bar_config(self, **kwargs):
        """
        Configure the progress bar with additional parameters.
        
        Args:
            **kwargs: Arguments to pass to tqdm.
        """
        tqdm.set_options(**kwargs)

    def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = (guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg)
        return noise_cfg
    
    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ):
        # TODO(will): see if callbacks are needed
        # if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        #     callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs
        self.check_inputs(batch, inference_args=inference_args)

        # 2. Encode with text encoders
        batch = self.encode_prompt(batch=batch, text_encoder=self.text_encoder, is_secondary=False)
        
        # Encode with secondary text encoder if available
        if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            batch = self.encode_prompt(batch=batch, text_encoder=self.text_encoder_2, is_secondary=True)

        # 3. Apply classifier-free guidance if needed
        batch = self.maybe_apply_classifier_free_guidance(batch)

  
        # 4. Prepare timesteps
        n_tokens = batch.n_tokens
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(self.scheduler.set_timesteps,
                                                                    {"n_tokens": n_tokens})
        timesteps, num_inference_steps = self.retrieve_timesteps(batch, **extra_set_timesteps_kwargs)
        
        batch = self.adjust_video_length(batch, inference_args)

        # 5. Prepare latent variables
        batch = self.prepare_latents(batch)

        from fastvideo.utils.parallel_states import nccl_info
        world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
        # world_size = 1
        # rank = 0
        sp_group = True
        # sp_group = get_sp_group()
        if sp_group:
            # world_size = sp_group.world_size
            latents = rearrange(batch.latents, "b t (n s) h w -> b t n s h w", n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]
            batch.latents = latents

        # self.prepare_
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
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
        vae_dtype = PRECISION_TO_TYPE[inference_args.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not inference_args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        def dict_to_3d_list(mask_strategy, t_max=50, l_max=60, h_max=24):
            result = [[[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)]
            if mask_strategy is None:
                return result
            for key, value in mask_strategy.items():
                t, l, h = map(int, key.split('_'))
                result[t][l][h] = value
            return result

        mask_strategy = dict_to_3d_list(batch.mask_strategy)
        
        latents = batch.latents
        prompt_embeds = batch.prompt_embeds
        prompt_embeds_2 = batch.prompt_embeds_2
        prompt_mask = batch.attention_mask
        prompt_mask_2 = batch.attention_mask_2
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (torch.cat([latents] * 2) if batch.do_classifier_free_guidance else latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (torch.tensor(
                    [inference_args.embedded_cfg_scale] * latent_model_input.shape[0],
                    dtype=torch.float32,
                    device=batch.device,
                ).to(target_dtype) * 1000.0 if inference_args.embedded_cfg_scale is not None else None)
                # predict the noise residual
                with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                    # concat prompt_embeds_2 and prompt_embeds. Mismatch fill with zeros
                    if prompt_embeds_2.shape[-1] != prompt_embeds.shape[-1]:
                        prompt_embeds_2 = F.pad(
                            prompt_embeds_2,
                            (0, prompt_embeds.shape[2] - prompt_embeds_2.shape[1]),
                            value=0,
                        ).unsqueeze(1)
                    encoder_hidden_states = torch.cat([prompt_embeds_2, prompt_embeds], dim=1)
                    noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,
                        encoder_hidden_states,
                        t_expand,
                        prompt_mask,
                        mask_strategy=mask_strategy[i],
                        guidance=guidance_expand,
                        return_dict=False,
                    )[0]

                # perform guidance
                if batch.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + batch.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if batch.do_classifier_free_guidance and batch.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=batch.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # TODO(will): see if callbacks are needed
                # if callback_on_step_end is not None:
                #     callback_kwargs = {}
                #     for k in callback_on_step_end_tensor_inputs:
                #         callback_kwargs[k] = locals()[k]
                #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                #     latents = callback_outputs.pop("latents", latents)
                #     prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                #     negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     step_idx = i // getattr(self.scheduler, "order", 1)
                    #     callback(step_idx, t, latents)
                # batch.latents = latents

        # if sp_group:
        #     latents = sp_group.all_gather(latents, dim=2)
        from fastvideo.utils.communications import all_gather
        from fastvideo.utils.parallel_states import get_sequence_parallel_state
        if get_sequence_parallel_state():
            latents = all_gather(latents, dim=2)

        if not inference_args.output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

            if (hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor):
                latents = (latents / self.vae.config.scaling_factor + self.vae.config.shift_factor)
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                if inference_args.vae_tiling:
                    self.vae.enable_tiling()
                # if inference_args.vae_sp:
                #     self.vae.enable_parallel()
                image = self.vae.decode(latents, return_dict=False, generator=batch.generator)[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        #     return image

        return DiffusionPipelineOutput(videos=image)

    def maybe_free_model_hooks(self):
        r"""
        Method that performs the following:
        - Offloads all components.
        - Removes all model hooks that were added when using `enable_model_cpu_offload`, and then applies them again.
          In case the model has not been offloaded, this function is a no-op.
        - Resets stateful diffusers hooks of denoiser components if they were added with
          [`~hooks.HookRegistry.register_hook`].

        Make sure to add this function to the end of the `__call__` function of your pipeline so that it functions
        correctly when applying `enable_model_cpu_offload`.
        """
        for component in self.components.values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload(device=getattr(self, "_offload_device", "cuda"))

    def _get_batch_size(self, batch):
        """Helper to determine batch size from prompt data"""
        if batch.prompt is not None:
            if isinstance(batch.prompt, str):
                return 1
            elif isinstance(batch.prompt, list):
                return len(batch.prompt)
        assert batch.prompt_embeds is not None
        return batch.prompt_embeds.shape[0]
    
    def adjust_video_length(self, batch: ForwardBatch, inference_args: InferenceArgs):
        """Adjust video length based on VAE version"""
        video_length = batch.num_frames
        vae_ver = inference_args.vae
        if "884" in vae_ver:
            batch.num_frames = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            batch.num_frames = (video_length - 1) // 8 + 1
        return batch