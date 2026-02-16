from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
import math
import numpy as np
# import logger

def sde_step_with_logprob(
    self: UniPCMultistepScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
    return_pixel_log_prob: bool = False,
    return_dt_and_std_dev_t: bool = False,
    return_variance_noise_sum: bool = False,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()
        
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]

    self.sigmas = self.sigmas.to(sample.device)
    sigma = self.sigmas[step_index].view(-1, 1, 1, 1, 1)
    sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1, 1)
    sigma_max = self.sigmas[1].item()
    sigma_min = self.sigmas[-1].item()
    dt = sigma_prev - sigma

    # print(f"type(self): {type(self)}")
    # print(f"self.sigmas: {self.sigmas}")
    # print(f"step index: {step_index}")

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    variance_noise_sum_fp64 = 0.0
    if prev_sample is None:
        # When generator is provided (e.g. debug: torch.Generator("cpu").manual_seed(42)), use it for reproducible variance_noise.
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        if return_variance_noise_sum:
            variance_noise_sum_fp64 = float(variance_noise.detach().double().sum().item())
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    # No noise is added during evaluation
    if determistic:
        prev_sample = sample + dt * model_output

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        
    if return_dt_and_std_dev_t:
        if return_variance_noise_sum:
            return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt), variance_noise_sum_fp64
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, torch.sqrt(-1*dt)
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)

def wan_pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "np",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[
        Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
    ] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = False,
    kl_reward: float = 0.0,
    return_pixel_log_prob: bool = False,
    return_debug_metrics: bool = False,
):
    r"""
    The call function to the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        height (`int`, defaults to `480`):
            The height in pixels of the generated image.
        width (`int`, defaults to `832`):
            The width in pixels of the generated image.
        num_frames (`int`, defaults to `81`):
            The number of frames in the generated video.
        num_inference_steps (`int`, defaults to `50`):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        guidance_scale (`float`, defaults to `5.0`):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_videos_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
            generation deterministic.
        latents (`torch.Tensor`, *optional*):
            Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor is generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
            provided, text embeddings are generated from the `prompt` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between `PIL.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
            A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
            each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
            DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
            list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
            The dtype to use for the torch.amp.autocast.

    Examples:

    Returns:
        [`~WanPipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
            the first element is a list with the generated images and the second element is a list of `bool`s
            indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
    """

    if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds,
        negative_prompt_embeds,
        callback_on_step_end_tensor_inputs,
    )

    if num_frames % self.vae_scale_factor_temporal != 1:
        print(
            f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
        )
        num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
    num_frames = max(num_frames, 1)

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    device = self._execution_device

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = self.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    # 4. Prepare timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    # prepare_latents_generator = [torch.Generator("cpu").manual_seed(41)] * batch_size
    prepare_latents_generator = torch.Generator("cpu").manual_seed(41)
    latents = self.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        prepare_latents_generator,
        latents,
    )

    all_latents = [latents]
    all_log_probs = []
    all_kl = []
    if return_debug_metrics:
        debug_sum_initial_latents = latents.to(torch.float64).sum().item()
        debug_sum_intermediate_latents = debug_sum_initial_latents
        debug_sum_model_pred = 0.0
        debug_sum_prompt_embeds = prompt_embeds.to(torch.float64).sum().item()
        debug_sum_negative_prompt_embeds = negative_prompt_embeds.to(torch.float64).sum().item() if negative_prompt_embeds is not None else 0.0
        debug_model_pred_per_step = []
        debug_intermediate_latents_per_step = []
        debug_variance_noise_sum_per_step = []
        # Deterministic variance_noise for debug: same config as FastVideo (seed 42, device cpu).
        generator = torch.Generator("cpu").manual_seed(42)

    # 6. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    # print(timesteps)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            
            latents_ori = latents.clone()
            self._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0])

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)

            if self.do_classifier_free_guidance:
                noise_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if return_debug_metrics:
                s_mp = noise_pred.double().sum().item()
                debug_sum_model_pred += s_mp
                debug_model_pred_per_step.append(s_mp)

            if return_debug_metrics:
                latents, log_prob, prev_latents_mean, std_dev_t, _, variance_noise_sum = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    generator=generator,
                    determistic=determistic,
                    return_pixel_log_prob=return_pixel_log_prob,
                    return_dt_and_std_dev_t=True,
                    return_variance_noise_sum=True,
                )
                debug_intermediate_latents_per_step.append(latents.double().sum().item())
                debug_variance_noise_sum_per_step.append(variance_noise_sum)
            else:
                latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    t.unsqueeze(0),
                    latents.float(),
                    generator=generator,
                    determistic=determistic,
                    return_pixel_log_prob=return_pixel_log_prob
                )
            prev_latents = latents.clone()

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            if return_debug_metrics:
                debug_sum_intermediate_latents += latents.double().sum().item()

            # compute the previous noisy sample x_t -> x_t-1
            # latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            # use kl_reward & is sampling process
            if kl_reward>0 and not determistic:
                latent_model_input = torch.cat([latents_ori] * 2) if self.do_classifier_free_guidance else latents_ori
                with self.transformer.disable_adapter():
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred.to(prompt_embeds.dtype)
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                _, ref_log_prob, ref_prev_latents_mean, ref_std_dev_t = sde_step_with_logprob(
                    self.scheduler, 
                    noise_pred.float(), 
                    t.unsqueeze(0), 
                    latents_ori.float(),
                    prev_sample=prev_latents.float(),
                    determistic=determistic,
                )
                assert std_dev_t == ref_std_dev_t
                kl = (prev_latents_mean - ref_prev_latents_mean)**2 / (2 * std_dev_t**2)
                kl = kl.mean(dim=tuple(range(1, kl.ndim)))
                all_kl.append(kl)
            else:
                # no kl reward, we do not need to compute, just put a pre-position value, kl will be 0
                all_kl.append(torch.zeros(len(latents), device=latents.device))

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            # if XLA_AVAILABLE:
            #     xm.mark_step()

    self._current_timestep = None
    
    if not output_type == "latent":
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
    else:
        video = latents

    if return_debug_metrics:
        video_t = video if isinstance(video, torch.Tensor) else torch.tensor(video)
        debug_sum_decoded = video_t.double().sum().item()
        debug_metrics = {
            "sum_initial_latents_fp64": debug_sum_initial_latents,
            "sum_intermediate_latents_fp64": debug_sum_intermediate_latents,
            "sum_model_pred_fp64": debug_sum_model_pred,
            "sum_prompt_embeds_fp64": debug_sum_prompt_embeds,
            "sum_negative_prompt_embeds_fp64": debug_sum_negative_prompt_embeds,
            "sum_decoded_fp64": debug_sum_decoded,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "batch_size": batch_size,
            "debug_model_pred_per_step": debug_model_pred_per_step,
            "debug_intermediate_latents_per_step": debug_intermediate_latents_per_step,
            "debug_variance_noise_sum_per_step": debug_variance_noise_sum_per_step,
        }
    
    self.maybe_free_model_hooks()

    if not return_dict:
        if return_debug_metrics:
            return (video, all_latents, all_log_probs, all_kl, debug_metrics)
        return (video, all_latents, all_log_probs, all_kl)

    if return_debug_metrics:
        return WanPipelineOutput(frames=video), all_latents, all_log_probs, all_kl, debug_metrics
    return WanPipelineOutput(frames=video), all_latents, all_log_probs, all_kl
