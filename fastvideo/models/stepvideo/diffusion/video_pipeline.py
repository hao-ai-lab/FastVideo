# Copyright 2025 StepFun Inc. All Rights Reserved.

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pickle
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
import asyncio
import os
import json
from fastvideo.models.stepvideo.modules.model import StepVideoModel
from fastvideo.models.stepvideo.diffusion.scheduler import FlowMatchDiscreteScheduler
from fastvideo.models.stepvideo.utils import VideoProcessor
from stepvideo.utils.sliding_block_attention import get_sliding_block_attention_mask
from functools import partial
from torch.nn.attention.flex_attention import flex_attention
from fastvideo.utils.logging_ import main_print

def call_api_gen(url, api, port=8080):
    url = f"http://{url}:{port}/{api}-api"
    import aiohttp

    async def _fn(samples, *args, **kwargs):
        if api == 'vae':
            data = {
                "samples": samples,
            }
        elif api == 'caption':
            data = {
                "prompts": samples,
            }
        else:
            raise Exception(f"Not supported api: {api}...")

        async with aiohttp.ClientSession() as sess:
            data_bytes = pickle.dumps(data)
            async with sess.get(url, data=data_bytes,
                                timeout=12000) as response:
                result = bytearray()
                while not response.content.at_eof():
                    chunk = await response.content.read(1024)
                    result += chunk
                response_data = pickle.loads(result)
        return response_data

    return _fn


@dataclass
class StepVideoPipelineOutput(BaseOutput):
    video: Union[torch.Tensor, np.ndarray]


class StepVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using StepVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        transformer ([`StepVideoModel`]):
            Conditional Transformer to denoise the encoded image latents.
        scheduler ([`FlowMatchDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae_url:
            remote vae server's url.
        caption_url:
            remote caption (stepllm and clip) server's url.
    """

    def __init__(
        self,
        transformer: StepVideoModel,
        scheduler: FlowMatchDiscreteScheduler,
        vae_url: str = '127.0.0.1',
        caption_url: str = '127.0.0.1',
        save_path: str = './results',
        name_suffix: str = '',
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(
            self, "vae", None) else 8
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(
            self, "vae", None) else 16
        self.video_processor = VideoProcessor(save_path, name_suffix)

        self.vae_url = vae_url
        self.caption_url = caption_url
        self.setup_api(self.vae_url, self.caption_url)

    def setup_api(self, vae_url, caption_url):
        self.vae_url = vae_url
        self.caption_url = caption_url
        self.caption = call_api_gen(caption_url, 'caption')
        self.vae = call_api_gen(vae_url, 'vae')
        return self

    def encode_prompt(
        self,
        prompt: str,
        neg_magic: str = '',
        pos_magic: str = '',
    ):
        device = self._execution_device
        prompts = [prompt + pos_magic]
        bs = len(prompts)
        prompts += [neg_magic] * bs

        data = asyncio.run(self.caption(prompts))
        prompt_embeds, prompt_attention_mask, clip_embedding = data['y'].to(
            device), data['y_mask'].to(device), data['clip_embedding'].to(
                device)

        return prompt_embeds, clip_embedding, prompt_attention_mask

    def decode_vae(self, samples):
        samples = asyncio.run(self.vae(samples.cpu()))
        return samples

    def check_inputs(self, num_frames, width, height):
        num_frames = max(num_frames // 17 * 17, 1)
        width = max(width // 16 * 16, 16)
        height = max(height // 16 * 16, 16)
        return num_frames, width, height

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 64,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_frames, width, height = self.check_inputs(num_frames, width,
                                                      height)
        shape = (
            batch_size,
            max(num_frames // 17 * 3, 1),
            num_channels_latents,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )  # b,f,c,h,w
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if generator is None:
            generator = torch.Generator(device=self._execution_device)

        latents = torch.randn(shape,
                              generator=generator,
                              device=device,
                              dtype=dtype)
        return latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 544,
        width: int = 992,
        num_frames: int = 204,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        time_shift: float = 13.0,
        neg_magic: str = "",
        pos_magic: str = "",
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "mp4",
        output_file_name: Optional[str] = "",
        return_dict: bool = True,
        skip_time_steps: int = 10,
        mask_strategy_selected: List[int] = [1,2,6],
        mask_search_files_path: str = None,
        save_path: str = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `544`):
                The height in pixels of the generated image.
            width (`int`, defaults to `992`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `204`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `9.0`):
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            output_file_name(`str`, *optional*`):
                The output mp4 file name.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`StepVideoPipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~StepVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`StepVideoPipelineOutput`] is returned, otherwise a `tuple` is returned
                where the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # 1. Check inputs. Raise error if not correct
        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, prompt_embeds_2, prompt_attention_mask = self.encode_prompt(
            prompt=prompt,
            neg_magic=neg_magic,
            pos_magic=pos_magic,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        prompt_attention_mask = prompt_attention_mask.to(transformer_dtype)
        prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps,
                                     time_shift=time_shift,
                                     device=device)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.bfloat16,
            device,
            generator,
            latents,
        )

        # TODO add mask search

        img_size = (latents.shape[1], latents.shape[3], latents.shape[4])
        text_length = 0 # prompt_attention_mask.sum()
        
        mask_strategy_candidates = ["3,3,3", "6,1,6", "1,6,6", "6,6,1", "3,3,6", "3,6,3", "6,3,3", "6,6,6"]
        full_mask = get_sliding_block_attention_mask([6,6,6], (6, 8, 8), img_size, text_length, self.transformer.device) # 36*48*48
        selected_strategies = []
        # mask_strategy_selected = [0, 1, 2, 3]
        for index in mask_strategy_selected:
            strategy = mask_strategy_candidates[index]
            strategy_list = [int(x) for x in strategy.split(',')]
            selected_strategies.append(strategy_list)
            
        selected_attn_processor = []
        for ms in selected_strategies:
            mask = get_sliding_block_attention_mask(ms, (6, 8, 8), img_size, text_length, self.transformer.device)
            attn_processor = torch.compile(partial(flex_attention, block_mask=mask))
            selected_attn_processor.append(attn_processor)
        full_attn_processor = torch.compile(partial(flex_attention, block_mask=full_mask))
        selected_attn_processor.append(full_attn_processor)

        def read_specific_json_files(folder_path):
            json_contents = []
            
            # List files only in the current directory (no walk)
            files = os.listdir(folder_path)
            # Filter files
            matching_files = [f for f in files if 'mask' in f and f.endswith('.json')]
            main_print(matching_files)
            
            for file_name in matching_files:
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    json_contents.append(data)
                    
            return json_contents
        results = read_specific_json_files(mask_search_files_path)
        
        def average_head_losses(results):
            # Initialize a dictionary to store the averaged results
            averaged_losses = {}
            loss_type = 'L2_loss'
            # Get all loss types (e.g., 'L2_loss')
            averaged_losses[loss_type] = {}
            
            # Get all mask strategies (e.g., '[5, 3, 5]')
            
            for mask_strategy in selected_strategies:
                mask_strategy = str(mask_strategy)
                # Initialize with the shape of the data excluding the prompt dimension
                # Shape will be: [time_step, layer_idx, 24_heads]
                data_shape = np.array(results[0][loss_type][mask_strategy]).shape
                accumulated_data = np.zeros(data_shape)
                
                # Sum across all prompts
                for prompt_result in results:
                    accumulated_data += np.array(prompt_result[loss_type][mask_strategy])
                
                # Average by dividing by number of prompts
                averaged_data = accumulated_data / len(results)
                averaged_losses[loss_type][mask_strategy] = averaged_data

            return averaged_losses
        
        averaged_results = average_head_losses(results)
        selected_strategies.append([6, 6, 6])

        def select_best_mask_strategy(averaged_results, selected_strategies):
            best_masks = {}
            best_masks_save = {}
            loss_type = 'L2_loss'
            
            # Get the shape of time steps and layers
            time_steps = len(averaged_results[loss_type][str(selected_strategies[0])])
            layers = len(averaged_results[loss_type][str(selected_strategies[0])][0])
            
            # Counter for sparsity calculation
            total_tokens = 0  # total number of masked tokens
            total_length = 0  # total sequence length
              # First 10 time steps use full attention
            # Counter for strategy usage
            strategy_counts = {str(strategy): 0 for strategy in selected_strategies}
            head_num = 48
            full_attn_strategy = selected_strategies[-1]  # Last strategy is full attention
            main_print(f"Strategy {full_attn_strategy}, skip first {skip_time_steps} steps ")
            for t in range(time_steps):
                for l in range(layers):
                    for h in range(head_num):
                        if t < skip_time_steps:  # First 10 time steps use full attention
                        # if (l>11 and l<20) or l>=50 :  # First 10 time steps use full attention
                            strategy = full_attn_strategy
                            best_strategy_idx = len(selected_strategies) - 1
                        else:
                            # Get losses for this head across all strategies
                            head_losses = []
                            for strategy in selected_strategies[:-1]:  # Exclude full attention
                                head_losses.append(
                                    averaged_results[loss_type][str(strategy)][t][l][h]
                                )
                            
                            # Find which strategy gives minimum loss
                            best_strategy_idx = np.argmin(head_losses)
                            strategy = selected_strategies[best_strategy_idx]
                        
                        best_masks[f'{t}_{l}_{h}'] = strategy
                        best_masks_save[f'{t}_{l}_{h}'] = strategy
                        
                        # Calculate sparsity
                        nums = strategy  # strategy is already a list of numbers
                        total_tokens += nums[0] * nums[1] * nums[2]  # masked tokens for chosen strategy
                        total_length += 216  # total length always 
                        
                        # Count strategy usage
                        strategy_counts[str(strategy)] += 1
            
            overall_sparsity = 1 - total_tokens / total_length
            main_print(f"Overall sparsity: {overall_sparsity:.4f}")
            main_print("\nStrategy usage counts:")
            total_heads = time_steps * layers * head_num
            for strategy, count in strategy_counts.items():
                main_print(f"Strategy {strategy}: {count} heads ({count/total_heads*100:.2f}%)")
                
            return best_masks, best_masks_save
        def dict_to_3d_list(best_masks, t_max=50, l_max=48, h_max=48):
            result = [[[None for _ in range(h_max)] 
                    for _ in range(l_max)] 
                    for _ in range(t_max)]
            for key, value in best_masks.items():
                t, l, h = map(int, key.split('_'))
                result[t][l][h] = value
            return result
        best_mask_selections, best_masks_save = select_best_mask_strategy(averaged_results, selected_strategies)
        mask_strategy = dict_to_3d_list(best_mask_selections)
        file_path = os.path.join(save_path, 'mask_strategy_select.json')
        with open(file_path, 'w') as f:
            json.dump(best_masks_save, f, indent=4)
        main_print("successfully save mask_strategy")
        #best_mask_selections = None
        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = latent_model_input.to(transformer_dtype)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0]).to(
                    latent_model_input.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    return_dict=False,
                    mask_strategy=mask_strategy[i],
                )
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(model_output=noise_pred,
                                              timestep=t,
                                              sample=latents)

                progress_bar.update()

        if not torch.distributed.is_initialized() or int(
                torch.distributed.get_rank()) == 0:
            if not output_type == "latent":
                video = self.decode_vae(latents)
                video = self.video_processor.postprocess_video(
                    video,
                    output_file_name=output_file_name,
                    output_type=output_type)
            else:
                video = latents

            # Offload all models
            self.maybe_free_model_hooks()

            if not return_dict:
                return (video, )

            return StepVideoPipelineOutput(video=video)
