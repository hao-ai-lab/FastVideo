# Copyright 2025 The FastVideo Authors. All rights reserved.

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.image_processor import VaeImageProcessor
from fastvideo.pipelines.pipeline_base import DiffusionPipelineBase
# from transformers import CLIPTextModel, CLIPTokenizer, LlamaModel, LlamaTokenizerFast
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.logger import init_logger


# TODO(will): temporary import
from diffusers.models import AutoencoderKL
# from fastvideo.models.encoders.encoder import TextEncoder
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from fastvideo.models.hunyuan.modules import HYVideoDiffusionTransformer
from fastvideo.inference_args import InferenceArgs

logger = init_logger(__name__)


class HunyuanVideoPipeline(DiffusionPipelineBase):
    """
    Pipeline for text-to-video generation using HunyuanVideo.
    This is a concrete implementation of the AbstractDiffusionPipeline.
    
    This implementation assumes the availability of:
    - text_encoder: LlamaModel for prompt encoding
    - tokenizer: LlamaTokenizerFast for prompt tokenization
    - transformer: HunyuanVideoTransformer3DModel for latent diffusion
    - vae: AutoencoderKLHunyuanVideo for encoding/decoding latents
    - scheduler: FlowMatchEulerDiscreteScheduler for diffusion timesteps
    - text_encoder_2: CLIPTextModel for additional prompt encoding
    - tokenizer_2: CLIPTokenizer for additional prompt tokenization
    """
    
    # Overriding this to match hunyuan_video pipeline settings
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    is_video_pipeline = True
    
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        inferece_args: InferenceArgs = None,
    ):
        super().__init__()

        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.inference_args = inferece_args

        # TODO(will): add scheduler stuff
        
        # Register all the components using the provided method
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )

        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
    
    def check_inputs(self, batch: ForwardBatch, inference_args: InferenceArgs):
        height = batch.height
        width = batch.width
        video_length = batch.num_frames
        vae_ver = inference_args.vae
        prompt = batch.prompt
        prompt_embeds = batch.prompt_embeds
        negative_prompt = batch.negative_prompt
        negative_prompt_embeds = batch.negative_prompt_embeds
        
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(f"`video_length` has to be 1 or a multiple of 4 but is {video_length}.")
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(f"`video_length` has to be 1 or a multiple of 8 but is {video_length}.")

        # if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
        #     raise ValueError(f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
        #                      f" {type(callback_steps)}.")
        # if callback_on_step_end_tensor_inputs is not None and not all(k in self._callback_tensor_inputs
        #                                                               for k in callback_on_step_end_tensor_inputs):
        #     raise ValueError(
        #         f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
        #     )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                             f" {negative_prompt_embeds}. Please make sure to only forward one of the two.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}.")
            