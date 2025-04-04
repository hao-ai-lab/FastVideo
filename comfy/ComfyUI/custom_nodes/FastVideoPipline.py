from __future__ import annotations
import torch
import os
import sys
import json
import hashlib
import traceback
import math
import time
import random
import logging
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.insert(0, "/workspace/FastVideo")


import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import comfy.clip_vision
import comfy.model_management
from comfy.cli_args import args
import importlib
import folder_paths
import latent_preview
import node_helpers

from fastvideo.models.hunyuan.inference import HunyuanVideoSampler
from fastvideo.v1.pipelines.hunyuan.hunyuan_pipeline import HunyuanVideoPipeline
from fastvideo.models.hunyuan.vae import load_vae

from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.stages import (CLIPTextEncodingStage,
                                           ConditioningStage, DecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           LlamaEncodingStage,
                                           TimestepPreparationStage
)
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.sample.v1_fastvideo_inference import initialize_distributed_and_parallelism
from fastvideo.v1.distributed import (init_distributed_environment,
                                      initialize_model_parallel)


MAX_RESOLUTION = 16384

# class HunyuanInferenceArgs:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "model_path": ("STRING", {"default": "/workspace/FastVideo/data/FastHunyuan-diffusers", "tooltip": "Path to the Hunyuan model."}), # default need to be changed
#                 "text_len": ("INT", {"default": 256, "min": 1}),
#                 "text_len_2": ("INT", {"default": 77, "min": 1}),
#                 "text_encoder_precision": (["fp16", "fp32", "bf16"],),
#                 "text_encoder_precision_2": (["fp16", "fp32", "bf16"],),
#                 "hidden_state_skip_layer": ("INT", {"default": 2}),
#                 "device": ("STRING", {"default": "cuda"}),
#                 "use_cpu_offload": ("BOOLEAN", {"default": False}),
#                 "seed": ("INT", {"default": 1024}),
#                 "guidance_scale": ("FLOAT", {"default": 1.0}),
#                 "guidance_rescale": ("FLOAT", {"default": 0.0}),
#                 "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
#                 "scheduler_type": ("STRING", {"default": "euler"}),
#                 "height": ("INT", {"default": 720}),
#                 "width": ("INT", {"default": 1280}),
#                 "num_frames": ("INT", {"default": 117}),
#                 "num_inference_steps": ("INT", {"default": 50}),
#                 "output_path": ("STRING", {"default": "outputs/"}),
#                 "output_type": (["pil"],),
#                 "flow_shift": ("INT", {"default": 7}),
#                 "precision": (["fp16", "fp32", "bf16"],),
#                 "vae_precision": (["fp16", "fp32", "bf16"],),
#                 "vae_tiling": ("BOOLEAN", {"default": True}),
#                 "vae_sp": ("BOOLEAN", {"default": False}),
#                 "flow_solver": ("STRING", {"default": "euler"}),
#                 "denoise_type": ("STRING", {"default": "flow"}),
#                 "num_videos": ("INT", {"default": 1}),
#                 "fps": ("INT", {"default": 24}),
#                 "disable_autocast": ("BOOLEAN", {"default": False}),
#                 "log_level": ("STRING", {"default": "info"}),
#             }
#         }

#     RETURN_TYPES = ("INFERENCE_ARGS",)
#     RETURN_NAMES = ("inference_args",)
#     FUNCTION = "create"
#     CATEGORY = "custom/arguments"

#     def create(
#         self,
#         model_path,
#         text_len,
#         text_len_2,
#         text_encoder_precision,
#         text_encoder_precision_2,
#         hidden_state_skip_layer,
#         device,
#         use_cpu_offload,
#         seed,
#         guidance_scale,
#         guidance_rescale,
#         embedded_cfg_scale,
#         scheduler_type,
#         height,
#         width,
#         num_frames,
#         num_inference_steps,
#         output_path,
#         output_type,
#         flow_shift,
#         precision,
#         vae_precision,
#         vae_tiling,
#         vae_sp,
#         flow_solver,
#         denoise_type,
#         num_videos,
#         fps,
#         disable_autocast,
#         log_level,
#     ):

#         args = InferenceArgs(
#             model_path=model_path,
#             text_len=text_len,
#             text_len_2=text_len_2,
#             text_encoder_precision=text_encoder_precision,
#             text_encoder_precision_2=text_encoder_precision_2,
#             hidden_state_skip_layer=hidden_state_skip_layer,
#             device_str=device,
#             use_cpu_offload=use_cpu_offload,
#             seed=seed,
#             guidance_scale=guidance_scale,
#             guidance_rescale=guidance_rescale,
#             embedded_cfg_scale=embedded_cfg_scale,
#             scheduler_type=scheduler_type,
#             height=height,
#             width=width,
#             num_frames=num_frames,
#             num_inference_steps=num_inference_steps,
#             output_path=output_path,
#             output_type=output_type,
#             flow_shift=flow_shift,
#             precision=precision,
#             vae_precision=vae_precision,
#             vae_tiling=vae_tiling,
#             vae_sp=vae_sp,
#             flow_solver=flow_solver,
#             denoise_type=denoise_type,
#             num_videos=num_videos,
#             fps=fps,
#             disable_autocast=disable_autocast,
#             log_level=log_level,
#         )
#         initialize_distributed_and_parallelism(args)
#         return (args,)

class HunyuanInferenceArgs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "/workspace/FastVideo/data/FastHunyuan-diffusers"}),
                # Model and path configuration
                "trust_remote_code": ("BOOLEAN", {"default": False}),
                "revision": ("STRING", {"default": ""}),
                
                # Parallelism
                "tp_size": ("INT", {"default": 1, "min": 1}),
                "sp_size": ("INT", {"default": 1, "min": 1}),
                "dist_timeout": ("INT", {"default": 0, "min": 0}),
                
                # Video generation parameters
                "height": ("INT", {"default": 720}),
                "width": ("INT", {"default": 1280}),
                "num_frames": ("INT", {"default": 117}),
                "num_inference_steps": ("INT", {"default": 50}),
                "guidance_scale": ("FLOAT", {"default": 1.0}),
                "guidance_rescale": ("FLOAT", {"default": 0.0}),
                "embedded_cfg_scale": ("FLOAT", {"default": 6.0}),
                "flow_shift": ("INT", {"default": 7}),
                "output_type": (["pil"], {"default": "pil"}),
                
                # Model configuration
                "precision": (["fp32", "fp16", "bf16"], {"default": "bf16"}),
                
                # VAE configuration
                "vae_precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "vae_tiling": ("BOOLEAN", {"default": True}),
                "vae_sp": ("BOOLEAN", {"default": False}),
                
                # Text encoder configuration
                "text_encoder_precision": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "text_len": ("INT", {"default": 256}),
                "hidden_state_skip_layer": ("INT", {"default": 2}),
                
                # Secondary text encoder
                "text_encoder_precision_2": (["fp32", "fp16", "bf16"], {"default": "fp16"}),
                "text_len_2": ("INT", {"default": 77}),
                
                # Flow Matching parameters
                "flow_solver": (["euler", "rk4", "heun"], {"default": "euler"}),
                "denoise_type": (["flow", "diffusion"], {"default": "flow"}),
                
                # STA parameters
                "mask_strategy_file_path": ("STRING", {"default": ""}),
                "enable_torch_compile": ("BOOLEAN", {"default": False}),
                
                # Scheduler options
                "scheduler_type": (["euler", "dpm", "ddim"], {"default": "euler"}),
                
                # HunYuan specific parameters
                "neg_prompt": ("STRING", {"default": "", "multiline": True}),
                "num_videos": ("INT", {"default": 1}),
                "fps": ("INT", {"default": 24}),
                "use_cpu_offload": ("BOOLEAN", {"default": False}),
                "disable_autocast": ("BOOLEAN", {"default": False}),
                
                # Logging
                "log_level": (["info", "debug", "warning", "error"], {"default": "info"}),
                
                # Inference parameters
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "prompt_path": ("STRING", {"default": ""}),
                "output_path": ("STRING", {"default": "outputs/"}),
                "seed": ("INT", {"default": 1024}),
                "device_str": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("INFERENCE_ARGS",)
    RETURN_NAMES = ("inference_args",)
    FUNCTION = "create"
    CATEGORY = "custom/arguments"

    def create(self, **kwargs):
        tp_size = kwargs["tp_size"]  # default will be 2 from INPUT_TYPES
        sp_size = kwargs["sp_size"]  # default will be 1 from INPUT_TYPES
        world_size = tp_size * sp_size

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        for field in ["revision", "mask_strategy_file_path", "prompt_path"]:
            if kwargs[field] == "":
                kwargs[field] = None

        args = InferenceArgs(**kwargs)
        print("before initialize_distributed_and_parallelism")
        initialize_distributed_and_parallelism(args)
        print("after initialize_distributed_and_parallelism")
        return (args,)

class LoadModules:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inference_args": ("INFERENCE_ARGS",),
                "name": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODULES",)
    RETURN_NAMES = ("modules",)
    FUNCTION = "load"

    CATEGORY = "custom/Module"

    def load(self, inference_args, name):
        pipeline = HunyuanVideoPipeline(inference_args.model_path, inference_args)
    
        module = pipeline.get_module(name)
        return module


class VAELoaderFast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "component_model_path": ("STRING", {"default": ""}),
                "transformers_or_diffusers": (["transformers", "diffusers"],),
                "architecture": ("STRING", {"default": ""}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("PIPELINE_COMPONENT",)
    RETURN_NAMES = ("vae_module",)
    FUNCTION = "load"
    CATEGORY = "custom/pipeline"

    def load(self, component_model_path, transformers_or_diffusers, architecture, inference_args):
        return (
            PipelineComponentLoader.load_module(
                module_name="vae",
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args
            ),
        )

class TextEncoderLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "component_model_path": ("STRING", {"default": ""}),
                "transformers_or_diffusers": (["transformers", "diffusers"],),
                "architecture": ("STRING", {"default": ""}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("PIPELINE_COMPONENT",)
    RETURN_NAMES = ("text_encoder_module",)
    FUNCTION = "load"
    CATEGORY = "custom/pipeline"

    def load(self, component_model_path, transformers_or_diffusers, architecture, inference_args):
        return (
            PipelineComponentLoader.load_module(
                module_name="text_encoder",
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args
            ),
        )

class TransformerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "component_model_path": ("STRING", {"default": ""}),
                "transformers_or_diffusers": (["transformers", "diffusers"],),
                "architecture": ("STRING", {"default": ""}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("PIPELINE_COMPONENT",)
    RETURN_NAMES = ("transformer_module",)
    FUNCTION = "load"
    CATEGORY = "custom/pipeline"

    def load(self, component_model_path, transformers_or_diffusers, architecture, inference_args):
        return (
            PipelineComponentLoader.load_module(
                module_name="transformer",
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args
            ),
        )
class SchedulerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "component_model_path": ("STRING", {"default": ""}),
                "transformers_or_diffusers": (["transformers", "diffusers"],),
                "architecture": ("STRING", {"default": ""}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("PIPELINE_COMPONENT",)
    RETURN_NAMES = ("scheduler_module",)
    FUNCTION = "load"
    CATEGORY = "custom/pipeline"

    def load(self, component_model_path, transformers_or_diffusers, architecture, inference_args):
        return (
            PipelineComponentLoader.load_module(
                module_name="scheduler",
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args
            ),
        )

class TokenizerLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "component_model_path": ("STRING", {"default": ""}),
                "transformers_or_diffusers": (["transformers", "diffusers"],),
                "architecture": ("STRING", {"default": ""}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("PIPELINE_COMPONENT",)
    RETURN_NAMES = ("tokenizer_module",)
    FUNCTION = "load"
    CATEGORY = "custom/pipeline"

    def load(self, component_model_path, transformers_or_diffusers, architecture, inference_args):
        return (
            PipelineComponentLoader.load_module(
                module_name="tokenizer",
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                inference_args=inference_args
            ),
        )

class CreateForwardBatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_videos_per_prompt": ("INT", {"default": 1, "min": 1}),
                "n_tokens": ("INT", {"default": 77, "min": 1}),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "create"
    CATEGORY = "custom/functional"

    def create(
        self,
        prompt,
        negative_prompt,
        num_videos_per_prompt,
        n_tokens,
        inference_args
    ):

        device = torch.device(inference_args.device_str)
        batch = ForwardBatch(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            height=inference_args.height,
            width=inference_args.width,
            num_frames=inference_args.num_frames,
            num_inference_steps=inference_args.num_inference_steps,
            guidance_scale=inference_args.guidance_scale,
            eta=0.0,
            n_tokens=n_tokens,
            data_type="video" if inference_args.num_frames > 1 else "image",
            device=device,
            extra={},  # You can pass additional metadata here
        )

        return (batch,)

class InputValidationStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "vae": ("MODULES",),
                "transformer": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args):
        #initialization pipline
        vae_scale_factor = 2**(len(vae.block_out_channels) - 1)
        inference_args.vae_scale_factor = vae_scale_factor

        # self.image_processor = VaeImageProcessor(
        #     vae_scale_factor=vae_scale_factor)
        # self.add_module("image_processor", self.image_processor)

        num_channels_latents = transformer.in_channels
        inference_args.num_channels_latents = num_channels_latents

        stage = InputValidationStage()
        return (stage(forward_batch, inference_args),)

class LlamaEncodingStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("c",),
                "inference_args": ("INFERENCE_ARGS",),
                "text_encoder": ("MODULES",),
                "tokenizer": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, text_encoder, tokenizer):
        from fastvideo.v1.pipeline_stages import LlamaEncodingStage
        stage = LlamaEncodingStage(text_encoder=text_encoder,
                                   tokenizer=tokenizer,)
        return (stage(forward_batch, inference_args),)

class CLIPTextEncodingStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "text_encoder_2": ("MODULES",),
                "tokenizer_2": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, text_encoder_2, tokenizer_2):
        stage = CLIPTextEncodingStage(text_encoder=text_encoder_2,
                                      tokenizer=tokenizer_2,)
        return (stage(forward_batch, inference_args),)

class ConditioningStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args):
        from fastvideo.v1.pipeline_stages import ConditioningStage
        stage = ConditioningStage()
        return (stage(forward_batch, inference_args),)

class TimestepPreparationStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "scheduler": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, scheduler):
        stage = TimestepPreparationStage(scheduler = scheduler)
        return (stage(forward_batch, inference_args),)

class LatentPreparationStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "scheduler": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, scheduler):
        stage = LatentPreparationStage(scheduler = scheduler)
        return (stage(forward_batch, inference_args),)

class DenoisingStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "scheduler": ("MODULES",),
                "transformer": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, scheduler, transformer):
        stage = DenoisingStage(transformer=transformer, scheduler=scheduler)
        return (stage(forward_batch, inference_args),)

class DecodingStage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "inference_args": ("INFERENCE_ARGS",),
                "vae": ("MODULES",),
            }
        }

    RETURN_TYPES = ("FORWARD_BATCH",)
    RETURN_NAMES = ("forward_batch",)
    FUNCTION = "run"
    CATEGORY = "custom/pipeline/stages"

    def run(self, forward_batch, inference_args, vae):
        stage = DecodingStage(vae = vae)
        return (stage(forward_batch, inference_args),)

class VideoOutputsToFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "forward_batch": ("FORWARD_BATCH",),
                "output_path": ("STRING", {"default": "outputs/"}),
                "prompt": ("STRING", {"default": "A scenic mountain landscape"}),
                "fps": ("INT", {"default": 24}),
            }
        }

    RETURN_TYPES = ("IMAGE",)  # List of torch tensors (frames)
    RETURN_NAMES = ("frames",)
    FUNCTION = "convert_outputs"
    CATEGORY = "fastvideo"

    def convert_outputs(self, outputs, output_path, prompt, fps):
        from fastvideo.v1.pipelines.implementations.hunyuan.hunyuan_pipeline import DiffusionPipelineOutput
        outputs = DiffusionPipelineOutput(videos=batch.videos) 
        # Rearrange video tensor from B C T H W -> T B C H W
        video_frames = rearrange(outputs, "b c t h w -> t b c h w")

        # Convert each timestep to a grid, then to tensor
        frames = []
        for x in video_frames:
            grid = torchvision.utils.make_grid(x, nrow=6)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)  # CHW -> HWC
            frame_np = (grid * 255).numpy().astype(np.uint8)
            frame_tensor = torch.from_numpy(frame_np).float() / 255.0
            frames.append(frame_tensor)

        # Save video
        os.makedirs(output_path, exist_ok=True)
        video_file = os.path.join(output_path, f"{prompt[:100]}.mp4")
        imageio.mimsave(video_file, [(f.numpy() * 255).astype(np.uint8) for f in frames], fps=fps)

        return (frames,)

# Register the custom node
NODE_CLASS_MAPPINGS = {
    "HunyuanInferenceArgs": HunyuanInferenceArgs,
    "LoadModules": LoadModules,
    "VAELoaderFast": VAELoaderFast,
    "TextEncoderLoader": TextEncoderLoader,
    "TransformerLoader": TransformerLoader,
    "InputValidationStage": InputValidationStage,
    "LlamaEncodingStage": LlamaEncodingStage,
    "CLIPTextEncodingStage": CLIPTextEncodingStage,
    "ConditioningStage": ConditioningStage,
    "TimestepPreparationStage": TimestepPreparationStage,
    "LatentPreparationStage": LatentPreparationStage,
    "DenoisingStage": DenoisingStage,
    "DecodingStage": DecodingStage,
    "CreateForwardBatchNode": CreateForwardBatchNode,
    "VideoOutputsToFrames": VideoOutputsToFrames
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanInferenceArgs": "Hunyuan InferenceArgs",
    "LoadModules": "Load Modules",
    "VAELoaderFast": "VAE Loader",
    "TextEncoderLoader": "Text Encoder Loader",
    "TransformerLoader": "Transformer Loader",
    "InputValidationStage": "Input Validation Stage",
    "LlamaEncodingStage": "Llama Encoding Stage",
    "CLIPTextEncodingStage": "CLIPText Encoding Stage",
    "ConditioningStage": "Conditioning Stage",
    "TimestepPreparationStage": "Timestep Preparation Stage",
    "LatentPreparationStage": "LatentPreparationStage",
    "DenoisingStage": "Denoising Stage",
    "DecodingStage": "Decoding Stage",
    "CreateForwardBatchNode": "Create Forward Batch",
    "VideoOutputsToFrames": "Video Outputs To Frames"
}
