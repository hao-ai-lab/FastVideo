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
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.models.hunyuan.diffusion.schedulers import FlowMatchDiscreteScheduler
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.vae import load_vae
from fastvideo.models.hunyuan.modules.modulate_layers import modulate
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state, nccl_info
from fastvideo.sample.sample_t2v_hunyuan_STA import teacache_forward, initialize_distributed

MAX_RESOLUTION = 16384

class FastVideoSampler:
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "model_path": ("STRING", {"default": "/workspace/FastVideo/data/hunyuan", "tooltip": "Path to the Hunyuan model."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text prompt for video generation."}),
                "negative_prompt": ("STRING", {"default": "", "tooltip": "Negative prompt for video generation."}),
                "width": ("INT", {"default": 600, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Width of the generated video."}),
                "height": ("INT", {"default": 500, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Height of the generated video."}),
                "num_frames": ("INT", {"default": 57, "min": 1, "max": 1000, "tooltip": "Number of frames in the generated video."}),
                "num_inference_steps": ("INT", {"default": 5, "min": 1, "max": 1000, "tooltip": "Number of inference steps."}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "Guidance scale for the generation."}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for reproducibility."}),
                "flow_shift": ("INT", {"default": 7, "min": 0, "max": 100, "tooltip": "Flow shift parameter for video generation."}),
                "enable_teacache": ("BOOLEAN", {"default": True, "tooltip": "Enable Teacache for faster inference."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Relative L1 threshold for Teacache."}),
            }
        }

    RETURN_TYPES = ("IMAGE",) 
    OUTPUT_TOOLTIPS = ("The generated video frames as an image tensor.",)
    FUNCTION = "generate_video"

    CATEGORY = "video"
    DESCRIPTION = "Generates a video from a text prompt using the Hunyuan model."

    def generate_video(self, model_path, prompt, negative_prompt, width, height, num_frames, num_inference_steps, guidance_scale, seed, flow_shift, enable_teacache, rel_l1_thresh):
        #create args manuallyf
        args = SimpleNamespace(
            prompt=prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            model_path=model_path,
            output_path="outputs_video/hunyuan_STA/",
            fps=24,
            sliding_block_size="8,6,10",
            denoise_type="flow",
            seed=seed,
            neg_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            embedded_cfg_scale=6.0,
            flow_shift=flow_shift,
            batch_size=1,
            num_videos=1,
            load_key="module",
            use_cpu_offload=False,
            dit_weight="/workspace/FastVideo/data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            reproduce=False,
            disable_autocast=False,
            flow_reverse=True,
            flow_solver="euler",
            use_linear_quadratic_schedule=False,
            linear_schedule_end=25,
            model="HYVideo-T/2-cfgdistill",
            latent_channels=16,
            precision="bf16",
            rope_theta=256,
            vae="884-16c-hy",
            vae_precision="fp16",
            vae_tiling=True,
            vae_sp=False,
            text_encoder="llm",
            text_encoder_precision="fp16",
            text_states_dim=4096,
            text_len=256,
            tokenizer="llm",
            prompt_template="dit-llm-encode",
            prompt_template_video="dit-llm-encode-video",
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            text_encoder_2="clipL",
            text_encoder_precision_2="fp16",
            text_states_dim_2=768,
            tokenizer_2="clipL",
            text_len_2=77,
            skip_time_steps=10,
            mask_strategy_selected=[1, 2, 6],
            rel_l1_thresh=rel_l1_thresh,
            enable_teacache=enable_teacache,
            enable_torch_compile=False,
            mask_strategy_file_path="/workspace/FastVideo/assets/mask_strategy_hunyuanon"
        )
        #initialize_distributed()
        print(nccl_info.sp_size)

        models_root_path = Path(model_path)
        # Initialize the Hunyuan video sampler
        sampler = HunyuanVideoSampler.from_pretrained(model_path, args=args)

        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        with open(args.mask_strategy_file_path, 'r') as f:
            mask_strategy = json.load(f)

        # Set up the pipeline
        sampler.pipeline.transformer.__class__.enable_teacache = enable_teacache
        sampler.pipeline.transformer.__class__.rel_l1_thresh = rel_l1_thresh

        sampler.pipeline.transformer.__class__.cnt = 0
        sampler.pipeline.transformer.__class__.num_steps = num_inference_steps
        sampler.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
        sampler.pipeline.transformer.__class__.previous_modulated_input = None
        sampler.pipeline.transformer.__class__.previous_residual = None
        sampler.pipeline.transformer.__class__.forward = teacache_forward

        # Generate the video

        outputs = sampler.predict(
            prompt=prompt,
            height=args.height,
            width=args.width,
            video_length=args.num_frames,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            mask_strategy=None,
        )

        # Convert the video frames to an image tensor
        video_frames = rearrange(outputs["samples"], "b c t h w -> t b c h w") 
        frames = []
        for x in video_frames:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Convert frames to a format compatible with ComfyUI (e.g., list of tensors)
        frame_tensors = [torch.from_numpy(frame).float() / 255.0 for frame in frames]
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        imageio.mimsave(os.path.join(args.output_path, f"{prompt[:100]}.mp4"), frame_tensors, fps=args.fps)

        # Return the frames
        return (frame_tensors,) 

# Register the custom node
NODE_CLASS_MAPPINGS = {
    "FastVideoSampler": FastVideoSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FastVideoSampler": "Fast Video Sampler",
}
