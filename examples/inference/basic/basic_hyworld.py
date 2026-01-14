# SPDX-License-Identifier: Apache-2.0
"""
Basic example for HyWorld (HY-WorldPlay) video generation using FastVideo.

This example replicates the same functionality as HY-WorldPlay/run.sh,
demonstrating image-to-video generation with camera trajectory control.
"""

import os
import time
import math
import numpy as np
import torch
import imageio
import torchvision
from einops import rearrange

from fastvideo import VideoGenerator
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import shallow_asdict, align_to
from fastvideo.logger import init_logger

# Import HyWorld utilities from FastVideo
from fastvideo.models.dits.hyworld import (
    pose_to_input,
    compute_latent_num,
)

logger = init_logger(__name__)


class HyWorldVideoGenerator(VideoGenerator):
    """Extended VideoGenerator that adds HyWorld-specific parameters to batch.extra."""

    def _generate_single_video(self, prompt: str, sampling_param=None, **kwargs):
        """Override to add viewmats, Ks, and action to batch.extra."""
        fastvideo_args = self.fastvideo_args
        pipeline_config = fastvideo_args.pipeline_config

        if sampling_param is None:
            from fastvideo.configs.sample import SamplingParam
            sampling_param = SamplingParam.from_pretrained(fastvideo_args.model_path)

        # Update sampling param with kwargs
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(sampling_param, key):
                    setattr(sampling_param, key, value)

        # Get pose string from sampling_param or kwargs
        pose = kwargs.get('pose', getattr(sampling_param, 'POSE', 'w-31'))
        num_frames = kwargs.get('num_frames', getattr(sampling_param, 'num_frames', 125))

        # Calculate number of latents
        latent_num = compute_latent_num(num_frames)

        # Convert pose to viewmats, Ks, and action
        viewmats, Ks, action = pose_to_input(pose, latent_num)

        # Convert to tensors and add batch dimension
        viewmats = viewmats.unsqueeze(0)  # (1, T, 4, 4)
        Ks = Ks.unsqueeze(0)  # (1, T, 3, 3)
        action = action.unsqueeze(0)  # (1, T)

        # Validate inputs
        prompt = prompt.strip()
        sampling_param = sampling_param.__class__(**shallow_asdict(sampling_param))
        output_path = kwargs.get("output_path", sampling_param.output_path)
        sampling_param.prompt = prompt

        if sampling_param.negative_prompt is not None:
            sampling_param.negative_prompt = sampling_param.negative_prompt.strip()

        # Validate dimensions
        if (sampling_param.height <= 0 or sampling_param.width <= 0 or
            sampling_param.num_frames <= 0):
            raise ValueError(
                f"Height, width, and num_frames must be positive integers")

        temporal_scale_factor = pipeline_config.vae_config.arch_config.temporal_compression_ratio
        num_frames = sampling_param.num_frames
        num_gpus = fastvideo_args.num_gpus
        use_temporal_scaling_frames = pipeline_config.vae_config.use_temporal_scaling_frames

        # Adjust number of frames based on number of GPUs
        if use_temporal_scaling_frames:
            orig_latent_num_frames = (num_frames - 1) // temporal_scale_factor + 1
        else:
            orig_latent_num_frames = sampling_param.num_frames // 17 * 3

        if orig_latent_num_frames % fastvideo_args.num_gpus != 0:
            if use_temporal_scaling_frames:
                new_num_frames = (orig_latent_num_frames - 1) * temporal_scale_factor + 1
            else:
                divisor = math.lcm(3, num_gpus)
                orig_latent_num_frames = (
                    (orig_latent_num_frames + divisor - 1) // divisor) * divisor
                new_num_frames = orig_latent_num_frames // 3 * 17

            logger.info(
                "Adjusting number of frames from %s to %s based on number of GPUs (%s)",
                sampling_param.num_frames, new_num_frames, fastvideo_args.num_gpus)
            sampling_param.num_frames = new_num_frames

        # Calculate sizes
        target_height = align_to(sampling_param.height, 16)
        target_width = align_to(sampling_param.width, 16)

        # Calculate latent sizes
        latents_size = [(sampling_param.num_frames - 1) // 4 + 1,
                        sampling_param.height // 8, sampling_param.width // 8]
        n_tokens = latents_size[0] * latents_size[1] * latents_size[2]

        # Prepare batch
        batch = ForwardBatch(
            **shallow_asdict(sampling_param),
            eta=0.0,
            n_tokens=n_tokens,
            VSA_sparsity=fastvideo_args.VSA_sparsity,
        )

        # Add HyWorld-specific parameters to batch.extra
        batch.extra['viewmats'] = viewmats
        batch.extra['Ks'] = Ks
        batch.extra['action'] = action
        batch.extra['chunk_latent_frames'] = 16  # For bidirectional model

        # Run inference
        start_time = time.perf_counter()
        output_batch = self.executor.execute_forward(batch, fastvideo_args)
        samples = output_batch.output
        logging_info = output_batch.logging_info

        gen_time = time.perf_counter() - start_time
        logger.info("Generated successfully in %.2f seconds", gen_time)

        # Process outputs
        videos = rearrange(samples, "b c t h w -> t b c h w")
        frames = []
        for x in videos:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save video if requested
        if batch.save_video:
            imageio.mimsave(output_path, frames, fps=batch.fps, format="mp4")
            logger.info("Saved video to %s", output_path)

        if batch.return_frames:
            return frames
        else:
            return {
                "samples": samples,
                "frames": frames,
                "prompts": prompt,
                "size": (target_height, target_width, batch.num_frames),
                "generation_time": gen_time,
                "logging_info": logging_info,
                "trajectory": output_batch.trajectory_latents,
                "trajectory_timesteps": output_batch.trajectory_timesteps,
                "trajectory_decoded": output_batch.trajectory_decoded,
            }


OUTPUT_PATH = "video_samples_hyworld"


def main():
    # Parameters matching run.sh from HY-WorldPlay
    PROMPT = (
        'A paved pathway leads towards a stone arch bridge spanning a calm body of water.  '
        'Lush green trees and foliage line the path and the far bank of the water. '
        'A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. '
        'The water reflects the surrounding greenery and the sky.  '
        'The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. '
        'The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  '
        'The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
    )

    # Update these paths to match your setup (from run.sh)
    IMAGE_PATH = os.getenv("IMAGE_PATH", "assets/hyworld.png")
    # MODEL_PATH = os.getenv(
    #     "MODEL_PATH",
    #     "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038"
    # )
    # # For bidirectional model (as used in run.sh)
    # ACTION_MODEL_PATH = os.getenv(
    #     "BI_ACTION_MODEL_PATH",
    #     "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/969249711ab41203e8d8d9f784a82372e9070ac5/bidirectional_model/diffusion_pytorch_model.safetensors"
    # )

    SEED = 1
    NUM_FRAMES = 125
    HEIGHT = 480
    WIDTH = 832  # 16:9 aspect ratio for 480p
    POSE = 'a-31'  # Forward movement for 31 latents

    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Warning: Image path {IMAGE_PATH} does not exist. Image-to-video requires an image.")
        print("Please set IMAGE_PATH environment variable or update the IMAGE_PATH in the script.")
        return

    # FastVideo will automatically use the optimal default arguments for the model
    generator = HyWorldVideoGenerator.from_pretrained(
        "/mnt/weka/home/hao.zhang/mhuo/data/hyworld",  # Base HunyuanVideo-1.5 model path
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        # Load action model weights if available
        # init_weights_from_safetensors=ACTION_MODEL_PATH if os.path.exists(ACTION_MODEL_PATH) else None,
    )

    # Generate video with the same parameters as run.sh
    video = generator.generate_video(
        prompt=PROMPT,
        image_path=IMAGE_PATH,
        output_path=OUTPUT_PATH,
        save_video=True,
        negative_prompt="",
        num_frames=NUM_FRAMES,
        fps=24,
        height=HEIGHT,
        width=WIDTH,
        seed=SEED,
        pose=POSE,  # HyWorld-specific: camera trajectory
    )

    print(f"Video generated successfully and saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
