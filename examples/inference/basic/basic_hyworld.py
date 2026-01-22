# SPDX-License-Identifier: Apache-2.0
"""
Basic example for HYWorld (HY-WorldPlay) video generation using FastVideo.

This example replicates the same functionality as HY-WorldPlay/run.sh,
demonstrating image-to-video generation with camera trajectory control.
"""

import time
import math
import numpy as np
import imageio
import torchvision
from einops import rearrange

from fastvideo import VideoGenerator
from fastvideo.pipelines import ForwardBatch
from fastvideo.utils import shallow_asdict, align_to
from fastvideo.logger import init_logger

from fastvideo.models.dits.hyworld.pose import pose_to_input, compute_latent_num
from fastvideo.models.dits.hyworld.resolution_utils import get_resolution_from_image

logger = init_logger(__name__)

class HYWorldVideoGenerator(VideoGenerator):
    """Extended VideoGenerator that adds HYWorld-specific parameters to batch.extra."""

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

        # Add HYWorld-specific parameters to batch.extra
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

# Default prompt from HY-WorldPlay run.sh
DEFAULT_PROMPT = 'A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.'
DEFAULT_IMAGE = 'https://raw.githubusercontent.com/Tencent-Hunyuan/HY-WorldPlay/main/assets/img/test.png'

def main():
    import argparse

    # pose: (a, w, s, d) - (15, 31)
    # num_frames: (61, 125)
    parser = argparse.ArgumentParser(description="HYWorld video generation with FastVideo")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for video generation")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Path or URL to input image")
    parser.add_argument("--pose", type=str, default='a-15', help="Pose string (e.g., 'a-31', 'w-31', 's-31', 'd-31')")
    parser.add_argument("--output_path", type=str, default='video_samples_hyworld', help="Output video path")
    parser.add_argument("--num-frames", type=int, default=61, help="Number of frames")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--resolution", type=str, default="480p", help="Only support 480p for now")
    args = parser.parse_args()

    # Automatically determine resolution from input image
    HEIGHT, WIDTH = get_resolution_from_image(args.image, args.resolution)
    print(f"Image: {args.image}")
    print(f"Pose: {args.pose}")
    print(f"Resolution: {HEIGHT}x{WIDTH} (from {args.resolution} buckets)")
    print(f"Num frames: {args.num_frames}")
    print(f"Output path: {args.output_path}")

    # Initialize generator
    print("\nInitializing VideoGenerator for HYWorld...")
    
    generator = HYWorldVideoGenerator.from_pretrained(
        "FastVideo/HY-WorldPlay-Bidirectional-Diffusers",
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
        image_encoder_cpu_offload=True,
    )

    # Generate video
    print("\nGenerating video...")
    start_time = time.time()
    video = generator.generate_video(
        prompt=args.prompt,
        image_path=args.image,
        output_path=args.output_path,
        save_video=True,
        negative_prompt="",
        num_frames=args.num_frames,
        fps=24,
        height=HEIGHT,
        width=WIDTH,
        seed=args.seed,
        pose=args.pose,
    )
    elapsed = time.time() - start_time

    print(f"\nVideo generated successfully!")
    print(f"Saved to: {args.output_path}")
    print(f"Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
