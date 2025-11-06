# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.vision_utils import load_image, load_video, pil_to_numpy, numpy_to_pt, normalize, resize
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage
from fastvideo.pipelines.stages.validators import (StageValidators,
                                                   VerificationResult)
from fastvideo.utils import best_output_size

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.
    
    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.
    """

    def _generate_seeds(self, batch: ForwardBatch,
                        fastvideo_args: FastVideoArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_videos_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds
        # Peiyuan: using GPU seed will cause A100 and H100 to generate different results...
        batch.generator = [
            torch.Generator("cpu").manual_seed(seed) for seed in seeds
        ]

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """
        Validate and prepare inputs.
        
        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.
            
        Returns:
            The validated batch information.
        """

        self._generate_seeds(batch, fastvideo_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError(
                "Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (batch.do_classifier_free_guidance and batch.negative_prompt is None
                and batch.negative_prompt_embeds is None):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided")

        # Validate height and width
        if batch.height is None or batch.width is None:
            raise ValueError(
                "Height and width must be provided. Please set `height` and `width`."
            )
        if batch.height % 8 != 0 or batch.width % 8 != 0:
            raise ValueError(
                f"Height and width must be divisible by 8 but are {batch.height} and {batch.width}."
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale <= 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # for i2v, get image from image_path
        # @TODO(Wei) hard-coded for wan2.2 5b ti2v for now. Should put this in image_encoding stage
        if batch.image_path is not None:
            if batch.image_path.endswith(".mp4"):
                image = load_video(batch.image_path)[0]
            else:
                image = load_image(batch.image_path)
            batch.pil_image = image

        # further processing for ti2v task
        if fastvideo_args.pipeline_config.ti2v_task and batch.pil_image is not None:
            img = batch.pil_image
            ih, iw = img.height, img.width
            patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size
            vae_stride = fastvideo_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            dh, dw = patch_size[1] * vae_stride, patch_size[2] * vae_stride
            max_area = 704 * 1280
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)

            scale = max(ow / iw, oh / ih)
            img = img.resize((round(iw * scale), round(ih * scale)),
                             Image.LANCZOS)
            logger.info("resized img height: %s, img width: %s", img.height,
                        img.width)

            # center-crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))
            assert img.width == ow and img.height == oh

            # to tensor
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(
                self.device).unsqueeze(1)
            img = img.unsqueeze(0)
            batch.height = oh
            batch.width = ow
            batch.pil_image = img

        # for v2v, get control video from video path
        if batch.video_path is not None:
            pil_images, original_fps = load_video(batch.video_path,
                                                  return_fps=True)
            logger.info("Loaded video with %s frames, original FPS: %s",
                        len(pil_images), original_fps)

            # Get target parameters from batch
            target_fps = batch.fps
            target_num_frames = batch.num_frames
            target_height = batch.height
            target_width = batch.width

            if target_fps is not None and original_fps is not None:
                frame_skip = max(1, int(original_fps // target_fps))
                if frame_skip > 1:
                    pil_images = pil_images[::frame_skip]
                    effective_fps = original_fps / frame_skip
                    logger.info(
                        "Resampled video from %.1f fps to %.1f fps (skip=%s)",
                        original_fps, effective_fps, frame_skip)

            # Limit to target number of frames
            if target_num_frames is not None and len(
                    pil_images) > target_num_frames:
                pil_images = pil_images[:target_num_frames]
                logger.info("Limited video to %s frames (from %s total)",
                            target_num_frames, len(pil_images))

            # Resize each PIL image to target dimensions
            resized_images = []
            for pil_img in pil_images:
                resized_img = resize(pil_img,
                                     target_height,
                                     target_width,
                                     resize_mode="default",
                                     resample="lanczos")
                resized_images.append(resized_img)

            # Convert PIL images to numpy array
            video_numpy = pil_to_numpy(resized_images)
            video_numpy = normalize(video_numpy)
            video_tensor = numpy_to_pt(video_numpy)

            # Rearrange to [C, T, H, W] and add batch dimension -> [B, C, T, H, W]
            input_video = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)

            batch.video_latent = input_video

        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.positive_int])
        result.add_check("num_videos_per_prompt", batch.num_videos_per_prompt,
                         V.positive_int)
        result.add_check(
            "prompt_or_embeds", None, lambda _: V.string_or_list_strings(
                batch.prompt) or V.list_not_empty(batch.prompt_embeds))
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("num_inference_steps", batch.num_inference_steps,
                         V.positive_int)
        result.add_check(
            "guidance_scale", batch.guidance_scale, lambda x: not batch.
            do_classifier_free_guidance or V.positive_float(x))
        return result

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator,
                         V.generator_or_list_generators)
        return result
