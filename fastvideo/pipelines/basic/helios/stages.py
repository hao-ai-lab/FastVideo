# SPDX-License-Identifier: Apache-2.0
"""Model-specific stages for Helios-Distilled T2V inference."""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.input_validation import InputValidationStage


class HeliosInputValidationStage(InputValidationStage):
    """Validate the intentionally narrow first Helios contribution: T2V."""

    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        batch = super().forward(batch, fastvideo_args)
        if not isinstance(batch.height, int) or not isinstance(batch.width, int):
            raise ValueError("Helios T2V expects scalar height and width")
        if batch.height % 64 or batch.width % 64:
            raise ValueError("Helios height and width must be divisible by 64 for two pyramid downsamples")
        if not isinstance(batch.num_frames, int) or batch.num_frames <= 0:
            raise ValueError(f"Helios num_frames must be a positive integer, got {batch.num_frames}")
        if batch.num_videos_per_prompt != 1:
            raise ValueError("Helios currently supports num_videos_per_prompt=1")
        if isinstance(batch.prompt, list) and len(batch.prompt) != 1:
            raise ValueError("Helios currently accepts one prompt per FastVideo request")
        if batch.image_path is not None or batch.video_path is not None or batch.pil_image is not None:
            raise ValueError("This initial Helios contribution supports T2V only")
        if batch.latents is not None:
            raise ValueError("Pre-generated Helios chunk latents are not supported yet")

        steps = batch.pyramid_num_inference_steps_list
        if steps is None:
            steps = [batch.num_inference_steps] * 3
        if len(steps) != 3 or any(not isinstance(value, int) or value <= 0 for value in steps):
            raise ValueError(f"Helios requires three positive pyramid step counts, got {steps}")
        batch.pyramid_num_inference_steps_list = list(steps)

        history_sizes = batch.history_sizes or [16, 2, 1]
        if len(history_sizes) != 3 or any(not isinstance(value, int) or value <= 0 for value in history_sizes):
            raise ValueError(f"Helios requires three positive history sizes, got {history_sizes}")
        batch.history_sizes = sorted(history_sizes, reverse=True)
        if batch.num_latent_frames_per_chunk != 9:
            raise ValueError("Helios-Distilled requires num_latent_frames_per_chunk=9")
        if not batch.keep_first_frame:
            raise ValueError("The initial Helios T2V port requires keep_first_frame=True")
        if batch.is_skip_first_chunk:
            raise ValueError("is_skip_first_chunk is only meaningful for conditioned Helios modes")
        if batch.zero_steps < 0:
            raise ValueError("Helios zero_steps must be non-negative")
        return batch
