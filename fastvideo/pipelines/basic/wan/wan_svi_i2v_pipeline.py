# SPDX-License-Identifier: Apache-2.0
"""
Wan I2V pipeline variant for Stable-Video-Infinity multi-clip inference.

This module contains an implementation of the SVI-flavored Wan I2V
pipeline using the modular pipeline architecture.
"""

import dataclasses

import PIL.Image
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from fastvideo.pipelines.basic.wan.wan_i2v_pipeline import WanImageToVideoPipeline
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, DenoisingStage, ImageEncodingStage,
                                        InputValidationStage, LatentPreparationStage, SVIImageVAEEncodingStage,
                                        TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


def _tensor_to_pil_list(frames: torch.Tensor) -> list[PIL.Image.Image]:
    """Convert a video tensor to a list of PIL frames."""
    arr = (frames.detach().to(torch.float32).cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    return [PIL.Image.fromarray(a) for a in arr]


class WanSVIImageToVideoPipeline(WanImageToVideoPipeline):
    """
    Pipeline for Stable-Video-Infinity multi-clip I2V generation on Wan 2.1.
    """

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=5.0)

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        if (self.get_module("image_encoder") is not None and self.get_module("image_processor") is not None):
            self.add_stage(
                stage_name="image_encoding_stage",
                stage=ImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ),
            )
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )
        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=SVIImageVAEEncodingStage(vae=self.get_module("vae")),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if not self.post_init_called:
            self.post_init()

        n_steps = int(batch.num_inference_steps)
        batch.sigmas = torch.linspace(1.0, 0.0, n_steps + 1)[:-1].tolist()

        num_clips = max(1, int(batch.svi_num_clips or 1))
        num_motion = max(1, int(batch.svi_num_motion_frames or 1))

        if isinstance(batch.prompt, list):
            prompts: list[str | None] = list(batch.prompt)
        else:
            prompts = [batch.prompt]
        if len(prompts) < num_clips:
            prompts = prompts + [prompts[-1]] * (num_clips - len(prompts))
        elif len(prompts) > num_clips:
            prompts = prompts[:num_clips]

        if num_clips == 1:
            batch.prompt = prompts[0]
            for stage in self.stages:
                batch = stage(batch, fastvideo_args)
            return batch

        # Multi-clip needs the reference image up front to construct motion frames
        # for clip 0. Run InputValidationStage now to resolve image_path -> pil_image.
        self.input_validation_stage(batch, fastvideo_args)
        assert isinstance(batch.pil_image, PIL.Image.Image)
        random_ref = batch.svi_random_ref_frame or batch.pil_image
        motion_frames: list[PIL.Image.Image] = (batch.svi_first_frames or [batch.pil_image])

        clip_outputs: list[torch.Tensor] = []
        for clip_idx in range(num_clips):
            clip_batch = dataclasses.replace(
                batch,
                prompt=prompts[clip_idx],
                pil_image=motion_frames[0],
                # Clear image_path so per-clip InputValidationStage does not
                # reload the original ref and clobber motion_frames[0].
                image_path=None,
                svi_first_frames=motion_frames,
                svi_random_ref_frame=random_ref,
                prompt_embeds=[],
                negative_prompt_embeds=None,
                prompt_attention_mask=None,
                negative_attention_mask=None,
                clip_embedding_pos=None,
                clip_embedding_neg=None,
                image_embeds=[],
                preprocessed_image=None,
                latents=None,
                image_latent=None,
                noise_pred=None,
                output=None,
                timesteps=None,
                timestep=None,
                step_index=None,
                is_prompt_processed=False,
            )
            for stage in self.stages:
                clip_batch = stage(clip_batch, fastvideo_args)

            assert clip_batch.output is not None
            clip_outputs.append(clip_batch.output)
            logger.info("SVI clip %d/%d generated, frames shape=%s", clip_idx + 1, num_clips, clip_batch.output.shape)

            if clip_idx + 1 < num_clips:
                tail = clip_batch.output[0, :, -num_motion:, :, :]
                motion_frames = _tensor_to_pil_list(tail)

        # Drop the first num_motion frames of each follow-up clip to avoid duplicating
        # the previous clip's tail in the stitched output.
        concatenated = [clip_outputs[0]]
        for video in clip_outputs[1:]:
            concatenated.append(video[:, :, num_motion:, :, :])
        batch.output = torch.cat(concatenated, dim=2)
        return batch


EntryClass = WanSVIImageToVideoPipeline
