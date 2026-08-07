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
    arr = (frames.detach().to(torch.float32).cpu().clamp(0, 1) * 255.0).to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    return [PIL.Image.fromarray(a) for a in arr]


def _validate_multiclip_frames(num_motion: int, num_frames: int) -> None:
    """num_motion must be < num_frames, else follow-up clips stitch to empty."""
    if num_motion >= num_frames:
        raise ValueError(f"svi_num_motion_frames ({num_motion}) must be smaller than num_frames ({num_frames}) "
                         "for multi-clip generation; otherwise stitched follow-up clips would be empty.")


def _stitch_clip_outputs(clip_outputs: list[torch.Tensor], num_motion: int) -> torch.Tensor:
    """Stitch clips using the SVI overlap convention."""
    clips = [video[:, :, :-num_motion] for video in clip_outputs[:-1]]
    return torch.cat([*clips, clip_outputs[-1]], dim=2)


def _resolve_clip_prompts(
    prompt: str | list[str] | None,
    clip_prompts: list[str] | None,
    num_clips: int,
) -> list[str]:
    """Return exactly one non-empty prompt per generated clip."""
    if clip_prompts is None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("SVI requires a non-empty primary prompt")
        return [prompt] * num_clips

    if len(clip_prompts) != num_clips:
        raise ValueError(f"svi_clip_prompts must contain exactly svi_num_clips entries "
                         f"({num_clips}), but got {len(clip_prompts)}")
    if any(not isinstance(item, str) or not item.strip() for item in clip_prompts):
        raise ValueError("svi_clip_prompts entries must be non-empty strings")
    return list(clip_prompts)


def _clip_seed(seed: int | None, clip_idx: int, seed_stride: int) -> int:
    """Match the SVI reference by varying diffusion noise per clip."""
    if seed is None:
        raise ValueError("SVI requires a seed")
    return int(seed) + clip_idx * seed_stride


class WanSVIImageToVideoPipeline(WanImageToVideoPipeline):
    """
    Pipeline for Stable-Video-Infinity multi-clip I2V generation on Wan 2.1.
    """

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=fastvideo_args.pipeline_config.flow_shift,
        )

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
        if batch.sigmas is None:
            batch.sigmas = torch.linspace(1.0, 0.0, n_steps + 1)[:-1].tolist()

        num_clips = max(1, int(batch.svi_num_clips or 1))
        num_motion = max(1, int(batch.svi_num_motion_frames or 1))
        seed_stride = int(batch.svi_seed_stride)

        prompts = _resolve_clip_prompts(batch.prompt, batch.svi_clip_prompts, num_clips)
        num_frames = int(batch.num_frames) if batch.num_frames is not None else 0
        if num_clips > 1:
            _validate_multiclip_frames(num_motion, num_frames)

        # Resolve image_path before constructing the first motion window.
        self.input_validation_stage(batch, fastvideo_args)
        assert isinstance(batch.pil_image, PIL.Image.Image)
        assert isinstance(batch.height, int) and isinstance(batch.width, int)
        reference_frame = batch.pil_image.resize((batch.width, batch.height))
        random_ref = batch.svi_random_ref_frame or reference_frame
        motion_frames = batch.svi_first_frames or [reference_frame]

        clip_outputs: list[torch.Tensor] = []
        for clip_idx in range(num_clips):
            clip_batch = dataclasses.replace(
                batch,
                prompt=prompts[clip_idx],
                seed=_clip_seed(batch.seed, clip_idx, seed_stride),
                pil_image=motion_frames[0],
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
            logger.info(
                "SVI clip %d/%d generated with seed=%d, frames shape=%s",
                clip_idx + 1,
                num_clips,
                clip_batch.seed,
                clip_batch.output.shape,
            )

            if clip_idx + 1 < num_clips:
                tail = clip_batch.output[0, :, -num_motion:, :, :]
                motion_frames = _tensor_to_pil_list(tail)

        batch.output = _stitch_clip_outputs(clip_outputs, num_motion)
        return batch


EntryClass = WanSVIImageToVideoPipeline
