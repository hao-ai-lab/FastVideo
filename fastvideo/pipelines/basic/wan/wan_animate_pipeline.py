# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-Animate character animation / replacement pipeline.

A reference image fixes who the character is; a preprocessed skeleton video
drives the body; a preprocessed face-crop video drives the expression. In
replace mode a background video and character mask additionally pin the scene,
and the character is composited into it.

Inputs are the *preprocessed* artifacts the official ``wan/modules/animate/
preprocess`` pipeline produces (src_pose.mp4, src_face.mp4, src_ref.png, and
for replace mode src_bg.mp4 + src_mask.mp4) -- same contract as diffusers'
``WanAnimatePipeline``, which also deliberately does not ship preprocessing.

CFG is off by default (official guide_scale 1.0; the text prompt is non-core).
When enabled, the negative pass keeps pose and conditioning but blanks the
face crops to -1 (black), matching diffusers.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.basic.wan.animate_stages import (AnimateConditioningLatentsStage, AnimateDecodingStage,
                                                          AnimateFaceVideoStage, AnimateLatentPreparationStage,
                                                          AnimatePoseVideoEncodingStage)
from fastvideo.pipelines.stages import (ConditioningStage, DenoisingStage, ImageEncodingStage, InputValidationStage,
                                        TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class WanAnimatePipeline(LoRAPipeline, ComposedPipelineBase):
    """v1 scope: a single 77-frame segment; multi-segment chaining is a follow-up.

    The relighting LoRA (replace mode) loads through the LoRAPipeline mixin's
    standard ``lora_path`` mechanism; it is not fetched automatically.
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "image_encoder", "image_processor"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        # CLIP features of the reference image; joins the text cross-attention
        # as 257 leading tokens (Wan-I2V convention).
        self.add_stage(stage_name="image_encoding_stage",
                       stage=ImageEncodingStage(
                           image_encoder=self.get_module("image_encoder"),
                           image_processor=self.get_module("image_processor"),
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        # +1 latent frame: the reference slot, denoised alongside the video.
        self.add_stage(stage_name="latent_preparation_stage",
                       stage=AnimateLatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                           transformer=self.get_module("transformer")))

        # The 20-channel y = [mask | cond latent] x [ref | target frames].
        self.add_stage(stage_name="conditioning_latents_stage",
                       stage=AnimateConditioningLatentsStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="pose_encoding_stage",
                       stage=AnimatePoseVideoEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="face_video_stage", stage=AnimateFaceVideoStage())

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(transformer=self.get_module("transformer"),
                                            scheduler=self.get_module("scheduler")))

        # Drop the reference slot, then decode.
        self.add_stage(stage_name="decoding_stage", stage=AnimateDecodingStage(vae=self.get_module("vae")))


EntryClass = WanAnimatePipeline
