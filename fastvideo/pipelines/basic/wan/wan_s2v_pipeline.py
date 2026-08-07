# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V speech-to-video pipeline.

Audio-driven generation: a reference image fixes who the subject is, the prompt
sets the scene, and the speech track drives the motion. The audio is encoded
once up front (it is the same for every denoising step) and cross-attended to
inside 12 of the transformer's 40 blocks.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.basic.wan.s2v_stages import S2VDecodingStage, S2VRefImageEncodingStage
from fastvideo.pipelines.stages import (AudioEncodingStage, ConditioningStage, DenoisingStage, InputValidationStage,
                                        LatentPreparationStage, TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class WanSpeechToVideoPipeline(LoRAPipeline, ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler", "audio_encoder", "audio_processor"
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

        # Audio is constant across denoising steps, so encode it once here
        # rather than inside the loop.
        self.add_stage(stage_name="audio_encoding_stage",
                       stage=AudioEncodingStage(
                           audio_encoder=self.get_module("audio_encoder"),
                           audio_processor=self.get_module("audio_processor"),
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer")))

        # The reference image becomes conditioning tokens appended to the video
        # sequence: one latent frame, encoded alone (not the I2V padded-video
        # format the shared stage produces).
        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=S2VRefImageEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(transformer=self.get_module("transformer"),
                                            scheduler=self.get_module("scheduler")))

        # Official recipe: decode with the reference latent prepended so the
        # causal VAE has temporal context, then trim the warm-up frames.
        self.add_stage(stage_name="decoding_stage", stage=S2VDecodingStage(vae=self.get_module("vae")))


EntryClass = WanSpeechToVideoPipeline
