# SPDX-License-Identifier: Apache-2.0
"""
Lucy-Edit video editing pipeline implementation.

This module implements the Lucy-Edit-Dev video editing pipeline using
the modular pipeline architecture. Lucy-Edit performs instruction-guided
video editing by encoding the input video through the VAE and concatenating
the video latents with noise latents along the channel dimension before
denoising with the Wan2.2-based transformer.

Reference: https://huggingface.co/decart-ai/Lucy-Edit-Dev
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines import ComposedPipelineBase, LoRAPipeline
from fastvideo.pipelines.stages import (
    ConditioningStage,
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
    VideoVAEEncodingStage,
)

logger = init_logger(__name__)


class LucyEditPipeline(LoRAPipeline, ComposedPipelineBase):
    """Lucy-Edit video editing pipeline.

    This pipeline takes an input video and a text editing instruction,
    encodes the video into latent space, concatenates the video latents
    with noise latents, and denoises to produce an edited video.

    The pipeline uses:
    - WanTransformer3DModel (in_channels=96 = 48 noise + 48 video latent)
    - AutoencoderKLWan (z_dim=48, Wan2.2 enhanced VAE)
    - UMT5EncoderModel (T5 text encoder)
    - UniPCMultistepScheduler
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Set up pipeline stages with proper dependency injection.

        Lucy-Edit pipeline stages:
        1. Input validation
        2. Text encoding (T5)
        3. Conditioning (CFG setup)
        4. Timestep preparation
        5. Latent preparation (noise generation)
        6. Video VAE encoding (encode input video to latents)
        7. Denoising (transformer with concatenated [noise, video] latents)
        8. Decoding (VAE decode to pixel space)
        """

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="conditioning_stage",
            stage=ConditioningStage(),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(
                scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # Encode input video through VAE - stores encoded latents
        # in batch.video_latent for channel-wise concatenation
        self.add_stage(
            stage_name="video_encoding_stage",
            stage=VideoVAEEncodingStage(vae=self.get_module("vae")),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )


EntryClass = LucyEditPipeline
