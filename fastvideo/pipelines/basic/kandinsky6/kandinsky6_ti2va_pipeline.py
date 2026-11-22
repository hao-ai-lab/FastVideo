# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages.input_validation import InputValidationStage
from fastvideo.pipelines.stages.kandinsky6 import (
    Kandinsky6AudioDecodingStage,
    Kandinsky6CFGResolutionStage,
    Kandinsky6DecodingStage,
    Kandinsky6DenoisingStage,
    Kandinsky6ImageEncodingStage,
    Kandinsky6LatentPreparationStage,
    Kandinsky6TextEncodingStage,
)
from fastvideo.pipelines.stages.timestep_preparation import TimestepPreparationStage
from fastvideo.utils import verify_model_config_and_directory

NVFP4_TRANSFORMER_DIR = "nvfp4/transformer"


class Kandinsky6TI2VAPipeline(ComposedPipelineBase):
    """Kandinsky6 TI2VA pipeline: text, optionally plus a conditioning image,
    to synchronized video+audio, for the base (flow-matching) and the pi-Flow
    distilled checkpoint alike (the bundle's scheduler component decides the
    sampler). There is deliberately no separate T2VA/I2VA pipeline class --
    one stage chain handles both, like the official Diffusers
    Kandinsky6TI2VAPipeline: Kandinsky6ImageEncodingStage is a no-op when the
    request carries no conditioning image and injects it as an extra
    reference frame when one is supplied.
    """

    # model_index.json declares "audio_vae" (mel-VAE decoder) and "vocoder" (BigVGAN-v2,
    # mel->waveform) as two separate components; Kandinsky6AudioDecodingStage calls
    # audio_vae.decode() then vocoder(mel), like LTX-2's audio_vae/vocoder pair.
    _required_config_modules = [
        "scheduler",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    @classmethod
    def nvfp4_component_dirs(cls) -> tuple[str, ...]:
        """Components from a model snapshot that shares assets across precisions."""
        return tuple(NVFP4_TRANSFORMER_DIR if name == "transformer" else name
                     for name in cls._required_config_modules)

    def _load_config(self, model_path: str) -> dict[str, Any]:
        packed_transformer = os.path.join(model_path, NVFP4_TRANSFORMER_DIR)
        quant_config = self.fastvideo_args.pipeline_config.dit_config.quant_config
        if (getattr(quant_config, "layer_profile", None) == "kandinsky6"
                and os.path.isfile(os.path.join(packed_transformer, H3_NVFP4_DIT_EXPORT_FILENAME))):
            config = verify_model_config_and_directory(
                model_path,
                required_component_dirs=self.nvfp4_component_dirs(),
            )
            # An absolute path preserves dispatch to the standard TransformerLoader.
            self._extra_config_module_map = {
                **self._extra_config_module_map,
                "transformer": os.path.abspath(packed_transformer),
            }
        else:
            config = super()._load_config(model_path)
        return config

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        # Recomputes do_classifier_free_guidance under Kandinsky6's own CFG rule (abs(g-1)>1e-6,
        # forced off for PiFlow) before text encoding decides whether to encode a negative prompt.
        self.add_stage(
            stage_name="cfg_resolution_stage",
            stage=Kandinsky6CFGResolutionStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="text_encoding_stage",
            stage=Kandinsky6TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=Kandinsky6LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
        )

        # No-op unless the request supplied a conditioning image; see
        # Kandinsky6ImageEncodingStage's docstring.
        self.add_stage(
            stage_name="image_encoding_stage",
            stage=Kandinsky6ImageEncodingStage(vae=self.get_module("vae")),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=Kandinsky6DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=Kandinsky6AudioDecodingStage(
                audio_vae=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=Kandinsky6DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = Kandinsky6TI2VAPipeline
