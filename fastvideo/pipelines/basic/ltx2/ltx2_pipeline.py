# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 text-to-video pipeline.
"""

import os
from typing import Any

from transformers import AutoTokenizer

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (DecodingStage, InputValidationStage,
                                        LTX2AudioDecodingStage,
                                        LTX2DenoisingStage,
                                        LTX2LatentPreparationStage,
                                        TextEncodingStage)

logger = init_logger(__name__)


class LTX2Pipeline(ComposedPipelineBase):

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
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
            stage_name="latent_preparation_stage",
            stage=LTX2LatentPreparationStage(
                transformer=self.get_module("transformer"), ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2DenoisingStage(
                transformer=self.get_module("transformer"), ),
        )

        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=LTX2AudioDecodingStage(
                audio_decoder=self.get_module("audio_vae"),
                vocoder=self.get_module("vocoder"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae")),
        )

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        tokenizer = self.get_module("tokenizer")
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_index = self._load_config(self.model_path)
        logger.info("Loading pipeline modules from config: %s", model_index)

        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        model_index.pop("workload_type", None)

        if len(model_index) <= 1:
            raise ValueError(
                "model_index.json must contain at least one pipeline module")

        required_modules = self.required_config_modules
        modules: dict[str, Any] = {}

        for module_name, module_spec in model_index.items():
            if not isinstance(module_spec, list) or len(module_spec) < 1:
                continue
            transformers_or_diffusers = module_spec[0]
            if transformers_or_diffusers is None:
                if module_name in self.required_config_modules:
                    self.required_config_modules.remove(module_name)
                continue
            if module_name not in required_modules:
                continue
            if loaded_modules is not None and module_name in loaded_modules:
                modules[module_name] = loaded_modules[module_name]
                continue

            component_model_path = os.path.join(self.model_path, module_name)
            if module_name == "tokenizer" and not os.path.isdir(
                    component_model_path):
                gemma_path = os.path.join(self.model_path, "text_encoder",
                                          "gemma")
                if os.path.isdir(gemma_path):
                    component_model_path = gemma_path
                else:
                    raise ValueError(
                        "Tokenizer directory missing and Gemma weights were not found."
                    )

            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                fastvideo_args=fastvideo_args,
            )
            logger.info("Loaded module %s from %s", module_name,
                        component_model_path)
            modules[module_name] = module

        if "tokenizer" in required_modules and "tokenizer" not in modules:
            gemma_path = os.path.join(self.model_path, "text_encoder", "gemma")
            if os.path.isdir(gemma_path):
                modules["tokenizer"] = AutoTokenizer.from_pretrained(
                    gemma_path, local_files_only=True)

        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(
                    f"Required module {module_name} was not loaded properly")

        return modules


EntryClass = LTX2Pipeline
