# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan video diffusion pipeline implementation.

This module contains an implementation of the Hunyuan video diffusion pipeline
using the modular pipeline architecture.
"""

from diffusers.image_processor import VaeImageProcessor

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.v1.pipelines.stages import (StepvideoPromptEncodingStage,
                                           DecodingStage, StepVideoDecodingStage,
                                           DenoisingStage, InputValidationStage,
                                           LatentPreparationStage,
                                           TimestepPreparationStage)

from copy import deepcopy
from typing import Any, Dict
from fastvideo.v1.models.loader.component_loader import PipelineComponentLoader
import os
logger = init_logger(__name__)


import pickle
def call_api_gen(url, api, port=8080):
    url = f"http://{url}:{port}/{api}-api"
    import aiohttp

    async def _fn(samples, *args, **kwargs):
        if api == 'vae':
            data = {
                "samples": samples,
            }
        elif api == 'caption':
            data = {
                "prompts": samples,
            }
        else:
            raise Exception(f"Not supported api: {api}...")

        async with aiohttp.ClientSession() as sess:
            data_bytes = pickle.dumps(data)
            async with sess.get(url, data=data_bytes, timeout=12000) as response:
                result = bytearray()
                while not response.content.at_eof():
                    chunk = await response.content.read(1024)
                    result += chunk
                response_data = pickle.loads(result)
        return response_data

    return _fn


class StepVideoPipeline(ComposedPipelineBase):

    _required_config_modules = [
        "transformer", "scheduler"
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=StepvideoPromptEncodingStage(
                           caption_client=self.get_module("caption"),
                       ))

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(
                           transformer=self.get_module("transformer"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=StepVideoDecodingStage(vae_client=self.get_module("vae")))

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        """
        Initialize the pipeline.
        """
        caption = call_api_gen("127.0.0.1", 'caption')
        vae = call_api_gen("127.0.0.1", 'vae')
        self.add_module("caption", caption)
        self.add_module("vae", vae)
        # same as setting vae_scale_factor to 16 for default. TODO: check original implementation
        fastvideo_args.vae_scale_factor = vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        fastvideo_args.num_channels_latents = self.get_module("transformer").in_channels
        
    def load_modules(self, fastvideo_args: FastVideoArgs) -> Dict[str, Any]:
        """
        Load the modules from the config.
        """
        logger.info("Loading pipeline modules from config: %s", self.config)
        modules_config = deepcopy(self.config)

        # remove keys that are not pipeline modules
        modules_config.pop("_class_name")
        modules_config.pop("_diffusers_version")

        # some sanity checks
        assert len(
            modules_config
        ) > 1, "model_index.json must contain at least one pipeline module"

        required_modules = [
            "transformer", "scheduler"
        ]
        for module_name in required_modules:
            if module_name not in modules_config:
                raise ValueError(
                    f"model_index.json must contain a {module_name} module")
        logger.info("Diffusers config passed sanity checks")

        # all the component models used by the pipeline
        modules = {}
        for module_name, (transformers_or_diffusers,
                          architecture) in modules_config.items():
            component_model_path = os.path.join(self.model_path, module_name)
            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                architecture=architecture,
                fastvideo_args=fastvideo_args,
            )
            logger.info("Loaded module %s from %s", module_name,
                        component_model_path)

            if module_name in modules:
                logger.warning("Overwriting module %s", module_name)
            modules[module_name] = module

        required_modules = self.required_config_modules
        # Check if all required modules were loaded
        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(
                    f"Required module {module_name} was not loaded properly")

        return modules


EntryClass = StepVideoPipeline
