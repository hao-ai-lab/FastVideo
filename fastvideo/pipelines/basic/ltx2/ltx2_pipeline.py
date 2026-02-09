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
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.stages import (
    DecodingStage, InputValidationStage, LTX2AudioDecodingStage,
    LTX2DenoisingStage, LTX2LatentPreparationStage, LTX2RefineInitStage,
    LTX2RefineLoRAStage, LTX2UpsampleStage, STAGE_2_DISTILLED_SIGMA_VALUES,
    LTX2TextEncodingStage)

logger = init_logger(__name__)


class LTX2Pipeline(LoRAPipeline):

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        refine_enabled = fastvideo_args.ltx2_refine_enabled

        self.add_stage(
            stage_name="input_validation_stage",
            stage=InputValidationStage(),
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=LTX2TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        if refine_enabled:
            self.add_stage(
                stage_name="ltx2_refine_init_stage",
                stage=LTX2RefineInitStage(),
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

        if refine_enabled:
            stage2_sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
            stage2_steps = fastvideo_args.ltx2_refine_num_inference_steps
            expected_steps = len(stage2_sigmas) - 1
            if stage2_steps != expected_steps:
                logger.warning(
                    "ltx2_refine_num_inference_steps=%s does not match distilled schedule; "
                    "using %s steps to align with stage2 sigmas.",
                    stage2_steps,
                    expected_steps,
                )
                stage2_steps = expected_steps

            transformer_refine = self.get_module("transformer_refine",
                                                 self.get_module("transformer"))

            self.add_stage(
                stage_name="ltx2_upsample_stage",
                stage=LTX2UpsampleStage(
                    upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    transformer=transformer_refine,
                    sigmas=stage2_sigmas,
                    add_noise=fastvideo_args.ltx2_refine_add_noise,
                ),
            )

            if fastvideo_args.ltx2_refine_lora_path:
                self.add_stage(
                    stage_name="ltx2_refine_lora_stage",
                    stage=LTX2RefineLoRAStage(
                        pipeline=self,
                        lora_path=fastvideo_args.ltx2_refine_lora_path,
                    ),
                )

            self.add_stage(
                stage_name="ltx2_refine_denoising_stage",
                stage=LTX2DenoisingStage(
                    transformer=transformer_refine,
                    sigmas_override=stage2_sigmas,
                    num_inference_steps_override=stage2_steps,
                    force_guidance_scale=fastvideo_args.
                    ltx2_refine_guidance_scale,
                    initial_audio_latents_key="ltx2_audio_latents",
                ),
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
        if fastvideo_args.debug_model_sums:
            os.environ["LTX2_PIPELINE_DEBUG_LOG"] = "1"
            if fastvideo_args.debug_model_sums_path:
                os.environ[
                    "LTX2_PIPELINE_DEBUG_PATH"] = fastvideo_args.debug_model_sums_path
            else:
                logger.warning(
                    "debug_model_sums is enabled but debug_model_sums_path is not set; no model sums will be logged."
                )
        else:
            os.environ.pop("LTX2_PIPELINE_DEBUG_LOG", None)
            os.environ.pop("LTX2_PIPELINE_DEBUG_PATH", None)
        if fastvideo_args.debug_model_detail:
            os.environ["LTX2_DEBUG_DETAIL"] = "1"
            if fastvideo_args.debug_model_detail_path:
                os.environ[
                    "LTX2_PIPELINE_DEBUG_DETAIL_PATH"] = fastvideo_args.debug_model_detail_path
            else:
                logger.warning(
                    "debug_model_detail is enabled but debug_model_detail_path is not set; no detailed hooks will be logged."
                )
        else:
            os.environ.pop("LTX2_DEBUG_DETAIL", None)
            os.environ.pop("LTX2_PIPELINE_DEBUG_DETAIL_PATH", None)

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

        # Apply optional FastVideo-specific refine defaults embedded in model_index.json.
        def _resolve_refine_path(value: str | None) -> str | None:
            if value is None:
                return None
            if os.path.isabs(value):
                return value
            candidate = os.path.join(self.model_path, value)
            if os.path.exists(candidate):
                return candidate
            return value

        if model_index.get("fastvideo_refine_enabled") is True:
            if fastvideo_args.refine_enabled is None:
                fastvideo_args.ltx2_refine_enabled = True
        if fastvideo_args.refine_upsampler_path is None and fastvideo_args.ltx2_refine_upsampler_path is None:
            fastvideo_args.ltx2_refine_upsampler_path = _resolve_refine_path(
                model_index.get("fastvideo_refine_upsampler_path"))
            if fastvideo_args.ltx2_refine_upsampler_path is None and "spatial_upsampler" in model_index:
                fastvideo_args.ltx2_refine_upsampler_path = _resolve_refine_path(
                    "spatial_upsampler")
        if fastvideo_args.refine_transformer_path is None and fastvideo_args.ltx2_refine_transformer_path is None:
            fastvideo_args.ltx2_refine_transformer_path = _resolve_refine_path(
                model_index.get("fastvideo_refine_transformer_path"))
        if fastvideo_args.refine_lora_path is None and fastvideo_args.ltx2_refine_lora_path is None:
            fastvideo_args.ltx2_refine_lora_path = _resolve_refine_path(
                model_index.get("fastvideo_refine_lora_path"))
        if fastvideo_args.refine_num_inference_steps is None and model_index.get(
                "fastvideo_refine_num_inference_steps") is not None:
            fastvideo_args.ltx2_refine_num_inference_steps = int(
                model_index["fastvideo_refine_num_inference_steps"])
        if fastvideo_args.refine_guidance_scale is None and model_index.get(
                "fastvideo_refine_guidance_scale") is not None:
            fastvideo_args.ltx2_refine_guidance_scale = float(
                model_index["fastvideo_refine_guidance_scale"])
        if fastvideo_args.refine_add_noise is None and model_index.get(
                "fastvideo_refine_add_noise") is not None:
            fastvideo_args.ltx2_refine_add_noise = bool(
                model_index["fastvideo_refine_add_noise"])
        if fastvideo_args.refine_noise_path is None and fastvideo_args.ltx2_refine_noise_path is None:
            fastvideo_args.ltx2_refine_noise_path = _resolve_refine_path(
                model_index.get("fastvideo_refine_noise_path"))
        if fastvideo_args.refine_audio_noise_path is None and fastvideo_args.ltx2_refine_audio_noise_path is None:
            fastvideo_args.ltx2_refine_audio_noise_path = _resolve_refine_path(
                model_index.get("fastvideo_refine_audio_noise_path"))

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

        if fastvideo_args.ltx2_refine_enabled:
            upsampler_path = fastvideo_args.ltx2_refine_upsampler_path
            if upsampler_path is None:
                raise ValueError(
                    "ltx2_refine_enabled is True but ltx2_refine_upsampler_path was not provided."
                )
            if not os.path.isdir(upsampler_path):
                raise ValueError(
                    "ltx2_refine_upsampler_path must be a directory containing Diffusers-style "
                    f"upsampler weights; got {upsampler_path}")
            config_path = os.path.join(upsampler_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(
                    "ltx2_refine_upsampler_path must contain a Diffusers config.json; "
                    f"missing {config_path}")
            if loaded_modules is not None and "spatial_upsampler" in loaded_modules:
                modules["spatial_upsampler"] = loaded_modules[
                    "spatial_upsampler"]
            else:
                modules[
                    "spatial_upsampler"] = PipelineComponentLoader.load_module(
                        module_name="spatial_upsampler",
                        component_model_path=upsampler_path,
                        transformers_or_diffusers="diffusers",
                        fastvideo_args=fastvideo_args,
                    )
            logger.info("Loaded module spatial_upsampler from %s",
                        upsampler_path)

            if loaded_modules is not None and "transformer_refine" in loaded_modules:
                modules["transformer_refine"] = loaded_modules[
                    "transformer_refine"]
            elif fastvideo_args.ltx2_refine_transformer_path:
                modules[
                    "transformer_refine"] = PipelineComponentLoader.load_module(
                        module_name="transformer",
                        component_model_path=fastvideo_args.
                        ltx2_refine_transformer_path,
                        transformers_or_diffusers="diffusers",
                        fastvideo_args=fastvideo_args,
                    )
                logger.info("Loaded module transformer_refine from %s",
                            fastvideo_args.ltx2_refine_transformer_path)

        return modules


EntryClass = LTX2Pipeline
