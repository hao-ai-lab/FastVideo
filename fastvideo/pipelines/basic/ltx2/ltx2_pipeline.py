# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 text-to-video pipeline.
"""

import os
from typing import Any

from transformers import AutoTokenizer

from fastvideo.api.resolution import ResolvedGeneratorConfig
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.lora_pipeline import LoRAPipeline
from fastvideo.pipelines.stages import (DecodingStage, InputValidationStage, LTX2AudioDecodingStage, LTX2DenoisingStage,
                                        LTX2LatentPreparationStage, LTX2RefineInitStage, LTX2RefineLoRAStage,
                                        LTX2TextEncodingStage, LTX2UpsampleStage, STAGE_2_DISTILLED_SIGMA_VALUES)

logger = init_logger(__name__)

# Values of the stage-2 refine settings that neither the input nor the checkpoint's model_index.json decided.
_REFINE_DEFAULTS: dict[str, Any] = {
    "pipeline.ltx2.refine.enabled": False,
    "pipeline.ltx2.refine.add_noise": True,
    "pipeline.ltx2.refine.guidance_scale": 1.0,
    "pipeline.ltx2.refine.num_inference_steps": 3,
}
# Refine component paths, by the model_index.json key that supplies the checkpoint's default.
_REFINE_PATH_KEYS = {
    "pipeline.ltx2.refine.transformer_path": "fastvideo_refine_transformer_path",
    "pipeline.ltx2.refine.lora_path": "fastvideo_refine_lora_path",
    "pipeline.ltx2.refine.noise_path": "fastvideo_refine_noise_path",
    "pipeline.ltx2.refine.audio_noise_path": "fastvideo_refine_audio_noise_path",
}


def _resolve_refine_path(model_path: str, value: str | None) -> str | None:
    """Resolve a refine component path, preferring paths inside the local
    model snapshot when the value is relative."""
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    candidate = os.path.join(model_path, value)
    if os.path.exists(candidate):
        return candidate
    return value


def _resolve_refine_upsampler_path(model_path: str, model_index: dict[str, Any]) -> str | None:
    """Resolve the refine upsampler directory for a local model snapshot.

    model_index.json keys ("fastvideo_refine_upsampler_path", then
    "spatial_upsampler") are the documented override and win when present.
    Otherwise probe the snapshot root: published LTX-2 distilled checkpoints
    bundle the upsampler weights on disk (spelled "spatial_upscaler")
    without declaring them in model_index.json.
    """
    path = _resolve_refine_path(model_path, model_index.get("fastvideo_refine_upsampler_path"))
    if path is None and "spatial_upsampler" in model_index:
        path = _resolve_refine_path(model_path, "spatial_upsampler")
    if path is None:
        for dirname in ("spatial_upscaler", "spatial_upsampler"):
            candidate = os.path.join(model_path, dirname)
            if os.path.isdir(candidate):
                return candidate
    return path


def _checkpoint_refine_values(resolved_config: ResolvedGeneratorConfig, model_path: str,
                              model_index: dict[str, Any]) -> dict[str, Any]:
    """The refine settings that the checkpoint's ``model_index.json`` and the refine defaults change.

    Distilled checkpoints bundle ``fastvideo_refine_*`` defaults so the pipeline can configure itself. Each one fills
    its setting only while the setting is ``None``, so a value from the input always wins; ``fastvideo_refine_enabled``
    can only turn refine on. The upsampler falls back to a directory in the snapshot. The four switches that are still
    ``None`` afterwards take ``_REFINE_DEFAULTS``. The result holds only the paths whose value changes.
    """

    def current(path: str) -> Any:
        return resolved_config.provenance(path).value

    values: dict[str, Any] = {}
    if model_index.get("fastvideo_refine_enabled") is True and current("pipeline.ltx2.refine.enabled") is None:
        values["pipeline.ltx2.refine.enabled"] = True
    if current("pipeline.components.upsampler_weights") is None:
        values["pipeline.components.upsampler_weights"] = _resolve_refine_upsampler_path(model_path, model_index)
    for path, key in _REFINE_PATH_KEYS.items():
        if current(path) is None:
            values[path] = _resolve_refine_path(model_path, model_index.get(key))
    for path, key, convert in (
        ("pipeline.ltx2.refine.num_inference_steps", "fastvideo_refine_num_inference_steps", int),
        ("pipeline.ltx2.refine.guidance_scale", "fastvideo_refine_guidance_scale", float),
        ("pipeline.ltx2.refine.add_noise", "fastvideo_refine_add_noise", bool),
    ):
        if current(path) is None and model_index.get(key) is not None:
            values[path] = convert(model_index[key])
    for path, default in _REFINE_DEFAULTS.items():
        if values.get(path, current(path)) is None:
            values[path] = default
    return {path: value for path, value in values.items() if value != current(path)}


class LTX2Pipeline(LoRAPipeline):

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
        "audio_vae",
        "vocoder",
    ]

    def create_pipeline_stages(self, resolved_config: ResolvedGeneratorConfig):
        refine_enabled = resolved_config.pipeline.ltx2.refine.enabled

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
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=LTX2DenoisingStage(transformer=self.get_module("transformer"), ),
        )

        if refine_enabled:
            stage2_steps = resolved_config.pipeline.ltx2.refine.num_inference_steps
            # LTX-2 refine currently supports two explicitly tested step
            # counts:
            # - 3 steps: official distilled schedule
            # - 2 steps: custom reduced schedule used for faster
            #   experimentation
            # Other values are intentionally rejected to avoid silent
            # quality regressions.
            if stage2_steps == 3:
                # Official distilled stage-2 refine schedule.
                stage2_sigmas = STAGE_2_DISTILLED_SIGMA_VALUES
            elif stage2_steps == 2:
                # Reduced 2-step refine schedule (explicitly omits 0.725).
                stage2_sigmas = [
                    STAGE_2_DISTILLED_SIGMA_VALUES[0],
                    STAGE_2_DISTILLED_SIGMA_VALUES[2],
                    STAGE_2_DISTILLED_SIGMA_VALUES[3],
                ]
            else:
                logger.warning(
                    "For LTX-2 refinement, "
                    "pipeline.ltx2.refine.num_inference_steps=%s is not a tested "
                    "setting. Using denoising steps other than 2 or 3 "
                    "may cause quality degradation.",
                    stage2_steps,
                )
                raise ValueError("LTX-2 refinement supports only 2 or 3 denoising "
                                 "steps.")

            transformer_refine = self.get_module("transformer_refine", self.get_module("transformer"))

            self.add_stage(
                stage_name="ltx2_upsample_stage",
                stage=LTX2UpsampleStage(
                    upsampler=self.get_module("spatial_upsampler"),
                    vae=self.get_module("vae"),
                    transformer=transformer_refine,
                    sigmas=stage2_sigmas,
                    add_noise=resolved_config.pipeline.ltx2.refine.add_noise,
                ),
            )

            if resolved_config.pipeline.ltx2.refine.lora_path:
                self.add_stage(
                    stage_name="ltx2_refine_lora_stage",
                    stage=LTX2RefineLoRAStage(
                        pipeline=self,
                        lora_path=resolved_config.pipeline.ltx2.refine.lora_path,
                    ),
                )

            self.add_stage(
                stage_name="ltx2_refine_denoising_stage",
                stage=LTX2DenoisingStage(
                    transformer=transformer_refine,
                    sigmas_override=stage2_sigmas,
                    num_inference_steps_override=len(stage2_sigmas) - 1,
                    force_guidance_scale=(resolved_config.pipeline.ltx2.refine.guidance_scale),
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

    def initialize_pipeline(self, resolved_config: ResolvedGeneratorConfig):
        tokenizer = self.get_module("tokenizer")
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

    def load_modules(
        self,
        resolved_config: ResolvedGeneratorConfig,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        model_index = self._load_config(self.model_path)
        logger.info("Loading pipeline modules from config: %s", model_index)

        # Apply the refine defaults bundled in model_index.json, then the remaining refine defaults. The pipeline owns
        # its config; later stages read the rebound one.
        refine_values = _checkpoint_refine_values(resolved_config, self.model_path, model_index)
        if refine_values:
            resolved_config = resolved_config.with_override("checkpoint:model_index.json", refine_values)
            self.resolved_config = resolved_config

        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        model_index.pop("workload_type", None)

        if len(model_index) <= 1:
            raise ValueError("model_index.json must contain at least one pipeline module")

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
            if module_name == "tokenizer" and not os.path.isdir(component_model_path):
                gemma_path = os.path.join(self.model_path, "text_encoder", "gemma")
                if os.path.isdir(gemma_path):
                    component_model_path = gemma_path
                else:
                    raise ValueError("Tokenizer directory missing and Gemma weights "
                                     "were not found.")

            module = PipelineComponentLoader.load_module(
                module_name=module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                resolved_config=resolved_config,
            )
            logger.info("Loaded module %s from %s", module_name, component_model_path)
            modules[module_name] = module

        if "tokenizer" in required_modules and "tokenizer" not in modules:
            gemma_path = os.path.join(self.model_path, "text_encoder", "gemma")
            if os.path.isdir(gemma_path):
                modules["tokenizer"] = AutoTokenizer.from_pretrained(gemma_path, local_files_only=True)

        for module_name in required_modules:
            if module_name not in modules or modules[module_name] is None:
                raise ValueError(f"Required module {module_name} was not loaded properly")

        if resolved_config.pipeline.ltx2.refine.enabled:
            upsampler_path = resolved_config.pipeline.components.upsampler_weights
            if upsampler_path is None:
                raise ValueError("pipeline.ltx2.refine.enabled is True but "
                                 "pipeline.components.upsampler_weights was not provided.")
            if not os.path.isdir(upsampler_path):
                raise ValueError("pipeline.components.upsampler_weights must be a directory "
                                 "containing Diffusers-style upsampler weights; "
                                 f"got {upsampler_path}")
            config_path = os.path.join(upsampler_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError("pipeline.components.upsampler_weights must contain a Diffusers "
                                 f"config.json; missing {config_path}")
            if (loaded_modules is not None and "spatial_upsampler" in loaded_modules):
                modules["spatial_upsampler"] = loaded_modules["spatial_upsampler"]
            else:
                modules["spatial_upsampler"] = (PipelineComponentLoader.load_module(
                    module_name="spatial_upsampler",
                    component_model_path=upsampler_path,
                    transformers_or_diffusers="diffusers",
                    resolved_config=resolved_config,
                ))
            logger.info("Loaded module spatial_upsampler from %s", upsampler_path)

            if (loaded_modules is not None and "transformer_refine" in loaded_modules):
                modules["transformer_refine"] = loaded_modules["transformer_refine"]
            elif resolved_config.pipeline.ltx2.refine.transformer_path:
                modules["transformer_refine"] = (PipelineComponentLoader.load_module(
                    module_name="transformer_refine",
                    component_model_path=(resolved_config.pipeline.ltx2.refine.transformer_path),
                    transformers_or_diffusers="diffusers",
                    resolved_config=resolved_config,
                ))
                logger.info(
                    "Loaded module transformer_refine from %s",
                    resolved_config.pipeline.ltx2.refine.transformer_path,
                )

        return modules


EntryClass = LTX2Pipeline
