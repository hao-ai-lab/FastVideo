# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K: regenerate a 768P MiniMax-H3 draft video at 2560x1440.

``draft video -> H3 VAE encode -> latent upscaler -> 3 fixed-sigma Euler steps (H3 DiT + merged LoRA + Sharp curve
AdaLN branches) -> H3 VAE decode``. The release repo holds the unmodified MiniMax-H3 components under ``base_model/``
next to ``lora/`` and ``latent_upsampler/``; see ``docs/inference/open_h3_regenerate_2k.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from fastvideo import envs
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.upsamplers.minimax_h3_latent_upscaler import MiniMaxH3LatentUpscaler
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
from fastvideo.pipelines.basic.minimax_h3.stages import MiniMaxH3VideoDecodingStage
from fastvideo.pipelines.basic.open_h3_regenerate_2k.sharp_curve import SharpCurveBranches
from fastvideo.pipelines.basic.open_h3_regenerate_2k.stages import (
    Regenerate2KDenoisingStage,
    Regenerate2KFinalizeStage,
    Regenerate2KLatentPreparationStage,
)
from fastvideo.utils import maybe_download_model, verify_model_config_and_directory

logger = init_logger(__name__)

BASE_MODEL_DIR = "base_model"
LORA_FILE = "lora/open_h3_regenerate_2k_lora.safetensors"
UPSCALER_FILE = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
_BASE_COMPONENTS = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae", "transformer", "scheduler",
                    "audio_scheduler")


def resolve_refine_transformer_dir(configured: str | None) -> str | None:
    """Absolute DiT component directory for ``refine_transformer_dir``; ``None`` keeps the release's own."""
    if configured is None:
        return None
    root = os.path.abspath(os.path.expanduser(configured))
    nested = os.path.join(root, "transformer")
    path = nested if os.path.isfile(os.path.join(nested, "config.json")) else root
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise ValueError(f"refine_transformer_dir {configured!r} holds no MiniMax-H3 transformer config.json.")
    return path


def refuse_adaln_cache() -> None:
    """Raise when an H3 AdaLN cache is enabled: it bypasses the projection modules that carry the Sharp branches."""
    if envs.FASTVIDEO_H3_ADALN_CACHE.get() or envs.FASTVIDEO_H3_ADALN_TABLE.get():
        raise ValueError("Open-H3-Regenerate-2K needs the AdaLN projections: unset FASTVIDEO_H3_ADALN_CACHE and "
                         "FASTVIDEO_H3_ADALN_TABLE, whose cached modulation skips the 50 Sharp curve branches.")


class OpenH3Regenerate2KPipeline(MiniMaxH3Pipeline):
    """MiniMax-H3 T2VA layout with a draft-latent start, fixed sigmas and a silent output."""

    _extra_config_module_map = {name: f"{BASE_MODEL_DIR}/{name}" for name in _BASE_COMPONENTS}
    # The merged LoRA reaches only these linears; the AdaLN projections carry the Sharp curve branches instead.
    lora_target_modules = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "ff.fc_in", "ff.fc_out"]

    @classmethod
    def get_hf_download_allow_patterns(cls) -> list[str]:
        components = [f"{BASE_MODEL_DIR}/{name}/**" for name in _BASE_COMPONENTS]
        return [f"{BASE_MODEL_DIR}/model_index.json", *components, "lora/**", "latent_upsampler/*.safetensors"]

    def _load_config(self, model_path: str) -> dict[str, Any]:
        refuse_adaln_cache()
        revision = getattr(self.fastvideo_args, "revision", None)
        config = self.fastvideo_args.pipeline_config
        self.model_path = maybe_download_model(model_path,
                                               revision=revision,
                                               allow_patterns=self.get_hf_download_allow_patterns())
        if self.fastvideo_args.lora_path is None:
            self.fastvideo_args.lora_path = os.path.join(self.model_path, LORA_FILE)
        refine_transformer_dir = resolve_refine_transformer_dir(config.refine_transformer_dir)
        if refine_transformer_dir is not None:
            logger.info("Refining on the configured DiT at %s", refine_transformer_dir)
            self._extra_config_module_map = {**self._extra_config_module_map, "transformer": refine_transformer_dir}
        return verify_model_config_and_directory(os.path.join(self.model_path, BASE_MODEL_DIR),
                                                 required_component_dirs=())

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        super().initialize_pipeline(fastvideo_args)
        self._upscaler: MiniMaxH3LatentUpscaler | None = None
        self._curve: SharpCurveBranches | None = None

    def _load_upscaler(self) -> MiniMaxH3LatentUpscaler:
        if self._upscaler is None:
            model = MiniMaxH3LatentUpscaler.from_checkpoint(os.path.join(self.model_path, UPSCALER_FILE))
            self._upscaler = model.to(device=self.device, dtype=torch.bfloat16)
        return self._upscaler

    def _load_curve(self) -> SharpCurveBranches:
        if self._curve is None:
            self._curve = SharpCurveBranches(Path(self.model_path))
        return self._curve

    def _add_denoise_stages(self, *, ref2va: bool) -> None:
        config = self.fastvideo_args.pipeline_config
        vae = self.get_module("vae")
        audio_vae = self.get_module("audio_vae")
        transformer = self.get_module("transformer")
        if vae is None or audio_vae is None or transformer is None:
            raise RuntimeError("Open-H3-Regenerate-2K needs the video VAE, audio VAE and transformer loaded.")
        self.add_stage(
            "latent_preparation_stage",
            Regenerate2KLatentPreparationStage(vae,
                                               audio_vae,
                                               self.get_module("scheduler"),
                                               load_upscaler=self._load_upscaler,
                                               draft_size=config.draft_size,
                                               start_sigma=config.refine_start_sigma,
                                               draft_latent_path=config.draft_latent_path))
        self.add_stage("denoising_stage",
                       Regenerate2KDenoisingStage(transformer, self._load_curve(), config.refine_start_sigma))
        self.add_stage("video_decoding_stage", MiniMaxH3VideoDecodingStage(vae=vae))
        self.add_stage("audio_decoding_stage", Regenerate2KFinalizeStage())
        self._denoise_stages_ready = True


EntryClass = OpenH3Regenerate2KPipeline

__all__ = ["EntryClass", "OpenH3Regenerate2KPipeline", "resolve_refine_transformer_dir"]
