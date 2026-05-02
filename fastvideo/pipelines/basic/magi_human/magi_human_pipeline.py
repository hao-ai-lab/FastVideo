# SPDX-License-Identifier: Apache-2.0
"""MagiHuman base text-to-AV pipeline.

Top-level composition for the daVinci-MagiHuman base model. Wires:

    InputValidationStage -> TextEncodingStage (T5-Gemma)
        -> MagiHumanLatentPreparationStage
        -> MagiHumanDenoisingStage
        -> DecodingStage (Wan 2.2 TI2V-5B VAE decode for video)
        -> MagiHumanAudioDecodingStage (Stable Audio Open 1.0 VAE decode)

The base checkpoint is a joint audio-visual generator; both the video
and audio paths run in the denoising loop and both are decoded.

`load_modules` is overridden so the four cross-variant shared components
(text_encoder, tokenizer, audio_vae, video vae) lazy-load from their
canonical upstream HF repos at first build time instead of being
bundled inside every converted MagiHuman variant. This keeps each
variant's converted repo at ~5-30 GB (transformer + scheduler +
model_index.json) instead of ~30-55 GB, and lets all variants share
the same ~25 GB of cached upstream weights.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from fastvideo.configs.models.encoders.t5gemma import T5GemmaEncoderConfig
from fastvideo.configs.models.vaes import OobleckVAEConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.encoders.t5gemma import T5GemmaEncoderModel
from fastvideo.models.vaes.sa_audio import SAAudioVAEModel
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.pipelines.basic.magi_human.stages import (
    MagiHumanAudioDecodingStage,
    MagiHumanDenoisingStage,
    MagiHumanLatentPreparationStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.stages import (
    DecodingStage,
    InputValidationStage,
    TextEncodingStage,
)

logger = init_logger(__name__)

_T5GEMMA_HF_ID = "google/t5gemma-9b-9b-ul2"
_SA_AUDIO_HF_ID = "stabilityai/stable-audio-open-1.0"
_WAN_VAE_HF_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _ensure_hf_token_env() -> str | None:
    """Surface any of the three common HF token env vars as `HF_TOKEN`.

    FastVideo workers spawn child processes that inherit env; both
    `huggingface_hub` and `transformers.AutoTokenizer.from_pretrained`
    look at `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` by default but not
    `HF_API_KEY`. If only the latter is set, gated downloads fail with
    401. Aliasing at pipeline-load time is the minimum-disruption fix.
    """
    for src in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        value = os.environ.get(src)
        if value:
            os.environ.setdefault("HF_TOKEN", value)
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", value)
            return value
    return None


class MagiHumanPipeline(ComposedPipelineBase):
    """Base MagiHuman text-to-AV pipeline (no LoRA, no distill, no SR)."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "audio_vae",
    ]

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load the variant-specific transformer + scheduler from the
        converted MagiHuman repo and lazy-load the four cross-variant
        shared components from their canonical upstream HF repos:

          * text_encoder, tokenizer -> ``google/t5gemma-9b-9b-ul2``
            (gated, requires HF token with accepted terms of use)
          * audio_vae -> ``stabilityai/stable-audio-open-1.0`` (gated)
          * vae -> ``Wan-AI/Wan2.2-TI2V-5B-Diffusers``

        Backwards-compatible with bundled converted repos: if any of
        these subfolders is present locally and listed in
        ``model_index.json``, the standard component loader picks it up
        via super(). Otherwise the loader is told to skip the entry and
        we lazy-load it here.
        """
        # T5-Gemma is gated: expose `HF_API_KEY` as `HF_TOKEN` if needed.
        _ensure_hf_token_env()

        # Temporarily drop the cross-variant shared keys from
        # required_config_modules so super() does not fail the "every
        # required entry must appear in model_index.json" check when
        # the converted repo declares a minimal model_index.json.
        deferred = []
        for key in ("text_encoder", "tokenizer", "audio_vae", "vae"):
            if key in self.required_config_modules:
                self.required_config_modules.remove(key)
                deferred.append(key)

        try:
            modules = super().load_modules(fastvideo_args, loaded_modules)
        finally:
            for key in deferred:
                if key not in self.required_config_modules:
                    self.required_config_modules.append(key)

        if loaded_modules and "text_encoder" in loaded_modules:
            modules["text_encoder"] = loaded_modules["text_encoder"]
        else:
            logger.info("Building T5-Gemma text encoder (lazy-load from %s)", _T5GEMMA_HF_ID)
            enc_config = T5GemmaEncoderConfig()
            enc_config.arch_config.t5gemma_model_path = _T5GEMMA_HF_ID
            modules["text_encoder"] = T5GemmaEncoderModel(enc_config)

        if loaded_modules and "tokenizer" in loaded_modules:
            modules["tokenizer"] = loaded_modules["tokenizer"]
        else:
            logger.info("Loading T5-Gemma tokenizer from %s", _T5GEMMA_HF_ID)
            modules["tokenizer"] = AutoTokenizer.from_pretrained(_T5GEMMA_HF_ID)

        if loaded_modules and "audio_vae" in loaded_modules:
            modules["audio_vae"] = loaded_modules["audio_vae"]
        else:
            logger.info(
                "Building Stable Audio Open 1.0 VAE (lazy-load from %s) — "
                "requires HF terms accepted for gated repo",
                _SA_AUDIO_HF_ID,
            )
            audio_config = OobleckVAEConfig()
            audio_config.pretrained_path = _SA_AUDIO_HF_ID
            modules["audio_vae"] = SAAudioVAEModel(audio_config)

        if loaded_modules and "vae" in loaded_modules:
            modules["vae"] = loaded_modules["vae"]
        else:
            modules["vae"] = self._load_video_vae(fastvideo_args)

        return modules

    def _load_video_vae(self, fastvideo_args: FastVideoArgs) -> Any:
        """Resolve the video VAE: prefer a bundled ``vae/`` subfolder in
        the converted repo (legacy), fall back to lazy-downloading the
        Wan 2.2 TI2V-5B VAE shards from upstream.

        Either way the load goes through FastVideo's standard
        ``VAELoader`` so the result is the same FV ``AutoencoderKLWan``
        nn.Module that the bundled path produces.
        """
        from fastvideo.models.loader.component_loader import VAELoader

        bundled = Path(self.model_path) / "vae"
        if bundled.is_dir() and (bundled / "config.json").is_file():
            logger.info("Loading bundled video VAE from %s", bundled)
            return VAELoader().load(str(bundled), fastvideo_args)

        from huggingface_hub import snapshot_download

        logger.info(
            "Bundled vae/ not found at %s; lazy-loading Wan 2.2 TI2V-5B VAE from %s",
            self.model_path,
            _WAN_VAE_HF_ID,
        )
        snapshot = snapshot_download(
            repo_id=_WAN_VAE_HF_ID,
            allow_patterns=["vae/*"],
        )
        vae_dir = os.path.join(snapshot, "vae")
        if not os.path.isdir(vae_dir):
            raise RuntimeError(
                f"snapshot_download returned {snapshot} but no vae/ "
                f"subfolder was found inside it. Check that {_WAN_VAE_HF_ID} "
                "still exposes a Diffusers-format vae/ folder.", )
        return VAELoader().load(vae_dir, fastvideo_args)

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        # MagiHuman applies `flow_shift` during timestep setup; keep the
        # scheduler constructor at its default no-op shift.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler()

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        pc = fastvideo_args.pipeline_config
        dit_arch = pc.dit_config.arch_config

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

        # Data-proxy + eval knobs come from the PipelineConfig (`pc`).
        # Only DiT-architecture fields live on `dit_arch` now.
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=MagiHumanLatentPreparationStage(
                vae_stride=tuple(pc.vae_stride),
                z_dim=pc.z_dim,
                patch_size=tuple(dit_arch.patch_size),
                fps=pc.fps,
                t5_gemma_target_length=pc.t5_gemma_target_length,
                coords_style=pc.coords_style,
                text_offset=pc.text_offset,
                audio_in_channels=dit_arch.audio_in_channels,
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=MagiHumanDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                patch_size=tuple(dit_arch.patch_size),
                video_in_channels=dit_arch.video_in_channels,
                audio_in_channels=dit_arch.audio_in_channels,
                video_txt_guidance_scale=pc.video_txt_guidance_scale,
                audio_txt_guidance_scale=pc.audio_txt_guidance_scale,
                cfg_number=pc.cfg_number,
                coords_style=pc.coords_style,
                video_guidance_high_t_threshold=pc.video_guidance_high_t_threshold,
                video_guidance_low_t_value=pc.video_guidance_low_t_value,
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )

        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=MagiHumanAudioDecodingStage(audio_vae=self.get_module("audio_vae"), ),
        )


EntryClass = MagiHumanPipeline
