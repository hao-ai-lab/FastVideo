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

`load_modules` is overridden because T5-Gemma (`google/t5gemma-9b-9b-ul2`)
is a gated 18 GB Google repo we intentionally do NOT bundle inside the
converted MagiHuman directory; the pipeline pulls it at load time from
the HF hub using the caller's HF token.
"""
from __future__ import annotations

import os
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
        """Load bundled modules via super(), then add T5-Gemma externally.

        The converted MagiHuman repo bundles `transformer/`, `vae/`, and
        `scheduler/`. The text encoder and tokenizer are fetched from
        `google/t5gemma-9b-9b-ul2` (gated Google repo, requires HF token
        with accepted terms of use — see `initialize_pipeline`).
        """
        # T5-Gemma is gated: expose `HF_API_KEY` as `HF_TOKEN` if needed.
        _ensure_hf_token_env()

        # Temporarily drop text_encoder + tokenizer + audio_vae so
        # super() doesn't fail the "every required_config_modules entry
        # must be in modules" check; we'll load them ourselves below.
        # All three are lazy-loaded from their own HF repos rather than
        # bundled in the converted MagiHuman directory: T5-Gemma (gated),
        # Stable Audio Open (gated), and the tokenizer (small, lives
        # alongside T5-Gemma).
        deferred = []
        for key in ("text_encoder", "tokenizer", "audio_vae"):
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

        return modules

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs) -> None:
        # MagiHuman's reference uses UniPC on its own — so we make sure we
        # build FastVideo's matching `FlowUniPCMultistepScheduler` here.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift, )

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
