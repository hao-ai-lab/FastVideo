# SPDX-License-Identifier: Apache-2.0
"""LTX-2 preprocessing pipeline for native FastVideo training data generation.

This module defines the LTX-2 preprocess pipeline used by FastVideo workflows
to build precomputed training artifacts from raw text/video datasets.

Usage:
- Entry is through preprocess workflows that register `PreprocessPipelineT2V`.
- Input samples should provide prompt text plus video metadata/loader fields
  consumed by `TextTransformStage` and `VideoTransformStage`.
- Output artifacts are written by the shared preprocessing workflow into
  `.precomputed/` (latents, conditions, and optional audio_latents).

Optional audio path:
- When audio preprocessing is enabled, this pipeline loads the native
  LTX-2 audio encoder and stores per-sample audio latents in
  `batch.extra["ltx2_audio_latents"]`.
"""

from __future__ import annotations

import glob
import os
from typing import Any, cast

import torch
import torchaudio
from safetensors.torch import load_file as safetensors_load_file

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.audio.ltx2_audio_processing import AudioProcessor
from fastvideo.models.audio.ltx2_audio_vae import LTX2AudioEncoder
from fastvideo.models.hf_transformer_utils import get_diffusers_config
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch, PreprocessBatch
from fastvideo.pipelines.preprocess.preprocess_stages import (
    TextTransformStage, VideoTransformStage)
from fastvideo.pipelines.stages import EncodingStage, PipelineStage

logger = init_logger(__name__)


class LTX2TextPrecomputeStage(PipelineStage):
    """Compute pre-connector Gemma embeddings for LTX-2 training."""

    def __init__(
        self,
        text_encoder: torch.nn.Module,
        tokenizer: Any,
        preprocess_text_fn,
        tokenizer_kwargs: dict[str, Any],
        padding_side: str,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.preprocess_text_fn = preprocess_text_fn
        self.tokenizer_kwargs = tokenizer_kwargs
        self.padding_side = padding_side

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        batch = cast(PreprocessBatch, batch)
        assert isinstance(batch.prompt, list)

        prompts = []
        for prompt in batch.prompt:
            if not isinstance(prompt, str):
                prompt = str(prompt)
            processed_prompt = self.preprocess_text_fn(prompt)
            prompts.append(
                processed_prompt if processed_prompt is not None else "")

        prompt_embeds, prompt_attention_mask = (
            self.text_encoder.preprocess_text_embeddings(
                prompts=prompts,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                padding_side=self.padding_side,
            ))
        batch.prompt_embeds = [prompt_embeds]
        batch.prompt_attention_mask = [prompt_attention_mask]
        return batch


class LTX2AudioEncodingStage(PipelineStage):
    """Extract audio from input videos and encode into LTX-2 audio latents."""

    def __init__(
        self,
        audio_encoder: torch.nn.Module,
        audio_processor: AudioProcessor,
        fallback_fps: int,
    ) -> None:
        self.audio_encoder = audio_encoder.eval()
        self.audio_processor = audio_processor
        self.fallback_fps = fallback_fps
        self.audio_dtype = next(audio_encoder.parameters()).dtype
        self.audio_device = next(audio_encoder.parameters()).device

    @staticmethod
    def _extract_audio(
        video_path: str,
        target_duration: float,
    ) -> tuple[torch.Tensor, int] | None:
        try:
            waveform, sample_rate = torchaudio.load(video_path)
        except Exception as e:
            logger.error("Failed to load audio from %s: %s", video_path, e)
            raise e

        target_samples = int(target_duration * sample_rate)
        if target_samples <= 0:
            return None

        current_samples = waveform.shape[-1]
        if current_samples > target_samples:
            waveform = waveform[..., :target_samples]
        elif current_samples < target_samples:
            padding = target_samples - current_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, sample_rate

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        batch = cast(PreprocessBatch, batch)
        assert isinstance(batch.video_loader, list)
        assert isinstance(batch.num_frames, list)
        assert isinstance(batch.fps, list)

        audio_latents: list[torch.Tensor | None] = []
        for idx, video_input in enumerate(batch.video_loader):
            if not isinstance(video_input, str):
                logger.warning(
                    "Skipping audio for sample %s: video loader is not a path string",
                    idx,
                )
                audio_latents.append(None)
                continue

            fps = float(batch.fps[idx]) if batch.fps[idx] else float(
                self.fallback_fps)
            if fps <= 0:
                fps = float(self.fallback_fps)
            target_duration = float(batch.num_frames[idx]) / fps

            audio_data = self._extract_audio(video_input, target_duration)
            if audio_data is None:
                audio_latents.append(None)
                continue

            waveform, sample_rate = audio_data
            waveform = waveform.unsqueeze(0).to(device=self.audio_device,
                                                dtype=self.audio_dtype)
            mel = self.audio_processor.waveform_to_mel(
                waveform,
                waveform_sample_rate=sample_rate).to(device=self.audio_device,
                                                     dtype=self.audio_dtype)
            latents = self.audio_encoder(mel).squeeze(0).detach().cpu()
            audio_latents.append(latents)

        batch.extra["ltx2_audio_latents"] = audio_latents
        return batch


class PreprocessPipelineT2V(ComposedPipelineBase):
    """Native LTX-2 preprocessing pipeline (text/video with optional audio)."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        tokenizer = self.get_module("tokenizer")
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

    def _load_ltx2_audio_encoder(
            self) -> tuple[torch.nn.Module, AudioProcessor]:
        audio_vae_path = os.path.join(self.model_path, "audio_vae")
        if not os.path.isdir(audio_vae_path):
            raise FileNotFoundError(
                f"Expected audio_vae directory for LTX-2 audio preprocessing: {audio_vae_path}"
            )

        config = get_diffusers_config(model=audio_vae_path)
        audio_encoder = LTX2AudioEncoder(config).to(
            device=get_local_torch_device(),
            dtype=torch.float32,
        )

        safetensors_list = glob.glob(
            os.path.join(audio_vae_path, "*.safetensors"))
        loaded: dict[str, torch.Tensor] = {}
        for sf_file in safetensors_list:
            loaded.update(safetensors_load_file(sf_file))

        encoder_state = {}
        for name, tensor in loaded.items():
            if name.startswith("encoder."):
                encoder_state[name.replace("encoder.", "")] = tensor
            elif name.startswith("per_channel_statistics."):
                encoder_state[name] = tensor

        target_module = getattr(audio_encoder, "model", audio_encoder)
        missing, unexpected = target_module.load_state_dict(encoder_state,
                                                            strict=False)
        if missing:
            logger.warning("Missing LTX-2 audio encoder keys: %s", missing[:8])
        if unexpected:
            logger.warning("Unexpected LTX-2 audio encoder keys: %s",
                           unexpected[:8])
        target_module.eval()

        audio_processor = AudioProcessor(
            sample_rate=target_module.sample_rate,
            mel_bins=target_module.mel_bins,
            mel_hop_length=target_module.mel_hop_length,
            n_fft=target_module.n_fft,
        ).to(next(target_module.parameters()).device)
        return target_module, audio_processor

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        assert fastvideo_args.preprocess_config is not None

        preprocess_cfg = fastvideo_args.preprocess_config
        self.add_stage(
            stage_name="text_transform_stage",
            stage=TextTransformStage(
                cfg_uncondition_drop_rate=preprocess_cfg.training_cfg_rate,
                seed=preprocess_cfg.seed,
            ),
        )

        text_encoder = self.get_module("text_encoder")
        tokenizer = self.get_module("tokenizer")
        encoder_config = fastvideo_args.pipeline_config.text_encoder_configs[0]
        tokenizer_kwargs = dict(encoder_config.tokenizer_kwargs)
        if "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = encoder_config.arch_config.text_len
        self.add_stage(
            stage_name="prompt_precompute_stage",
            stage=LTX2TextPrecomputeStage(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                preprocess_text_fn=fastvideo_args.pipeline_config.
                preprocess_text_funcs[0],
                tokenizer_kwargs=tokenizer_kwargs,
                padding_side=encoder_config.arch_config.padding_side,
            ),
        )

        self.add_stage(
            stage_name="video_transform_stage",
            stage=VideoTransformStage(
                train_fps=preprocess_cfg.train_fps,
                num_frames=preprocess_cfg.num_frames,
                max_height=preprocess_cfg.max_height,
                max_width=preprocess_cfg.max_width,
                do_temporal_sample=preprocess_cfg.do_temporal_sample,
            ),
        )
        if preprocess_cfg.with_audio:
            audio_encoder, audio_processor = self._load_ltx2_audio_encoder()
            self.add_stage(
                stage_name="audio_encoding_stage",
                stage=LTX2AudioEncodingStage(
                    audio_encoder=audio_encoder,
                    audio_processor=audio_processor,
                    fallback_fps=preprocess_cfg.train_fps,
                ),
            )

        self.add_stage(
            stage_name="video_encoding_stage",
            stage=EncodingStage(vae=self.get_module("vae")),
        )


EntryClass = PreprocessPipelineT2V
