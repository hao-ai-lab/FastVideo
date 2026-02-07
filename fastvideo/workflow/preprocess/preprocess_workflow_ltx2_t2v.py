# SPDX-License-Identifier: Apache-2.0
"""LTX-2 preprocessing workflow writing native .precomputed training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fastvideo.configs.configs import PreprocessConfig
from fastvideo.distributed.parallel_state import get_world_rank
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch
from fastvideo.workflow.preprocess.components import (
    PreprocessingDataValidator, VideoForwardBatchBuilder, build_dataset)
from fastvideo.workflow.preprocess.preprocess_workflow import PreprocessWorkflow

if TYPE_CHECKING:
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase

logger = init_logger(__name__)


class LTX2PrecomputedSaver:
    """Save LTX-2 preprocessing outputs to .pt files under .precomputed/."""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.latents_dir = self.output_root / "latents"
        self.conditions_dir = self.output_root / "conditions"
        self.audio_latents_dir: Path | None = None
        self.latents_dir.mkdir(parents=True, exist_ok=True)
        self.conditions_dir.mkdir(parents=True, exist_ok=True)

    def _to_rel_pt_path(self, video_name: str) -> Path:
        return Path(video_name).with_suffix(".pt")

    def save_batch(self, batch: PreprocessBatch) -> None:
        assert isinstance(batch.latents, torch.Tensor)
        assert isinstance(batch.prompt_embeds, list) and len(
            batch.prompt_embeds) > 0
        assert isinstance(batch.prompt_attention_mask, list) and len(
            batch.prompt_attention_mask) > 0
        assert isinstance(batch.video_file_name, list)
        assert isinstance(batch.num_frames, list)
        assert isinstance(batch.height, list)
        assert isinstance(batch.width, list)
        assert isinstance(batch.fps, list)

        prompt_embeds = batch.prompt_embeds[0]
        prompt_attention_mask = batch.prompt_attention_mask[0]
        assert isinstance(prompt_embeds, torch.Tensor)
        assert isinstance(prompt_attention_mask, torch.Tensor)

        audio_latents: list[torch.Tensor | None] = batch.extra.get(
            "ltx2_audio_latents", [])

        for idx, video_name in enumerate(batch.video_file_name):
            rel_path = self._to_rel_pt_path(video_name)

            latent_output = self.latents_dir / rel_path
            latent_output.parent.mkdir(parents=True, exist_ok=True)
            latent = batch.latents[idx].detach().cpu().contiguous()
            latent_payload = {
                "latents": latent,
                "num_frames": int(batch.num_frames[idx]),
                "height": int(batch.height[idx]),
                "width": int(batch.width[idx]),
                "fps": float(batch.fps[idx]),
            }
            torch.save(latent_payload, latent_output)

            condition_output = self.conditions_dir / rel_path
            condition_output.parent.mkdir(parents=True, exist_ok=True)
            condition_payload = {
                "prompt_embeds":
                prompt_embeds[idx].detach().cpu().contiguous(),
                "prompt_attention_mask":
                prompt_attention_mask[idx].detach().cpu().contiguous(),
            }
            torch.save(condition_payload, condition_output)

            if idx >= len(audio_latents) or audio_latents[idx] is None:
                continue

            if self.audio_latents_dir is None:
                self.audio_latents_dir = self.output_root / "audio_latents"
                self.audio_latents_dir.mkdir(parents=True, exist_ok=True)
            assert self.audio_latents_dir is not None

            audio_latent = audio_latents[idx].detach().cpu().contiguous()
            audio_output = self.audio_latents_dir / rel_path
            audio_output.parent.mkdir(parents=True, exist_ok=True)
            audio_payload = {
                "latents": audio_latent,
                "num_time_steps": int(audio_latent.shape[1]),
                "frequency_bins": int(audio_latent.shape[2]),
                "duration":
                float(batch.num_frames[idx]) / float(batch.fps[idx]),
            }
            torch.save(audio_payload, audio_output)


class PreprocessWorkflowLTX2T2V(PreprocessWorkflow):
    """LTX-2 workflow for generating native precomputed training tensors."""

    training_dataloader: DataLoader
    preprocess_pipeline: ComposedPipelineBase
    video_forward_batch_builder: VideoForwardBatchBuilder
    precomputed_saver: LTX2PrecomputedSaver

    @staticmethod
    def _resolve_precomputed_output_dir(
            preprocess_cfg: PreprocessConfig) -> Path:
        output_dir = Path(
            preprocess_cfg.dataset_output_dir).expanduser().resolve()
        if output_dir.name != ".precomputed":
            output_dir = output_dir / ".precomputed"
        return output_dir

    def register_components(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        preprocess_cfg = self.fastvideo_args.preprocess_config

        raw_data_validator = PreprocessingDataValidator(
            max_height=preprocess_cfg.max_height,
            max_width=preprocess_cfg.max_width,
            num_frames=preprocess_cfg.num_frames,
            train_fps=preprocess_cfg.train_fps,
            speed_factor=preprocess_cfg.speed_factor,
            video_length_tolerance_range=preprocess_cfg.
            video_length_tolerance_range,
            drop_short_ratio=preprocess_cfg.drop_short_ratio,
        )
        self.add_component("raw_data_validator", raw_data_validator)

        training_dataset = build_dataset(preprocess_cfg,
                                         split="train",
                                         validator=raw_data_validator)
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=preprocess_cfg.preprocess_video_batch_size,
            num_workers=preprocess_cfg.dataloader_num_workers,
            collate_fn=lambda x: x,
        )
        self.add_component("training_dataloader", training_dataloader)

        video_forward_batch_builder = VideoForwardBatchBuilder(
            seed=preprocess_cfg.seed)
        self.add_component("video_forward_batch_builder",
                           video_forward_batch_builder)

        output_root = self._resolve_precomputed_output_dir(preprocess_cfg)
        precomputed_saver = LTX2PrecomputedSaver(output_root)
        self.add_component("precomputed_saver", precomputed_saver)

    def prepare_system_environment(self) -> None:
        assert self.fastvideo_args.preprocess_config is not None
        preprocess_cfg = self.fastvideo_args.preprocess_config
        output_root = self._resolve_precomputed_output_dir(preprocess_cfg)
        output_root.mkdir(parents=True, exist_ok=True)
        self.precomputed_output_dir = output_root
        logger.info("LTX-2 precomputed output directory: %s",
                    self.precomputed_output_dir)

    def run(self) -> None:
        total_samples = 0
        for batch in tqdm(self.training_dataloader,
                          desc="Preprocessing LTX-2 training dataset",
                          unit="batch"):
            forward_batch: PreprocessBatch = self.video_forward_batch_builder(
                batch)
            forward_batch = self.preprocess_pipeline.forward(
                forward_batch, self.fastvideo_args)
            self.precomputed_saver.save_batch(forward_batch)
            total_samples += len(forward_batch.video_file_name)
        logger.info(
            "Finished LTX-2 preprocessing on rank %s with %s samples written to %s",
            get_world_rank(),
            total_samples,
            self.precomputed_output_dir,
        )
