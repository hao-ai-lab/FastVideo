# SPDX-License-Identifier: Apache-2.0
"""Waypoint-1-Small streaming inference pipeline."""

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ComposedPipelineBase, ForwardBatch
from fastvideo.pipelines.stages.waypoint_stages import (
    WaypointDecodingStage,
    WaypointDenoisingStage,
    WaypointTextEncodingStage,
)
from fastvideo.utils import PRECISION_TO_TYPE


class WaypointPipeline(ComposedPipelineBase):
    """Stage-composed Waypoint interactive world model pipeline."""

    _required_config_modules = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
    ]

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        transformer = self.get_module("transformer")
        dtype = next(transformer.parameters()).dtype
        transformer.to(dtype=dtype)
        transformer.denoise_step_emb.to(dtype=torch.float32)

        vae = self.get_module("vae")
        vae_dtype = PRECISION_TO_TYPE[fastvideo_args.pipeline_config.vae_precision]
        vae.to(dtype=vae_dtype)

        self.add_stage(
            "prompt_encoding_stage",
            WaypointTextEncodingStage(
                self.get_module("text_encoder"),
                self.get_module("tokenizer"),
            ),
        )
        self.add_stage(
            "denoising_stage",
            WaypointDenoisingStage(transformer, vae, dtype),
        )
        self.add_stage("decoding_stage", WaypointDecodingStage(vae))

    @torch.no_grad()
    def streaming_reset(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> None:
        if not self.post_init_called:
            self.post_init()
        batch = self.prompt_encoding_stage(batch, fastvideo_args)
        self.denoising_stage.streaming_reset(batch, fastvideo_args)

    @torch.no_grad()
    def streaming_step(
        self,
        keyboard_action: torch.Tensor,
        mouse_action: torch.Tensor,
        scroll_action: torch.Tensor | None = None,
    ) -> ForwardBatch:
        batch = self.denoising_stage.streaming_step(
            keyboard_action,
            mouse_action,
            scroll_action,
        )
        return self.decoding_stage(batch, self.denoising_stage._require_context().fastvideo_args)

    def streaming_clear(self) -> None:
        self.denoising_stage.streaming_clear()


EntryClass = [WaypointPipeline]
