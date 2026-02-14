# SPDX-License-Identifier: Apache-2.0
"""Pipeline stages for Waypoint-1-Small streaming world model.

These stages support the checklist requirement to reuse/create stages.
Waypoint uses a streaming interface; these stages encapsulate the logic
for text encoding and frame generation.
"""

import os

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.base import PipelineStage

_WAYPOINT_DEBUG = os.environ.get("WAYPOINT_DEBUG", "0") in ("1", "true", "yes")


class WaypointTextEncodingStage(PipelineStage):
    """Encode text prompt for Waypoint world model.

    Produces prompt_emb [B, P, D] and prompt_pad_mask [B, P] in
    batch.extra["waypoint_prompt_emb"] and batch.extra["waypoint_prompt_pad_mask"].
    """

    def __init__(self, text_encoder, tokenizer):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        """Encode prompt and store in batch.extra for Waypoint pipeline."""
        if batch.prompt is None:
            return batch

        text_encoder = self.text_encoder
        tokenizer = self.tokenizer
        max_length = getattr(getattr(text_encoder, "config", None), "text_len",
                             512)
        text_inputs = tokenizer(
            batch.prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        enc_device = next(text_encoder.parameters()).device
        input_ids = text_inputs.input_ids.to(enc_device)
        attention_mask = text_inputs.attention_mask.to(enc_device)

        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        prompt_emb = outputs.last_hidden_state
        prompt_emb = prompt_emb * attention_mask.unsqueeze(-1).to(
            prompt_emb.dtype)
        prompt_pad_mask = attention_mask.eq(0)

        if batch.extra is None:
            batch.extra = {}
        batch.extra["waypoint_prompt_emb"] = prompt_emb
        batch.extra["waypoint_prompt_pad_mask"] = prompt_pad_mask

        if _WAYPOINT_DEBUG:
            log = init_logger(__name__)
            f = prompt_emb.float()
            log.info(
                "DEBUG [text_stage] input_ids shape=%s prompt_emb: mean=%.6f "
                "std=%.6f min=%.6f max=%.6f pad_mask_sum=%d",
                tuple(input_ids.shape),
                f.mean().item(),
                f.std().item(),
                f.min().item(),
                f.max().item(),
                int(prompt_pad_mask.sum().item()),
            )
        return batch
