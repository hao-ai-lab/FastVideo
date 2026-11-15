# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K conditioning: reuse a draft bundle's prompt embeddings instead of running the text encoder."""

from __future__ import annotations

from collections.abc import Callable

import torch

from fastvideo.distributed import get_local_torch_device
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
    MINIMAX_H3_TEXT_TOKEN_TAGS_KEY,
    MiniMaxH3ConditioningStage,
)
from fastvideo.pipelines.basic.minimax_h3.draft_bundle import (
    BUNDLE_EXTRA_KEY,
    bundle_prompt_conditioning,
    is_bundle_path,
    load_draft_bundle,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


class Regenerate2KConditioningStage(MiniMaxH3ConditioningStage):
    """MiniMax-H3 conditioning that takes the prompt embeddings from a draft bundle when it carries them.

    ``before_encode`` runs only once the parent has missed its prompt-embeds cache and the text encoder is actually
    needed; the pipeline uses it to drop a DiT kept resident across requests, so the encoder and the DiT are never in
    unified memory together.
    """

    before_encode: Callable[[], None] | None = None
    # Set by the pipeline: how to treat a bundle's prompt embeddings, and our own text-encoder fingerprint.
    prompt_embeds_policy: str = "match"
    text_encoder_fingerprint: Callable[[], str | None] | None = None

    def _before_text_encoder(self) -> None:
        if self.before_encode is not None:
            self.before_encode()

    @torch.no_grad()
    def forward(self, batch: ForwardBatch, fastvideo_args: FastVideoArgs) -> ForwardBatch:
        if is_bundle_path(batch.video_path):
            bundle = load_draft_bundle(str(batch.video_path))
            if bundle is not None:
                batch.extra[BUNDLE_EXTRA_KEY] = bundle
                ours = (self.text_encoder_fingerprint()
                        if self.prompt_embeds_policy == "match" and self.text_encoder_fingerprint is not None else None)
                conditioning = bundle_prompt_conditioning(bundle,
                                                          policy=self.prompt_embeds_policy,
                                                          text_encoder=ours,
                                                          prompt=batch.prompt)
                if conditioning is not None:
                    embeds, tags = conditioning
                    batch.prompt_embeds = [embeds.to(device=get_local_torch_device())]
                    batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = tags
                    logger.info("Open-H3-Regenerate-2K: prompt embeddings from the draft bundle; text encoder skipped")
                    return batch
                logger.info(
                    "Open-H3-Regenerate-2K: re-encoding the prompt (draft bundle embeddings not reused, policy=%s)",
                    self.prompt_embeds_policy)
        return super().forward(batch, fastvideo_args)


__all__ = ["Regenerate2KConditioningStage"]
