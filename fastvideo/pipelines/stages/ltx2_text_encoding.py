# SPDX-License-Identifier: Apache-2.0
"""
LTX2-specific text encoding stage with sequence parallelism broadcast support.

When running with sequence parallelism (SP), the Gemma text encoder is only
executed on rank 0, and the embeddings are broadcast to all other ranks.
This avoids I/O contention from all ranks loading the Gemma model simultaneously.
"""

import torch

from fastvideo.distributed.parallel_state import get_sp_group
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.stages.text_encoding import TextEncodingStage

logger = init_logger(__name__)


class LTX2TextEncodingStage(TextEncodingStage):
    """
    LTX2 text encoding stage with sequence parallelism support.
    
    When SP is enabled (sp_world_size > 1), only rank 0 runs the text encoder
    and broadcasts embeddings to other ranks. This avoids I/O contention from
    all ranks loading the Gemma model simultaneously, which can cause text
    encoding to take 100+ seconds instead of ~5 seconds.
    """

    @torch.no_grad()
    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> ForwardBatch:
        sp_group = get_sp_group()
        sp_world_size = sp_group.world_size
        sp_rank = sp_group.rank_in_group

        # Single GPU or no SP: use parent implementation
        if sp_world_size <= 1:
            return super().forward(batch, fastvideo_args)

        # SP enabled: only rank 0 encodes, then broadcasts
        if sp_rank == 0:
            logger.info(
                "[LTX2TextEncodingStage] SP rank 0: running text encoding")
            # Run encoding on rank 0
            result_batch = super().forward(batch, fastvideo_args)

            # Build broadcast dict from batch
            broadcast_dict = self._build_broadcast_dict(result_batch)

            # Broadcast to other ranks
            logger.info(
                "[LTX2TextEncodingStage] SP rank 0: broadcasting %d tensors",
                len(broadcast_dict))
            sp_group.broadcast_tensor_dict(broadcast_dict, src=0)

            return result_batch
        else:
            logger.info(
                "[LTX2TextEncodingStage] SP rank %d: receiving broadcast",
                sp_rank)
            # Other ranks: receive broadcast and populate batch
            broadcast_dict = sp_group.broadcast_tensor_dict(None, src=0)

            # Unpack into batch
            self._unpack_broadcast_to_batch(batch, broadcast_dict)

            logger.info(
                "[LTX2TextEncodingStage] SP rank %d: received %d prompt embeds",
                sp_rank, len(batch.prompt_embeds))

            return batch

    def _build_broadcast_dict(self,
                              batch: ForwardBatch) -> dict[str, torch.Tensor]:
        """Build dict of tensors to broadcast from rank 0."""
        d: dict[str, torch.Tensor] = {}

        # Use a dummy tensor for metadata since broadcast_tensor_dict expects tensors
        device = batch.prompt_embeds[0].device if batch.prompt_embeds else "cuda"

        # Prompt embeddings
        num_prompt_embeds = len(batch.prompt_embeds)
        d["_num_prompt_embeds"] = torch.tensor([num_prompt_embeds],
                                               device=device)
        for i, pe in enumerate(batch.prompt_embeds):
            d[f"prompt_embed_{i}"] = pe

        # Prompt attention masks
        has_prompt_masks = (batch.prompt_attention_mask is not None
                            and len(batch.prompt_attention_mask) > 0)
        d["_has_prompt_masks"] = torch.tensor([1 if has_prompt_masks else 0],
                                              device=device)
        if has_prompt_masks:
            for i, pm in enumerate(batch.prompt_attention_mask):
                d[f"prompt_mask_{i}"] = pm

        # Negative embeddings (CFG)
        has_neg_embeds = (batch.negative_prompt_embeds is not None
                          and len(batch.negative_prompt_embeds) > 0)
        d["_has_neg_embeds"] = torch.tensor([1 if has_neg_embeds else 0],
                                            device=device)
        if has_neg_embeds:
            d["_num_neg_embeds"] = torch.tensor(
                [len(batch.negative_prompt_embeds)], device=device)
            for i, ne in enumerate(batch.negative_prompt_embeds):
                d[f"neg_embed_{i}"] = ne

            # Negative attention masks
            has_neg_masks = (batch.negative_attention_mask is not None
                             and len(batch.negative_attention_mask) > 0)
            d["_has_neg_masks"] = torch.tensor([1 if has_neg_masks else 0],
                                               device=device)
            if has_neg_masks:
                for i, nm in enumerate(batch.negative_attention_mask):
                    d[f"neg_mask_{i}"] = nm

        # LTX2 audio embeddings
        has_audio_embeds = "ltx2_audio_prompt_embeds" in batch.extra
        d["_has_audio_embeds"] = torch.tensor([1 if has_audio_embeds else 0],
                                              device=device)
        if has_audio_embeds:
            audio_embeds = batch.extra["ltx2_audio_prompt_embeds"]
            d["_num_audio_embeds"] = torch.tensor([len(audio_embeds)],
                                                  device=device)
            for i, ae in enumerate(audio_embeds):
                d[f"audio_embed_{i}"] = ae

        # LTX2 audio negative embeddings
        has_audio_neg = "ltx2_audio_negative_embeds" in batch.extra
        d["_has_audio_neg"] = torch.tensor([1 if has_audio_neg else 0],
                                           device=device)
        if has_audio_neg:
            audio_neg = batch.extra["ltx2_audio_negative_embeds"]
            for i, audio_neg_embed in enumerate(audio_neg):
                d[f"audio_neg_embed_{i}"] = audio_neg_embed

        return d

    def _unpack_broadcast_to_batch(self, batch: ForwardBatch,
                                   d: dict[str, torch.Tensor]) -> None:
        """Unpack broadcast dict into batch on non-rank-0 processes."""
        # Prompt embeddings
        num_embeds = int(d["_num_prompt_embeds"].item())
        for i in range(num_embeds):
            batch.prompt_embeds.append(d[f"prompt_embed_{i}"])

        # Prompt attention masks
        has_prompt_masks = int(d["_has_prompt_masks"].item()) == 1
        if has_prompt_masks and batch.prompt_attention_mask is not None:
            for i in range(num_embeds):
                batch.prompt_attention_mask.append(d[f"prompt_mask_{i}"])

        # Negative embeddings (CFG)
        has_neg_embeds = int(d["_has_neg_embeds"].item()) == 1
        if has_neg_embeds:
            num_neg = int(d["_num_neg_embeds"].item())
            if batch.negative_prompt_embeds is not None:
                for i in range(num_neg):
                    batch.negative_prompt_embeds.append(d[f"neg_embed_{i}"])

            # Negative attention masks
            has_neg_masks = int(
                d.get("_has_neg_masks", torch.tensor([0])).item()) == 1
            if has_neg_masks and batch.negative_attention_mask is not None:
                for i in range(num_neg):
                    batch.negative_attention_mask.append(d[f"neg_mask_{i}"])

        # LTX2 audio embeddings
        has_audio_embeds = int(d["_has_audio_embeds"].item()) == 1
        if has_audio_embeds:
            num_audio = int(d["_num_audio_embeds"].item())
            audio_embeds = [d[f"audio_embed_{i}"] for i in range(num_audio)]
            batch.extra["ltx2_audio_prompt_embeds"] = audio_embeds

        # LTX2 audio negative embeddings
        has_audio_neg = int(d["_has_audio_neg"].item()) == 1
        if has_audio_neg:
            # Use same count as audio embeds
            num_audio = int(d["_num_audio_embeds"].item())
            audio_neg = [d[f"audio_neg_embed_{i}"] for i in range(num_audio)]
            batch.extra["ltx2_audio_negative_embeds"] = audio_neg
