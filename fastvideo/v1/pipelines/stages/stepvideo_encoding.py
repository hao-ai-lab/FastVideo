# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import torch

from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage
from fastvideo.v1.pipelines.stages.validators import StageValidators as V

logger = init_logger(__name__)


# The dedicated stepvideo prompt encoding stage.
class StepvideoPromptEncodingStage(PipelineStage):
    """
    Stage for encoding prompts using the remote caption API.
    
    This stage applies the magic string transformations and calls
    the remote caption service asynchronously to get:
      - primary prompt embeddings,
      - an attention mask,
      - and a clip embedding.
    """

    def __init__(self, stepllm, clip) -> None:
        super().__init__()
        # self.caption_client = caption_client  # This should have a call_caption(prompts: List[str]) method.
        self.stepllm = stepllm
        self.clip = clip

    def forward(self, batch: ForwardBatch, fastvideo_args) -> ForwardBatch:

        prompts = [batch.prompt + fastvideo_args.pipeline_config.pos_magic]
        bs = len(prompts)
        prompts += [fastvideo_args.pipeline_config.neg_magic] * bs
        with set_forward_context(current_timestep=0, attn_metadata=None):
            y, y_mask = self.stepllm(prompts)
            clip_emb, _ = self.clip(prompts)
            len_clip = clip_emb.shape[1]
            y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)
        pos_clip, neg_clip = clip_emb[:bs], clip_emb[bs:]

        # split positive vs negative text
        batch.prompt_embeds = y[:bs]  # [bs, seq_len, dim]
        batch.negative_prompt_embeds = y[bs:2 * bs]  # [bs, seq_len, dim]
        batch.prompt_attention_mask = y_mask[:bs]  # [bs, seq_len]
        batch.negative_attention_mask = y_mask[bs:2 * bs]  # [bs, seq_len]
        batch.clip_embedding_pos = pos_clip
        batch.clip_embedding_neg = neg_clip
        return batch

    def verify_input(self, batch: ForwardBatch,
                     fastvideo_args: FastVideoArgs) -> Dict[str, bool]:
        """Verify stepvideo encoding stage inputs."""
        return {
            # Text prompt for processing
            "prompt": V.string_not_empty(batch.prompt),
        }

    def verify_output(self, batch: ForwardBatch,
                      fastvideo_args: FastVideoArgs) -> Dict[str, bool]:
        """Verify stepvideo encoding stage outputs."""
        return {
            # Positive text embeddings: [batch_size, seq_len, hidden_dim]
            "prompt_embeds":
            V.is_tensor(batch.prompt_embeds)
            and V.tensor_with_dims(batch.prompt_embeds, 3),
            # Negative text embeddings: [batch_size, seq_len, hidden_dim]
            "negative_prompt_embeds":
            V.is_tensor(batch.negative_prompt_embeds)
            and V.tensor_with_dims(batch.negative_prompt_embeds, 3),
            # Attention masks: [batch_size, seq_len]
            "prompt_attention_mask":
            V.is_tensor(batch.prompt_attention_mask)
            and V.tensor_with_dims(batch.prompt_attention_mask, 2),
            "negative_attention_mask":
            V.is_tensor(batch.negative_attention_mask)
            and V.tensor_with_dims(batch.negative_attention_mask, 2),
            # CLIP embeddings: [batch_size, hidden_dim]
            "clip_embedding_pos":
            V.is_tensor(batch.clip_embedding_pos)
            and V.tensor_with_dims(batch.clip_embedding_pos, 2),
            "clip_embedding_neg":
            V.is_tensor(batch.clip_embedding_neg)
            and V.tensor_with_dims(batch.clip_embedding_neg, 2),
        }
