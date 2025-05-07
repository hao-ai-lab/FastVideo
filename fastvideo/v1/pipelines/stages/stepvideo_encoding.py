import torch

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.v1.pipelines.stages.base import PipelineStage

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

        prompts = [batch.prompt + fastvideo_args.pos_magic]
        bs = len(prompts)
        prompts += [fastvideo_args.neg_magic] * bs
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
