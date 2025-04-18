import asyncio
from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.fastvideo_args import FastVideoArgs
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
    def __init__(self, caption_client) -> None:
        super().__init__()
        self.caption_client = caption_client  # This should have a call_caption(prompts: List[str]) method.

    def forward(self, batch: ForwardBatch, fastvideo_args) -> ForwardBatch:
        # 1. Preprocess the prompt
        # Construct a list where the first entry is the prompt appended with the positive magic string.
        prompts = [batch.prompt + fastvideo_args.pos_magic]
        bs = len(prompts)
        # Then add the negative magic prompt repeated 'bs' times.
        prompts += [fastvideo_args.neg_magic] * bs

        # 2. Call the remote caption API asynchronously.
        # This mimics the v0 behavior using asyncio.run.
        data = asyncio.run(self.caption_client(prompts))
        
        # 3. Cast the returned tensors to the proper device.
        y = data['y']
        y_mask = data['y_mask']
        clip = data['clip_embedding']
        
        # split positive vs negative text
        batch.prompt_embeds          = y[:bs]          # [bs, seq_len, dim]
        batch.negative_prompt_embeds = y[bs:2*bs]      # [bs, seq_len, dim]
        batch.prompt_attention_mask  = y_mask[:bs]      # [bs, seq_len]
        batch.negative_attention_mask = y_mask[bs:2*bs] # [bs, seq_len]
        batch.clip_embedding = clip

        return batch