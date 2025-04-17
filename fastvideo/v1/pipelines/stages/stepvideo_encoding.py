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
        batch.prompt_embeds = data['y']
        batch.prompt_attention_mask = data['y_mask']
        batch.prompt_embeds_2 = data['clip_embedding']
        return batch