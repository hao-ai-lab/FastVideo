# SPDX-License-Identifier: Apache-2.0
"""
Prompt encoding stages for diffusion pipelines.

This module contains implementations of prompt encoding stages for diffusion pipelines.
"""

from typing import TypedDict

from fastvideo.v1.forward_context import set_forward_context
from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.logger import init_logger

from ..pipeline_batch_info import ForwardBatch
from .base import PipelineStage

logger = init_logger(__name__)
# TODO: this this hunyuan specific. Have an config for each model's special default hyperparameters
PROMPT_TEMPLATE_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. background environment, light, style and atmosphere."
    "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>")


class PromptTemplate(TypedDict):
    template: str
    crop_start: int


prompt_template_video: PromptTemplate = {
    "template": PROMPT_TEMPLATE_ENCODE_VIDEO,
    "crop_start": 95,
}


class LlamaEncodingStage(PipelineStage):
    """
    Stage for encoding text prompts into embeddings for diffusion models.
    
    This stage handles the encoding of text prompts into the embedding space
    expected by the diffusion model.
    """

    def __init__(self, text_encoder, tokenizer) -> None:
        """
        Initialize the prompt encoding stage.
        
        Args:
            enable_logging: Whether to enable logging for this stage.
            is_secondary: Whether this is a secondary text encoder.
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def forward(
        self,
        batch: ForwardBatch,
        inference_args: InferenceArgs,
    ) -> ForwardBatch:
        """
        Encode the prompt into text encoder hidden states.
        
        Args:
            batch: The current batch information.
            inference_args: The inference arguments.
            
        Returns:
            The batch with encoded prompt embeddings.
        """

        text = prompt_template_video["template"].format(batch.prompt)
        text_inputs = self.tokenizer(
            text,
            truncation=True,
            # better way to handle this?
            max_length=256,
            return_tensors="pt",
        )
        hidden_state_skip_layer = 2
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.text_encoder(
                input_ids=text_inputs["input_ids"].to(batch.device),
                output_hidden_states=hidden_state_skip_layer is not None,
            )

        last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer +
                                                    1)]
        crop_start = prompt_template_video.get("crop_start", -1)
        last_hidden_state = last_hidden_state[:, crop_start:]
        batch.prompt_embeds.append(last_hidden_state)

        return batch
