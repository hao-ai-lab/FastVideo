from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.utils import ModelOutput
from fastvideo.v1.logger import init_logger
from fastvideo.v1.forward_context import set_forward_context

logger = init_logger(__name__)

def use_default(value, default):
    return value if value is not None else default

@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):

    def __init__(
        self,
        text_encoder,
        tokenizer,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        device=None,
    ):
        super().__init__()
        # TODO(will): check if there's a cleaner way to do this
        self.text_encoder_type = text_encoder.config.architectures[0]
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (use_attention_mask is True), "Attention mask is True required when training videos."
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm


        if "T5" in self.text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "CLIPTextModel" in self.text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "LlamaModel" in self.text_encoder_type or "glm" in self.text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {self.text_encoder_type}")
        
        self.model = text_encoder
        # self.dtype = self.model.dtype
        self.device = device

        self.tokenizer = tokenizer

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If True, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        if self.prompt_template_video is not None:
            prompt_template = self.prompt_template_video["template"]
        
            text = self.apply_text_to_template(text, prompt_template)


        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return self.tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
            **kwargs,
        )

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        hidden_state_skip_layer=None,
        device=None,
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        attention_mask = (batch_encoding["attention_mask"].to(device) if use_attention_mask else None)

        # note: clip will need attention mask
        # TODO(will): unify interface with dit
        with set_forward_context(current_timestep=0, attn_metadata=None):
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=hidden_state_skip_layer is not None,
            )
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if self.prompt_template_video is not None:
        
            crop_start = self.prompt_template_video.get("crop_start", -1)

            last_hidden_state = last_hidden_state[:, crop_start:]
            attention_mask = (attention_mask[:, crop_start:] if use_attention_mask else None)
            total_length = attention_mask.sum()
            last_hidden_state = last_hidden_state[:, :total_length]
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
