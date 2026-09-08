import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

class CosmosPredictTextEncoder(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig = None):
        super().__init__()
        if config is None:
            # Default fallback for testing
            config = Qwen2_5_VLConfig()
        
        # We instantiate the standard HF model used by Cosmos Predict
        self.model = Qwen2_5_VLForConditionalGeneration(config)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for text encoding in Cosmos Predict.
        Extracts hidden states from all layers (except the embedding layer 0),
        normalizes them, and concatenates them to form prompt_embeds.
        """
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)) / (
                hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8
            )
            normalized_hidden_states.append(normalized_state)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        return prompt_embeds
