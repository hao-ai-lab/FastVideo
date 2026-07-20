# SPDX-License-Identifier: Apache-2.0
"""Native split DFN5B CLIP encoders used by MMAudio.

MMAudio uses one OpenCLIP checkpoint in two different ways: normalized
projected image embeddings and normalized token-wise text hidden states. The
classes here reuse FastVideo's native CLIP transformer while exposing those
two contracts as independently loadable pipeline components.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention.backends.sdpa import SDPAMetadata
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.mmaudio_clip import (
    MMAudioDFNCLIPTextConfig,
    MMAudioDFNCLIPVisionConfig,
)
from fastvideo.models.encoders.clip import CLIPTextModel, CLIPVisionModel
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.weight_utils import default_weight_loader


class MMAudioDFNCLIPTextEncoder(CLIPTextModel):
    """Return normalized CLIP hidden states for every text token."""

    def __init__(self, config: MMAudioDFNCLIPTextConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        if input_ids is not None:
            sequence_length = input_ids.shape[-1]
            device = input_ids.device
        elif inputs_embeds is not None:
            sequence_length = inputs_embeds.shape[-2]
            device = inputs_embeds.device
        else:
            raise ValueError("MMAudio CLIP text encoding requires input_ids or inputs_embeds.")
        model_dtype = next(self.parameters()).dtype
        causal_mask = torch.empty(
            (sequence_length, sequence_length),
            device=device,
            dtype=model_dtype,
        )
        causal_mask.fill_(float("-inf"))
        causal_mask.triu_(1)
        metadata = SDPAMetadata(
            current_timestep=0,
            attn_mask=causal_mask[None, None],
        )
        with set_forward_context(current_timestep=0, attn_metadata=metadata):
            output = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                **kwargs,
            )
        assert output.last_hidden_state is not None
        normalized = F.normalize(output.last_hidden_state, dim=-1)
        return BaseEncoderOutput(
            last_hidden_state=normalized,
            pooler_output=output.pooler_output,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            attention_mask=output.attention_mask,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load either split OpenCLIP Q/K/V or converted fused QKV weights."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name in params:
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, tensor)
                loaded.add(name)
                continue
            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                target_name = name.replace(weight_name, param_name)
                if target_name not in params:
                    continue
                parameter = params[target_name]
                parameter.weight_loader(parameter, tensor, shard_id)
                loaded.add(target_name)
                break
        return loaded


class MMAudioDFNCLIPVisionEncoder(CLIPVisionModel):
    """Return normalized projected CLS embeddings for individual frames."""

    def __init__(self, config: MMAudioDFNCLIPVisionConfig) -> None:
        super().__init__(config)
        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

    def forward(
        self,
        pixel_values: torch.Tensor,
        feature_sample_layers: list[int] | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        del kwargs
        if feature_sample_layers is not None:
            raise ValueError("MMAudio DFN5B vision encoding requires the final CLS token")
        tokens = self.vision_model(pixel_values, feature_sample_layers=None)
        image_features = F.normalize(self.visual_projection(tokens[:, 0]), dim=-1)
        return BaseEncoderOutput(last_hidden_state=image_features, pooler_output=image_features)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)
        for name, tensor in weights:
            if name.startswith("vision_model.encoder.layers"):
                layer_index = int(name.split(".")[3])
                if layer_index >= layer_count:
                    continue
            # Converted checkpoints already contain fused qkv_proj tensors.
            # Check exact names before matching the ``v_proj`` substring that
            # also occurs at the end of ``qkv_proj``.
            if name in params:
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, tensor)
                loaded.add(name)
                continue
            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                target_name = name.replace(weight_name, param_name)
                if target_name not in params:
                    continue
                parameter = params[target_name]
                parameter.weight_loader(parameter, tensor, shard_id)
                loaded.add(target_name)
                break
            else:
                if name not in params:
                    continue
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, tensor)
                loaded.add(name)
        return loaded


EntryClass = [MMAudioDFNCLIPTextEncoder, MMAudioDFNCLIPVisionEncoder]
