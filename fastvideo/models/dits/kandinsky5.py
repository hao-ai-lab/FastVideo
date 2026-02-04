# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from diffusers.models.transformers import (
    Kandinsky5Transformer3DModel as DiffusersKandinsky5Transformer3DModel)
from diffusers.models.transformers.transformer_kandinsky import get_freqs

from fastvideo.configs.models.dits import Kandinsky5VideoConfig
from fastvideo.models.dits.base import BaseDiT


class Kandinsky5Transformer3DModel(BaseDiT):
    _fsdp_shard_conditions = Kandinsky5VideoConfig()._fsdp_shard_conditions
    _compile_conditions = Kandinsky5VideoConfig()._compile_conditions
    param_names_mapping = Kandinsky5VideoConfig().param_names_mapping
    reverse_param_names_mapping = Kandinsky5VideoConfig(
    ).reverse_param_names_mapping
    lora_param_names_mapping = Kandinsky5VideoConfig().lora_param_names_mapping
    _supported_attention_backends = Kandinsky5VideoConfig(
    )._supported_attention_backends

    def __init__(self, config: Kandinsky5VideoConfig,
                 hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config

        self.model = DiffusersKandinsky5Transformer3DModel(
            in_visual_dim=arch.in_visual_dim,
            in_text_dim=arch.in_text_dim,
            in_text_dim2=arch.in_text_dim2,
            time_dim=arch.time_dim,
            out_visual_dim=arch.out_visual_dim,
            patch_size=arch.patch_size,
            model_dim=arch.model_dim,
            ff_dim=arch.ff_dim,
            num_text_blocks=arch.num_text_blocks,
            num_visual_blocks=arch.num_visual_blocks,
            axes_dims=arch.axes_dims,
            visual_cond=arch.visual_cond,
            attention_type=arch.attention_type,
            attention_causal=arch.attention_causal,
            attention_local=arch.attention_local,
            attention_glob=arch.attention_glob,
            attention_window=arch.attention_window,
            attention_P=arch.attention_P,
            attention_wT=arch.attention_wT,
            attention_wW=arch.attention_wW,
            attention_wH=arch.attention_wH,
            attention_add_sta=arch.attention_add_sta,
            attention_method=arch.attention_method,
        )

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
        | None = None,
        guidance=None,
        pooled_projections: torch.Tensor | None = None,
        visual_rope_pos: tuple[torch.Tensor, torch.Tensor,
                               torch.Tensor] | list[torch.Tensor] | None = None,
        text_rope_pos: torch.Tensor | None = None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]

        if pooled_projections is None:
            if encoder_hidden_states_image is None:
                raise ValueError(
                    "pooled_projections must be provided for Kandinsky5.")
            pooled_projections = encoder_hidden_states_image
        if isinstance(pooled_projections, list):
            pooled_projections = pooled_projections[0]

        if visual_rope_pos is None or text_rope_pos is None:
            raise ValueError(
                "visual_rope_pos and text_rope_pos are required for Kandinsky5."
            )

        outputs = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=scale_factor,
            sparse_params=sparse_params,
            return_dict=return_dict,
        )

        if return_dict:
            return outputs
        return outputs.sample if hasattr(outputs, "sample") else outputs

    def materialize_non_persistent_buffers(self, device: torch.device,
                                           dtype: torch.dtype | None = None
                                           ) -> None:
        time_embeddings = self.model.time_embeddings
        if isinstance(time_embeddings.freqs,
                      torch.Tensor) and time_embeddings.freqs.is_meta:
            time_embeddings.freqs = get_freqs(
                time_embeddings.model_dim // 2,
                time_embeddings.max_period,
            )

        text_rope = self.model.text_rope_embeddings
        if isinstance(text_rope.args,
                      torch.Tensor) and text_rope.args.is_meta:
            freq = get_freqs(text_rope.dim // 2,
                             text_rope.max_period).to(device=device)
            pos = torch.arange(text_rope.max_pos,
                               dtype=freq.dtype,
                               device=device)
            text_rope._buffers["args"] = torch.outer(pos, freq)

        visual_rope = self.model.visual_rope_embeddings
        for i, (axes_dim,
                ax_max_pos) in enumerate(zip(visual_rope.axes_dims,
                                             visual_rope.max_pos)):
            name = f"args_{i}"
            buf = getattr(visual_rope, name, None)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = get_freqs(axes_dim // 2,
                                 visual_rope.max_period).to(device=device)
                pos = torch.arange(ax_max_pos,
                                   dtype=freq.dtype,
                                   device=device)
                visual_rope._buffers[name] = torch.outer(pos, freq)
