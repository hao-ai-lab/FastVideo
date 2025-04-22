# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
from typing import Dict, Optional

import torch
from diffusers.configuration_utils import register_to_config
from einops import rearrange, repeat
from torch import nn

# import everything from v1 later (reimplement)
from fastvideo.v1.layers.visual_embedding import (
                                                  PatchEmbed,
                                                  )
from fastvideo.v1.layers.layernorm import (LayerNormScaleShift, RMSNorm)
from fastvideo.v1.layers.linear import ReplicatedLinear
from fastvideo.v1.layers.mlp import MLP
from fastvideo.v1.models.dits.base import BaseDiT
# from fastvideo.models.stepvideo.modules.blocks import StepVideoTransformerBlock
# from fastvideo.models.stepvideo.modules.normalization import AdaLayerNormSingle, PixArtAlphaTextProjection
# from fastvideo.models.stepvideo.parallel import parallel_forward
# from fastvideo.models.stepvideo.utils import with_empty_init
from fastvideo.v1.models.dits.temp import StepVideoTransformerBlock, AdaLayerNormSingle, PixArtAlphaTextProjection, with_empty_init

from fastvideo.v1.platforms import _Backend

class StepVideoModel(BaseDiT):
    # (Optional) Keep the same attribute for compatibility with splitting, etc.
    _fsdp_shard_conditions = [
        lambda n, m: "transformer_blocks" in n and n.split(".")[-1].isdigit(),
        lambda n, m: "pos_embed" in n  # If needed for the patch embedding.
    ]
    _param_names_mapping = {
        r"^transformer_blocks\.(\d+)\.norm1\.(weight|bias)$":
        r"transformer_blocks.\1.norm1.norm.\2",
        r"^transformer_blocks\.(\d+)\.norm2\.(weight|bias)$":
        r"transformer_blocks.\1.norm2.norm.\2",

        r"^adaln_single\.emb\.timestep_embedder\.linear_1\.(weight|bias)$":
            r"adaln_single.emb.mlp.fc_in.\1",
        r"^adaln_single\.emb\.timestep_embedder\.linear_2\.(weight|bias)$":
            r"adaln_single.emb.mlp.fc_out.\1",

        r"^caption_projection\.linear_1\.(weight|bias)$":
            r"caption_projection.fc_in.\1",
        r"^caption_projection\.linear_2\.(weight|bias)$":
            r"caption_projection.fc_out.\1",
    }
    _supported_attention_backends = [
        _Backend.FLASH_ATTN, _Backend.TORCH_SDPA
    ]
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 128,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 48,
        dropout: float = 0.0,
        patch_size: int = 1,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        use_additional_conditions: Optional[bool] = False,
        caption_channels: Optional[int] | list | tuple = [6144, 1024],
        attention_type: Optional[str] = "torch",
    ):
        super().__init__()
        # Instead of using self.config, assign each parameter as an instance variable.
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.patch_size = patch_size
        self.norm_type = norm_type
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.use_additional_conditions = use_additional_conditions
        self.caption_channels = caption_channels
        self.attention_type = attention_type

        # Compute inner dimension.
        self.hidden_size = self.num_attention_heads * self.attention_head_dim

        # Image/video patch embedding.
        self.pos_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=self.in_channels,
            embed_dim=self.hidden_size,
        )

        # Transformer blocks.
        self.transformer_blocks = nn.ModuleList([
            StepVideoTransformerBlock(
                dim=self.hidden_size,
                attention_head_dim=self.attention_head_dim,
                attention_type=attention_type
            )
            for _ in range(self.num_layers)
        ])

        # Output blocks.
        # self.norm_out = nn.LayerNorm(self.hidden_size, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.norm_out = LayerNormScaleShift(self.hidden_size, norm_type="layer", eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.scale_shift_table = nn.Parameter(torch.randn(2, self.hidden_size) / (self.hidden_size ** 0.5))
        # self.proj_out = nn.Linear(self.hidden_size, patch_size * patch_size * self.out_channels)
        self.proj_out = ReplicatedLinear(self.hidden_size, patch_size * patch_size * self.out_channels)
        # Time modulation via adaptive layer norm.
        self.adaln_single = AdaLayerNormSingle(self.hidden_size)

        # Set up caption conditioning.
        if isinstance(self.caption_channels, int):
            caption_channel = self.caption_channels
        else:
            caption_channel, clip_channel = self.caption_channels
            # self.clip_projection = nn.Linear(clip_channel, self.hidden_size)
            self.clip_projection = ReplicatedLinear(clip_channel, self.hidden_size)
        self.caption_norm = nn.LayerNorm(caption_channel, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        # self.caption_norm = LayerNormScaleShift(caption_channel, norm_type="layer", eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.caption_projection = MLP(input_dim=caption_channel, mlp_hidden_dim=self.hidden_size, act_type="gelu_pytorch_tanh")

        # Flag to indicate if using parallel attention.
        self.parallel = (attention_type == "parallel")

    def patchfy(self, hidden_states):
        hidden_states = rearrange(hidden_states, 'b f c h w -> (b f) c h w')
        hidden_states = self.pos_embed(hidden_states)
        return hidden_states

    def prepare_attn_mask(self, encoder_attention_mask, encoder_hidden_states, q_seqlen):
        kv_seqlens = encoder_attention_mask.sum(dim=1).int()
        mask = torch.zeros([len(kv_seqlens), q_seqlen, max(kv_seqlens)],
                           dtype=torch.bool,
                           device=encoder_attention_mask.device)
        encoder_hidden_states = encoder_hidden_states[:, :max(kv_seqlens)]
        for i, kv_len in enumerate(kv_seqlens):
            mask[i, :, :kv_len] = 1
        return encoder_hidden_states, mask

    def block_forward(self,
                      hidden_states,
                      encoder_hidden_states=None,
                      t_expand=None,
                      rope_positions=None,
                      attn_mask=None,
                      parallel=True,
                      mask_strategy=None):

        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states,
                                  encoder_hidden_states,
                                  t_expand=t_expand,
                                  attn_mask=attn_mask,
                                  rope_positions=rope_positions,
                                  mask_strategy=mask_strategy[i])

        return hidden_states

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        t_expand: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        fps: torch.Tensor = None,
        return_dict: bool = True,
        mask_strategy=None,
        guidance=None,
    ):
        assert hidden_states.ndim == 5
        "hidden_states's shape should be (bsz, f, ch, h ,w)"
        frame= hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, 'b c f h w -> b f c h w', f=frame)
        if mask_strategy == None:
            mask_strategy = [None, None]
        bsz, frame, _, height, width = hidden_states.shape
        height, width = height // self.patch_size, width // self.patch_size
        
        hidden_states = self.patchfy(hidden_states)
        len_frame = hidden_states.shape[1]

        t_expand, embedded_timestep = self.adaln_single(t_expand)
        encoder_hidden_states = self.caption_projection(self.caption_norm(encoder_hidden_states))

        
        if encoder_hidden_states_2 is not None and hasattr(self, 'clip_projection'):
            clip_embedding, _ = self.clip_projection(encoder_hidden_states_2)
            encoder_hidden_states = torch.cat([clip_embedding, encoder_hidden_states], dim=1)

        hidden_states = rearrange(hidden_states, '(b f) l d->  b (f l) d', b=bsz, f=frame, l=len_frame).contiguous()
        encoder_hidden_states, attn_mask = self.prepare_attn_mask(encoder_attention_mask,
                                                                  encoder_hidden_states,
                                                                  q_seqlen=frame * len_frame)

        hidden_states = self.block_forward(hidden_states,
                                           encoder_hidden_states,
                                           t_expand=t_expand,
                                           rope_positions=[frame, height, width],
                                           attn_mask=attn_mask,
                                           parallel=self.parallel,
                                           mask_strategy=mask_strategy)
        # print(">>> after block_forward:", hidden_states.shape)
        
        hidden_states = rearrange(hidden_states, 'b (f l) d -> (b f) l d', b=bsz, f=frame, l=len_frame)

        embedded_timestep = repeat(embedded_timestep, 'b d -> (b f) d', f=frame).contiguous()

        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states, shift=shift.squeeze(1), scale=scale.squeeze(1))
        # Modulation
        hidden_states, _ = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size,
                                                     self.out_channels))

        hidden_states = rearrange(hidden_states, 'n h w p q c -> n c h p w q')
        output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))

        output = rearrange(output, '(b f) c h w -> b c f h w', f=frame)
        # if return_dict:
        #     return {'x': output}
        return output
