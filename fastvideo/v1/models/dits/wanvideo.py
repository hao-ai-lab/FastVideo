# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union, Dict, Any
from fastvideo.v1.attention import DistributedAttention, LocalAttention
from fastvideo.v1.layers.linear import ReplicatedLinear
from fastvideo.v1.layers.layernorm import LayerNormScaleShift, ScaleResidual, ScaleResidualLayerNormScaleShift, RMSNorm
from fastvideo.v1.layers.visual_embedding import PatchEmbed, TimestepEmbedder, ModulateProjection
from fastvideo.v1.layers.rotary_embedding import _apply_rotary_emb, get_rotary_pos_embed
from fastvideo.v1.distributed.parallel_state import get_sequence_model_parallel_world_size
# from torch.nn import RMSNorm
# TODO: RMSNorm ....
from fastvideo.v1.layers.mlp import MLP
from .base import BaseDiT


class WanImageEmbedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_features)
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        self.norm2 = nn.LayerNorm(out_features)

    def forward(self,
                encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.time_modulation = ModulateProjection(dim,
                                                  factor=6,
                                                  act_layer="silu")
        self.text_embedder = MLP(text_embed_dim,
                                 dim,
                                 dim,
                                 bias=True,
                                 act_type="gelu_pytorch_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        with torch.cuda.amp.autocast(dtype=torch.float32):
            temb = self.time_embedder(timestep.float())
            timestep_proj = self.time_modulation(temb)

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 parallel_attention=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.parallel_attention = parallel_attention

        # layers
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # Scaled dot product attention
        self.attn = LocalAttention(dropout_rate=0,
                                   softmax_scale=None,
                                   causal=False)

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        pass


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q.forward_native(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k.forward_native(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.add_k_proj = ReplicatedLinear(dim, dim)
        self.add_v_proj = ReplicatedLinear(dim, dim)
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q.forward_native(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k.forward_native(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        k_img = self.norm_added_k.forward_native(
            self.add_k_proj(context_img)[0]).view(b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        img_x = self.attn(q, k_img, v_img)
        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x, _ = self.to_out(x)
        return x


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(num_heads=num_heads,
                                          head_size=dim // num_heads,
                                          dropout_rate=0.0,
                                          causal=False)
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim, eps=eps)
            self.norm_k = RMSNorm(dim, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception
        assert cross_attn_norm is True
        self.self_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=True,
            dtype=torch.float32)

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim,
                                              num_heads,
                                              qk_norm=qk_norm,
                                              eps=eps)
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32)

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        assert temb.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.scale_shift_table + temb
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
                6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = self.norm1(hidden_states.float()).to(
            dtype=orig_dtype) * (1 + scale_msa) + shift_msa
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q.forward_native(query)
        if self.norm_k is not None:
            key = self.norm_k.forward_native(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        # Apply rotary embeddings
        cos, sin = freqs_cis
        query, key = _apply_rotary_emb(query, cos, sin,
                                       is_neox_style=False), _apply_rotary_emb(
                                           key, cos, sin, is_neox_style=False)

        attn_output, _ = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)

        # 2. Cross-attention
        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)

        return hidden_states


class WanTransformer3DModel(BaseDiT):
    _fsdp_shard_conditions = [
        lambda n, m: "blocks" in n and str.isdigit(n.split(".")[-1]),
    ]
    _param_names_mapping = {
        r"^patch_embedding\.(.*)$":
        r"patch_embedding.proj.\1",
        r"^condition_embedder\.text_embedder\.linear_1\.(.*)$":
        r"condition_embedder.text_embedder.fc_in.\1",
        r"^condition_embedder\.text_embedder\.linear_2\.(.*)$":
        r"condition_embedder.text_embedder.fc_out.\1",
        r"^condition_embedder\.time_embedder\.linear_1\.(.*)$":
        r"condition_embedder.time_embedder.mlp.fc_in.\1",
        r"^condition_embedder\.time_embedder\.linear_2\.(.*)$":
        r"condition_embedder.time_embedder.mlp.fc_out.\1",
        r"^condition_embedder\.time_proj\.(.*)$":
        r"condition_embedder.time_modulation.linear.\1",
        r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$":
        r"condition_embedder.image_embedder.ff.fc_in.\1",
        r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$":
        r"condition_embedder.image_embedder.ff.fc_out.\1",
        r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$":
        r"blocks.\1.to_q.\2",
        r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$":
        r"blocks.\1.to_k.\2",
        r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$":
        r"blocks.\1.to_v.\2",
        r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$":
        r"blocks.\1.to_out.\2",
        r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$":
        r"blocks.\1.norm_q.\2",
        r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$":
        r"blocks.\1.norm_k.\2",
        r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$":
        r"blocks.\1.attn2.to_out.\2",
        r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$":
        r"blocks.\1.ffn.fc_in.\2",
        r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$":
        r"blocks.\1.ffn.fc_out.\2",
        r"blocks\.(\d+)\.norm2\.(.*)$":
        r"blocks.\1.self_attn_residual_norm.norm.\2",
    }

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        text_len=512,
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.text_len = text_len

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(in_chans=in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=patch_size,
                                          flatten=False)

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim, ffn_dim, num_attention_heads,
                                qk_norm, cross_attn_norm, eps,
                                added_kv_proj_dim) for _ in range(num_layers)
        ])

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        seq_len: Optional[int] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if y is not None:
            hidden_states = torch.cat([hidden_states, y], dim=1)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # Get rotary embeddings
        d = self.inner_dim // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames * get_sequence_model_parallel_world_size(),
             post_patch_height, post_patch_width),
            self.inner_dim,
            self.num_attention_heads,
            rope_dim_list,
            rope_theta=10000)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if seq_len is None:
            seq_len = hidden_states.size(1)
        hidden_states = torch.cat([
            hidden_states,
            hidden_states.new_zeros(1, seq_len - hidden_states.size(1),
                                    hidden_states.size(2))
        ],
                                  dim=1)

        encoder_hidden_states = torch.cat([
            encoder_hidden_states,
            encoder_hidden_states.new_zeros(
                1, self.text_len - encoder_hidden_states.size(1),
                encoder_hidden_states.size(2))
        ],
                                          dim=1)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image)
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj,
                    freqs_cis)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, freqs_cis)

        # 5. Output norm, projection & unpatchify
        with torch.cuda.amp.autocast(dtype=torch.float32):
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(
                2, dim=1)
            hidden_states = self.norm_out(hidden_states.float(), shift, scale)
            hidden_states = self.proj_out(hidden_states)

        output = self.unpatchify(hidden_states, grid_sizes)

        return output.float()

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """

        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = u.permute(6, 0, 3, 1, 4, 2, 5)
            # u = torch.einsum('fhwpqrc->cfphqwr', u.contiguous())
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        out = torch.cat(out, dim=0)
        return out
