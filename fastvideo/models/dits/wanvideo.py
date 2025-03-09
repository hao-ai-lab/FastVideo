import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union, Dict, Any
from fastvideo.attention.distributed_attn import DistributedAttention, LocalAttention
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.layernorm import LayerNormScaleShift, ScaleResidual, ScaleResidualLayerNormScaleShift, RMSNorm
from fastvideo.layers.visual_embedding import PatchEmbed, TimestepEmbedder, ModulateProjection, unpatchify
from fastvideo.layers.rotary_embedding import _apply_rotary_emb, get_rotary_pos_embed
from fastvideo.distributed.parallel_state import get_sequence_model_parallel_world_size
from flash_attn import flash_attn_func
# from torch.nn import RMSNorm
# TODO: RMSNorm ....
from fastvideo.layers.mlp import MLP
from fastvideo.models.dits.base import BaseDiT

class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        # self.norm1 = FP32LayerNorm(in_features)
        self.norm1 = LayerNormScaleShift(in_features, norm_type="layer", elementwise_affine=True, dtype=torch.float32)
        # self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.ffn = MLP(in_features, in_features, out_features, act_type="gelu")
        # self.norm2 = FP32LayerNorm(out_features)
        self.norm2 = LayerNormScaleShift(out_features, norm_type="layer", elementwise_affine=True, dtype=torch.float32)

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states
    
class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        # self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        # self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.time_embedder = TimestepEmbedder(dim, frequency_embedding_size=time_freq_dim, act_layer="silu")
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        # self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.text_embedder = MLP(text_embed_dim, dim, bias=True, act_type="gelu_pytorch_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        temb = self.time_embedder(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

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
        self.q = ReplicatedLinear(dim, dim)
        self.k = ReplicatedLinear(dim, dim)
        self.v = ReplicatedLinear(dim, dim)
        self.o = ReplicatedLinear(dim, dim)
        # self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

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
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attn_func(q, k, v, k_lens=context_lens)
        # x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = ReplicatedLinear(dim, dim)
        self.v_img = ReplicatedLinear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        # self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_img = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

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
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        # img_x = flash_attention(q, k_img, v_img, k_lens=None)
        img_x = flash_attn_func(q, k_img, v_img, k_lens=None)
        # compute attention
        # x = flash_attention(q, k, v, k_lens=context_lens)
        x = flash_attn_func(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
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
        # self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm1 = LayerNormScaleShift(dim, norm_type="layer", eps=eps, elementwise_affine=False, dtype=torch.float32)
        # self.attn1 = Attention(
        #     query_dim=dim,
        #     heads=num_heads,
        #     kv_heads=num_heads,
        #     dim_head=dim // num_heads,
        #     qk_norm=qk_norm,
        #     eps=eps,
        #     bias=True,
        #     cross_attention_dim=None,
        #     out_bias=True,
        #     processor=WanAttnProcessor2_0(),
        # )
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)
        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(
            dropout_rate=0.0,
            causal=False
        )
        self.hidden_dim = dim
        self.num_attention_heads = num_heads
        dim_head = dim // num_heads
        if qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=eps)
            self.k_norm = RMSNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.q_norm = RMSNorm(dim_head * num_heads, eps=eps)
            self.k_norm = RMSNorm(dim_head * num_heads, eps=eps)
        else:
            print("QK Norm type not supported")
            raise Exception

        # 2. Cross-attention
        # self.attn2 = Attention(
        #     query_dim=dim,
        #     heads=num_heads,
        #     kv_heads=num_heads,
        #     dim_head=dim // num_heads,
        #     qk_norm=qk_norm,
        #     eps=eps,
        #     bias=True,
        #     cross_attention_dim=None,
        #     out_bias=True,
        #     added_kv_proj_dim=added_kv_proj_dim,
        #     added_proj_bias=True,
        #     processor=WanAttnProcessor2_0(),
        # )
        # self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        else:
            # T2V
            self.attn2 = WanT2VCrossAttention(dim, num_heads, qk_norm=qk_norm, eps=eps)
        self.norm2 = LayerNormScaleShift(dim, norm_type="layer", eps=eps, elementwise_affine=True, dtype=torch.float32) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        # self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        # self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = LayerNormScaleShift(dim, norm_type="layer", eps=eps, elementwise_affine=False, dtype=torch.float32)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Tuple(torch.Tensor, torch.Tensor) = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        query = self.to_q(norm_hidden_states)
        key = self.to_k(norm_hidden_states)
        value = self.to_v(norm_hidden_states)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        query = query.unflatten(2, (self.num_attention_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.num_attention_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.num_attention_heads, -1)).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = freqs_cis
        query, key = _apply_rotary_emb(query, cos, sin, is_neox_style=False), _apply_rotary_emb(key, cos, sin, is_neox_style=False)

        # attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        attn_output, _ = self.attn1(query, key, value)
        attn_output = self.to_out(attn_output)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, context=encoder_hidden_states, context_length=None)
        # attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class WanVideoDiT(BaseDiT):
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
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
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        # self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        # self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.norm_out = LayerNormScaleShift(inner_dim, norm_type="layer", eps=eps, elementwise_affine=False, dtype=torch.float32)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # rotary_emb = self.rope(hidden_states)
        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed((post_patch_num_frames * get_sequence_model_parallel_world_size(), post_patch_height, post_patch_width), self.inner_dim, self.num_attention_heads)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, freqs_cis
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, freqs_cis)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output