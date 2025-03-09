import torch
import torch.nn as nn
from typing import Optional, Tuple, List
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
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
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
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
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
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False