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
        # self.norm1 = LayerNormScaleShift(in_features, norm_type="layer", elementwise_affine=True, dtype=torch.float32)
        self.norm1 = nn.LayerNorm(in_features)
        # self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.ff = MLP(in_features, in_features, out_features, act_type="gelu")
        # self.norm2 = FP32LayerNorm(out_features)
        # self.norm2 = LayerNormScaleShift(out_features, norm_type="layer", elementwise_affine=True, dtype=torch.float32)
        self.norm2 = nn.LayerNorm(out_features)

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
        self.text_embedder = MLP(text_embed_dim, dim, dim, bias=True, act_type="gelu_pytorch_tanh")

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
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
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
        q = self.norm_q.forward_native(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k.forward_native(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)

        # compute attention
        x = flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False)
        # x = flash_attention(q, k, v, k_lens=context_lens)

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
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        # self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_added_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

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
        k_img = self.norm_added_k.forward_native(self.add_k_proj(context_img)[0]).view(b, -1, n, d)
        v_img = self.add_v_proj(context_img)[0].view(b, -1, n, d)
        # img_x = flash_attention(q, k_img, v_img, k_lens=None)
        img_x = flash_attn_func(q, k_img, v_img, dropout_p=0, softmax_scale=None, causal=False)
        # compute attention
        # x = flash_attention(q, k, v, k_lens=context_lens)
        x = flash_attn_func(q, k, v, dropout_p=0, softmax_scale=None, causal=False)

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
        # self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        # self.norm1 = LayerNormScaleShift(dim, norm_type="layer", eps=eps, elementwise_affine=False, dtype=torch.float32)
        self.norm1 = nn.LayerNorm(dim, eps, elementwise_affine=False)
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
            self.q_norm = RMSNorm(dim, eps=eps)
            self.k_norm = RMSNorm(dim, eps=eps)
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
        # self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        # self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        # self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm3 = LayerNormScaleShift(dim, norm_type="layer", eps=eps, elementwise_affine=False, dtype=torch.float32)
        # self.norm3 = WanLayerNorm(dim, eps)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        bs, seq_length, _ = hidden_states.shape
        orig_dtype = hidden_states.dtype
        assert temb.dtype == torch.float32
        with torch.cuda.amp.autocast(dtype=torch.float32):
            e = self.scale_shift_table + temb
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(6, dim=1)
            print("e: ", torch.sum(e.float()).item())
        assert shift_msa.dtype == torch.float32

        # torch.manual_seed(42)
        # hidden_states = torch.randn(hidden_states.shape, dtype=orig_dtype).to(hidden_states.device)
        # print(hidden_states.dtype)
        # 1. Self-attention
        print("hidden_states: ", torch.sum(hidden_states.float()).item())
        print("scale_msa: ", torch.sum(scale_msa).item())
        print("shift_msa: ", torch.sum(shift_msa).item())
        # norm_hidden_states = self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa
        norm_hidden_states = self.norm1(hidden_states.float()).to(dtype=orig_dtype) * (1 + scale_msa) + shift_msa
        print("norm_hidden_states: ", torch.sum(norm_hidden_states.float()).item())
        query, _ = self.to_q(norm_hidden_states)
        print("q: ", torch.sum(query.float()).item())
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.q_norm is not None:
            query = self.q_norm.forward_native(query)
            print("norm_q: ", torch.sum(query.float()).item())
        if self.k_norm is not None:
            key = self.k_norm.forward_native(key)
            # print("norm_k: ", torch.sum(key.float()).item())
        print("v: ", torch.sum(value.float()).item())

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        # Apply rotary embeddings
        # cos, sin = freqs_cis
        # query, key = _apply_rotary_emb(query, cos, sin, is_neox_style=True), _apply_rotary_emb(key, cos, sin, is_neox_style=True)

        # attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        attn_output, _ = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)
        print("self attn ouput: ", torch.sum(attn_output.float()).item())
        print("gate_msa: ", torch.sum(gate_msa.float()).item())
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + attn_output * gate_msa
            # hidden_states = hidden_states + attn_output

        print("Self attn + x: ", torch.sum(hidden_states.float()).item())

        # 2. Cross-attention
        with torch.cuda.amp.autocast(dtype=torch.float32):
            null_mod = torch.tensor([0], device=hidden_states.device)
            norm_hidden_states = self.norm2(hidden_states, null_mod, null_mod)
        print("x_norm before cross attn: ", torch.sum(norm_hidden_states.float()).item())
        print("encoder_hidden_states: ", torch.sum(encoder_hidden_states.float()).item())
        attn_output = self.attn2(norm_hidden_states, context=encoder_hidden_states, context_lens=None)
        print("cross attn output: ", torch.sum(attn_output.float()).item())
        # attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output
        print("cross attn + x: ", torch.sum(hidden_states.float()).item())

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states.float(), c_shift_msa, c_scale_msa).to(
            orig_dtype
        ).squeeze(1)
        # norm_hidden_states = self.norm3(hidden_states).float() * (1 + c_scale_msa) + c_shift_msa
        # norm_hidden_states = self.norm3(hidden_states)
        print("c_scale_msa: ", torch.sum(c_scale_msa.float()).item())
        print("c_shift_msa: ", torch.sum(c_shift_msa.float()).item())
        print("x_norm before ffn: ", torch.sum(norm_hidden_states.float()).item())
        ff_output = self.ffn(norm_hidden_states)
        print("ffn output: ", torch.sum(ff_output.float()).item())
        print("c_gate_msa: ", torch.sum(c_gate_msa.float()).item())
        with torch.cuda.amp.autocast(dtype=torch.float32):
            hidden_states = hidden_states + ff_output * c_gate_msa
            # hidden_states = hidden_states + ff_output
        print("final output: ", torch.sum(hidden_states.float()).item())

        return hidden_states

class WanVideoDiT(BaseDiT):
    _fsdp_shard_conditions = [
        lambda n, m: "blocks" in n and str.isdigit(n.split(".")[-1]),
    ]
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        text_len = 512,
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
        # self.norm_out = WanLayerNorm(inner_dim, eps)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        seq_len: int,
        encoder_hidden_states: torch.Tensor,
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

        # rotary_emb = self.rope(hidden_states)
        # Get rotary embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed((post_patch_num_frames * get_sequence_model_parallel_world_size(), post_patch_height, post_patch_width), self.inner_dim, self.num_attention_heads, [16, 56, 56], rope_theta=10000)
        freqs_cos = freqs_cos.to(hidden_states.device)
        freqs_sin = freqs_sin.to(hidden_states.device)
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        hidden_states = self.patch_embedding(hidden_states)
        grid_sizes = torch.stack(
            [torch.tensor(hidden_states[0].shape[1:], dtype=torch.long)])
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(1, seq_len - hidden_states.size(1), hidden_states.size(2))], dim=1)

        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states.new_zeros(1, self.text_len - encoder_hidden_states.size(1), encoder_hidden_states.size(2))], dim=1)
        # encoder_hidden_states = torch.stack([
        #                             torch.cat(
        #                                 [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        #                             for u in encoder_hidden_states
        #                         ])

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        # return timestep_proj

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
        with torch.cuda.amp.autocast(dtype=torch.float32):
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
            # hidden_states = self.norm_out(hidden_states, shift, scale).squeeze(1)
            print("shift: ", torch.sum(shift.float()).item())
            print("scale: ", torch.sum(scale.float()).item())
            print("hidden_states: ", torch.sum(hidden_states.float()).item())
            # hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
            hidden_states = self.norm_out(hidden_states.float(), shift, scale)
            print("hidden_states normed: ", torch.sum(hidden_states.float()).item())
            hidden_states = self.proj_out(hidden_states)

        # hidden_states = hidden_states.reshape(
        #     batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        # )
        # hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        # output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        # output = unpatchify(hidden_states, post_patch_num_frames, post_patch_height, post_patch_width, self.patch_size, self.out_channels)
        output = self.unpatchify(hidden_states, grid_sizes)

        return output.float()
        # return hidden_states
    
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