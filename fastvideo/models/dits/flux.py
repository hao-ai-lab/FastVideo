# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits import FluxConfig
from fastvideo.layers.activation import get_act_fn
from fastvideo.layers.layernorm import FP32LayerNorm, RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.rotary_embedding import get_1d_rotary_pos_embed
from fastvideo.models.dits.base import BaseDiT
from fastvideo.platforms import AttentionBackendEnum


def timestep_embedding(t: torch.Tensor,
                       dim: int,
                       max_period: int = 10000,
                       time_factor: float = 1000.0) -> torch.Tensor:
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period, dtype=torch.float32)) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, dtype: torch.dtype | None):
        super().__init__()
        self.in_layer = ReplicatedLinear(in_dim,
                                         hidden_dim,
                                         bias=True,
                                         params_dtype=dtype)
        self.act = get_act_fn("silu")
        self.out_layer = ReplicatedLinear(hidden_dim,
                                          hidden_dim,
                                          bias=True,
                                          params_dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.in_layer(x)
        x = self.act(x)
        x, _ = self.out_layer(x)
        return x


class QKNorm(nn.Module):

    def __init__(self, dim: int, dtype: torch.dtype | None):
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=1e-6, dtype=dtype)
        self.key_norm = RMSNorm(dim, eps=1e-6, dtype=dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v.dtype), k.to(v.dtype)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool,
                 dtype: torch.dtype | None,
                 supported_attention_backends: tuple[AttentionBackendEnum,
                                                     ...]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = ReplicatedLinear(dim,
                                    dim * 3,
                                    bias=qkv_bias,
                                    params_dtype=dtype)
        self.norm = QKNorm(head_dim, dtype=dtype)
        self.proj = ReplicatedLinear(dim,
                                     dim,
                                     bias=True,
                                     params_dtype=dtype)

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=head_dim,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        qkv, _ = self.qkv(x)
        q, k, v = rearrange(qkv,
                            "b l (k h d) -> k b l h d",
                            k=3,
                            h=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = self.attn(q, k, v, freqs_cis=freqs_cis)
        attn = attn.reshape(x.shape[0], x.shape[1], -1)
        out, _ = self.proj(attn)
        return out


class Modulation(nn.Module):

    def __init__(self, dim: int, double: bool, dtype: torch.dtype | None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = ReplicatedLinear(dim,
                                    self.multiplier * dim,
                                    bias=True,
                                    params_dtype=dtype)

    def forward(self, vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor]:
        out, _ = self.lin(torch.nn.functional.silu(vec))
        chunks = out[:, None, :].chunk(self.multiplier, dim=-1)
        return chunks  # shift/scale/gate tuples


class DoubleStreamBlock(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 qkv_bias: bool,
                 dtype: torch.dtype | None,
                 supported_attention_backends: tuple[AttentionBackendEnum,
                                                     ...],
                 prefix: str = ""):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.img_mod = Modulation(hidden_size, double=True, dtype=dtype)
        self.img_norm1 = FP32LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.img_attn = SelfAttention(hidden_size,
                                      num_heads,
                                      qkv_bias=qkv_bias,
                                      dtype=dtype,
                                      supported_attention_backends=
                                      supported_attention_backends)
        self.img_norm2 = FP32LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.img_mlp = MLP(hidden_size,
                           mlp_hidden_dim,
                           bias=True,
                           act_type="gelu_tanh",
                           dtype=dtype,
                           prefix=f"{prefix}.img_mlp")

        self.txt_mod = Modulation(hidden_size, double=True, dtype=dtype)
        self.txt_norm1 = FP32LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.txt_attn = SelfAttention(hidden_size,
                                      num_heads,
                                      qkv_bias=qkv_bias,
                                      dtype=dtype,
                                      supported_attention_backends=
                                      supported_attention_backends)
        self.txt_norm2 = FP32LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.txt_mlp = MLP(hidden_size,
                           mlp_hidden_dim,
                           bias=True,
                           act_type="gelu_tanh",
                           dtype=dtype,
                           prefix=f"{prefix}.txt_mlp")

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_shift1, img_scale1, img_gate1, img_shift2, img_scale2, img_gate2 = self.img_mod(
            vec)
        txt_shift1, txt_scale1, txt_gate1, txt_shift2, txt_scale2, txt_gate2 = self.txt_mod(
            vec)

        img_mod = self.img_norm1(img)
        img_mod = (1 + img_scale1) * img_mod + img_shift1
        txt_mod = self.txt_norm1(txt)
        txt_mod = (1 + txt_scale1) * txt_mod + txt_shift1

        qkv_img, _ = self.img_attn.qkv(img_mod)
        img_q, img_k, img_v = rearrange(qkv_img,
                                        "b l (k h d) -> k b l h d",
                                        k=3,
                                        h=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        qkv_txt, _ = self.txt_attn.qkv(txt_mod)
        txt_q, txt_k, txt_v = rearrange(qkv_txt,
                                        "b l (k h d) -> k b l h d",
                                        k=3,
                                        h=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=1)
        k = torch.cat((txt_k, img_k), dim=1)
        v = torch.cat((txt_v, img_v), dim=1)

        attn = self.img_attn.attn(q, k, v, freqs_cis=freqs_cis)
        txt_attn = attn[:, :txt.shape[1]]
        img_attn = attn[:, txt.shape[1]:]

        img = img + img_gate1 * self.img_attn.proj(img_attn.reshape(
            img.shape[0], img.shape[1], -1))[0]
        img = img + img_gate2 * self.img_mlp((1 + img_scale2) *
                                             self.img_norm2(img) +
                                             img_shift2)

        txt = txt + txt_gate1 * self.txt_attn.proj(txt_attn.reshape(
            txt.shape[0], txt.shape[1], -1))[0]
        txt = txt + txt_gate2 * self.txt_mlp((1 + txt_scale2) *
                                             self.txt_norm2(txt) +
                                             txt_shift2)
        return img, txt


class SingleStreamBlock(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 mlp_ratio: float,
                 dtype: torch.dtype | None,
                 supported_attention_backends: tuple[AttentionBackendEnum,
                                                     ...],
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.linear1 = ReplicatedLinear(hidden_size,
                                        hidden_size * 3 + self.mlp_hidden_dim,
                                        bias=True,
                                        params_dtype=dtype)
        self.linear2 = ReplicatedLinear(hidden_size + self.mlp_hidden_dim,
                                        hidden_size,
                                        bias=True,
                                        params_dtype=dtype)
        self.norm = QKNorm(hidden_size // num_heads, dtype=dtype)
        self.pre_norm = FP32LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6)
        self.mlp_act = get_act_fn("gelu_tanh")
        self.modulation = Modulation(hidden_size, double=False, dtype=dtype)

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=hidden_size // num_heads,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor,
                freqs_cis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec)[:3]
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift
        linear1_out, _ = self.linear1(x_mod)
        qkv, mlp = torch.split(
            linear1_out, [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv,
                            "b l (k h d) -> k b l h d",
                            k=3,
                            h=self.num_heads)
        q, k = self.norm(q, k, v)
        attn = self.attn(q, k, v, freqs_cis=freqs_cis)
        attn = attn.reshape(x.shape[0], x.shape[1], -1)
        out, _ = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + mod_gate * out


class LastLayer(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 patch_size: int,
                 out_channels: int,
                 dtype: torch.dtype | None,
                 prefix: str = ""):
        super().__init__()
        self.norm_final = FP32LayerNorm(hidden_size,
                                        elementwise_affine=False,
                                        eps=1e-6)
        self.linear = ReplicatedLinear(hidden_size,
                                       patch_size * patch_size * out_channels,
                                       bias=True,
                                       params_dtype=dtype,
                                       prefix=f"{prefix}.linear")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(hidden_size,
                             2 * hidden_size,
                             bias=True,
                             params_dtype=dtype,
                             prefix=f"{prefix}.adaLN_modulation"),
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x, _ = self.linear(x)
        return x


def _build_freqs_from_ids(ids: torch.Tensor, axes_dim: list[int],
                          theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    ids_0 = ids[0]  # [S, A]
    cos_list = []
    sin_list = []
    for i, dim in enumerate(axes_dim):
        cos, sin = get_1d_rotary_pos_embed(dim, ids_0[:, i], theta=theta)
        cos_list.append(cos)
        sin_list.append(sin)
    cos = torch.cat(cos_list, dim=1)
    sin = torch.cat(sin_list, dim=1)
    return cos, sin


def _build_ids_from_grid(height: int, width: int, n_axes: int,
                         device: torch.device) -> torch.Tensor:
    grid_y = torch.arange(height, device=device)
    grid_x = torch.arange(width, device=device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    if n_axes == 1:
        ids = torch.arange(height * width, device=device)[:, None]
    elif n_axes >= 2:
        extra = []
        if n_axes > 2:
            extra = [torch.zeros_like(yy) for _ in range(n_axes - 2)]
        ids = torch.stack([yy, xx, *extra], dim=-1)
    return ids


class FluxTransformer2DModel(BaseDiT):
    _fsdp_shard_conditions = FluxConfig()._fsdp_shard_conditions
    _compile_conditions = FluxConfig()._compile_conditions
    _supported_attention_backends = FluxConfig()._supported_attention_backends
    param_names_mapping = FluxConfig().param_names_mapping
    reverse_param_names_mapping = FluxConfig().reverse_param_names_mapping
    lora_param_names_mapping = FluxConfig().lora_param_names_mapping

    def __init__(self, config: FluxConfig, hf_config: dict[str, Any]):
        super().__init__(config=config, hf_config=hf_config)
        dtype = getattr(config, "dtype", None)

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents

        self.vec_in_dim = config.pooled_projection_dim
        self.context_in_dim = config.joint_attention_dim
        self.axes_dim = list(config.rope_axes_dim)
        self.theta = config.rope_theta
        self.guidance_embed = config.guidance_embeds
        self.mlp_ratio = config.mlp_ratio
        self.qkv_bias = getattr(config, "qkv_bias", False)

        self.img_in = ReplicatedLinear(self.in_channels,
                                       self.hidden_size,
                                       bias=True,
                                       params_dtype=dtype)
        self.time_in = MLPEmbedder(in_dim=256,
                                   hidden_dim=self.hidden_size,
                                   dtype=dtype)
        self.vector_in = MLPEmbedder(in_dim=self.vec_in_dim,
                                     hidden_dim=self.hidden_size,
                                     dtype=dtype)
        self.guidance_in = (MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, dtype=dtype)
                            if self.guidance_embed else nn.Identity())
        self.txt_in = ReplicatedLinear(self.context_in_dim,
                                       self.hidden_size,
                                       bias=True,
                                       params_dtype=dtype)

        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(
                self.hidden_size,
                self.num_attention_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                dtype=dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.double_blocks.{i}",
            ) for i in range(config.num_layers)
        ])

        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(
                self.hidden_size,
                self.num_attention_heads,
                mlp_ratio=self.mlp_ratio,
                dtype=dtype,
                supported_attention_backends=self._supported_attention_backends,
                prefix=f"{config.prefix}.single_blocks.{i}",
            ) for i in range(config.num_single_layers)
        ])

        self.final_layer = LastLayer(self.hidden_size,
                                     patch_size=1,
                                     out_channels=self.out_channels,
                                     dtype=dtype,
                                     prefix=f"{config.prefix}.final_layer")

        self.__post_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_2: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError(
                "FluxTransformer2DModel expects hidden_states with shape [B, C, T, H, W]"
            )

        img = rearrange(hidden_states, "b c t h w -> b (t h w) c")
        txt = encoder_hidden_states
        if isinstance(txt, list):
            txt = txt[0]

        y = encoder_hidden_states_2
        if y is None:
            y = torch.zeros(txt.shape[0],
                            self.vec_in_dim,
                            device=txt.device,
                            dtype=txt.dtype)

        img, _ = self.img_in(img)
        vec = self.time_in(timestep_embedding(timestep, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Guidance value is required for guidance-distilled Flux.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt, _ = self.txt_in(txt)

        bsz, txt_len, _ = txt.shape
        _, img_len, _ = img.shape
        if txt_ids is None:
            txt_ids = torch.zeros(bsz,
                                  txt_len,
                                  len(self.axes_dim),
                                  device=txt.device)
        if img_ids is None:
            _, _, _, h, w = hidden_states.shape
            ids = _build_ids_from_grid(h, w, len(self.axes_dim), txt.device)
            img_ids = ids.unsqueeze(0).expand(bsz, -1, -1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        freqs_cis = _build_freqs_from_ids(ids, self.axes_dim, self.theta)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, freqs_cis=freqs_cis)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, freqs_cis=freqs_cis)
        img = img[:, txt.shape[1]:, ...]

        img = self.final_layer(img, vec)
        img = rearrange(img, "b (t h w) c -> b c t h w", t=1, h=h, w=w)
        return img
