# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.configs.models.dits.dreamx_world import DreamXWorldConfig
from fastvideo.distributed.communication_op import (
    sequence_model_parallel_all_gather_with_unpad,
    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
from fastvideo.models.dits.wanvideo import (LayerNormScaleShift,
                                            PatchEmbed,
                                            WanTimeTextImageEmbedding,
                                            WanTransformer3DModel,
                                            WanTransformerBlock)
from fastvideo.platforms import AttentionBackendEnum, current_platform
from fastvideo.attention import LocalAttention
from fastvideo.distributed.parallel_state import get_sp_world_size
from fastvideo.layers.quantization import QuantizationConfig
from fastvideo.models.dits.base import BaseDiT


def _dreamx_invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    assert transforms.shape[-2:] == (4, 4)
    rot_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_inv,
                                    transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out.to(dtype=transforms.dtype)


def _dreamx_lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros(intrinsics.shape[:-2] + (4, 4),
                      device=intrinsics.device,
                      dtype=intrinsics.dtype)
    out[..., :3, :3] = intrinsics
    out[..., 3, 3] = 1.0
    return out


def _dreamx_invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    assert intrinsics.shape[-2:] == (3, 3)
    out = torch.zeros_like(intrinsics)
    out[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    out[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    out[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    out[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out.to(dtype=intrinsics.dtype)


def _dreamx_apply_tiled_projmat(feats: torch.Tensor,
                                matrix: torch.Tensor) -> torch.Tensor:
    batch, num_heads, seq_len, feat_dim = feats.shape
    proj_dim = matrix.shape[-1]
    assert feat_dim % proj_dim == 0

    if matrix.shape[1] == seq_len:
        feats = feats.view(batch, num_heads, seq_len, feat_dim // proj_dim,
                           proj_dim)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats)
        return out.reshape(batch, num_heads, seq_len, feat_dim)

    cameras = matrix.shape[1]
    assert seq_len > cameras and seq_len % cameras == 0
    feats = feats.reshape(batch, num_heads, cameras, -1,
                          feat_dim // proj_dim, proj_dim)
    out = torch.einsum("bcij,bncpkj->bncpki", matrix, feats)
    return out.reshape(batch, num_heads, seq_len, feat_dim)


def _dreamx_prope_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      viewmats: torch.Tensor, intrinsics: torch.Tensor):
    batch, num_heads, seq_len, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert intrinsics.shape == (batch, cameras, 3, 3)
    assert head_dim % 4 == 0

    intrinsics_norm = torch.zeros_like(intrinsics)
    intrinsics_norm[..., 0, 0] = intrinsics[..., 0, 0]
    intrinsics_norm[..., 1, 1] = intrinsics[..., 1, 1]
    intrinsics_norm[..., 2, 2] = 1.0

    proj = torch.einsum("...ij,...jk->...ik",
                        _dreamx_lift_k(intrinsics_norm), viewmats)
    proj_t = proj.transpose(-1, -2).to(dtype=viewmats.dtype)
    proj_inv = torch.einsum(
        "...ij,...jk->...ik",
        _dreamx_invert_se3(viewmats),
        _dreamx_lift_k(_dreamx_invert_k(intrinsics_norm)),
    ).to(dtype=viewmats.dtype)

    q = _dreamx_apply_tiled_projmat(q, proj_t)
    k = _dreamx_apply_tiled_projmat(k, proj_inv)
    v = _dreamx_apply_tiled_projmat(v, proj_inv)
    return q, k, v, proj


class DreamXPropeSelfAttention(nn.Module):
    """DreamX-World parallel PRoPE camera self-attention branch."""

    def __init__(self,
                 dim: int,
                 attn_dim: int,
                 num_heads: int,
                 qk_norm: str | bool = True,
                 eps: float = 1e-6,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = ""):
        super().__init__()
        assert attn_dim % num_heads == 0
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.qk_norm = qk_norm

        self.q_proj = ReplicatedLinear(dim,
                                       attn_dim,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.q_proj")
        self.k_proj = ReplicatedLinear(dim,
                                       attn_dim,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.k_proj")
        self.v_proj = ReplicatedLinear(dim,
                                       attn_dim,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.v_proj")
        self.out_proj = ReplicatedLinear(attn_dim,
                                         dim,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.out_proj")

        if qk_norm == "rms_norm":
            self.norm_q = RMSNorm(self.head_dim, eps=eps)
            self.norm_k = RMSNorm(self.head_dim, eps=eps)
        elif qk_norm in (True, "rms_norm_across_heads"):
            self.norm_q = RMSNorm(attn_dim, eps=eps)
            self.norm_k = RMSNorm(attn_dim, eps=eps)
        elif qk_norm is False:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            raise ValueError(f"Unsupported qk_norm for DreamX PRoPE: {qk_norm}")

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,
                                          AttentionBackendEnum.TORCH_SDPA))

    def forward(self, hidden_states: torch.Tensor,
                y_camera: dict[str, torch.Tensor]) -> torch.Tensor:
        if get_sp_world_size() > 1:
            # The transformer shards the sequence before the block loop and
            # this branch uses LocalAttention (no all-to-all): under
            # sequence parallelism each rank would attend only within its
            # own shard — silently wrong output. Fail loudly until this
            # path is ported to DistributedAttention and validated.
            raise NotImplementedError(
                "DreamXPropeSelfAttention does not support sequence "
                "parallelism yet (LocalAttention on a sharded sequence "
                "corrupts output). Run with sp_size=1.")
        batch_size, seq_len, _ = hidden_states.shape

        query, _ = self.q_proj(hidden_states)
        key, _ = self.k_proj(hidden_states)
        value, _ = self.v_proj(hidden_states)

        if self.qk_norm == "rms_norm":
            query = query.view(batch_size, seq_len, self.num_heads,
                               self.head_dim)
            key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
            query = self.norm_q(query)
            key = self.norm_k(key)
        else:
            query = self.norm_q(query).view(batch_size, seq_len,
                                            self.num_heads, self.head_dim)
            key = self.norm_k(key).view(batch_size, seq_len, self.num_heads,
                                        self.head_dim)

        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query, key, value, output_projection = _dreamx_prope_qkv(
            query,
            key,
            value,
            viewmats=y_camera["viewmats"],
            intrinsics=y_camera["K"],
        )

        out = self.attn(query.transpose(1, 2), key.transpose(1, 2),
                        value.transpose(1, 2))
        out = _dreamx_apply_tiled_projmat(out.transpose(1, 2),
                                          output_projection).transpose(1, 2)
        out = out.flatten(2)
        out, _ = self.out_proj(out)
        return out


class DreamXWorldTransformerBlock(WanTransformerBlock):

    def __init__(self,
                 dim: int,
                 ffn_dim: int,
                 num_heads: int,
                 qk_norm: str = "rms_norm_across_heads",
                 cross_attn_norm: bool = False,
                 eps: float = 1e-6,
                 added_kv_proj_dim: int | None = None,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 quant_config: QuantizationConfig | None = None,
                 prefix: str = "",
                 add_control_adapter: bool = True,
                 cam_method: str | None = "prope",
                 attn_compress: int = 1,
                 cam_self_attn_layers: tuple[int, ...] | None = None,
                 layer_idx: int | None = None):
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm,
                         eps, added_kv_proj_dim,
                         supported_attention_backends, quant_config, prefix)
        self.cam_self_attn = None
        add_cam_attn = add_control_adapter and cam_method == "prope"
        if add_cam_attn and cam_self_attn_layers is not None:
            add_cam_attn = layer_idx in cam_self_attn_layers
        if add_cam_attn:
            if num_heads % attn_compress != 0 or dim % attn_compress != 0:
                raise ValueError("DreamX attn_compress must divide dim and num_heads")
            self.cam_self_attn = DreamXPropeSelfAttention(
                dim,
                dim // attn_compress,
                num_heads // attn_compress,
                qk_norm=qk_norm,
                eps=eps,
                quant_config=quant_config,
                prefix=f"{prefix}.cam_self_attn")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        y_camera: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype

        if temb.dim() == 4:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(
                    6, dim=2)
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            e = self.scale_shift_table + temb.float()
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = e.chunk(
                6, dim=1)
        assert shift_msa.dtype == torch.float32

        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1 + scale_msa) + shift_msa).to(orig_dtype)
        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        key = key.squeeze(1).unflatten(2, (self.num_attention_heads, -1))
        value = value.squeeze(1).unflatten(2, (self.num_attention_heads, -1))

        attn_output, _ = self.attn1(
            query,
            key,
            value,
            original_seq_len,
            freqs_cis=freqs_cis,
        )
        attn_output = attn_output.flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)
        if self.cam_self_attn is not None and y_camera is not None:
            attn_output = attn_output + self.cam_self_attn(
                norm_hidden_states, y_camera)

        null_shift = null_scale = torch.tensor([0], device=hidden_states.device)
        norm_hidden_states, hidden_states = self.self_attn_residual_norm(
            hidden_states, attn_output, gate_msa, null_shift, null_scale)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        attn_output = self.attn2(norm_hidden_states,
                                 context=encoder_hidden_states,
                                 context_lens=None)
        norm_hidden_states, hidden_states = self.cross_attn_residual_norm(
            hidden_states, attn_output, 1, c_shift_msa, c_scale_msa)
        norm_hidden_states, hidden_states = norm_hidden_states.to(
            orig_dtype), hidden_states.to(orig_dtype)

        ff_output = self.ffn(norm_hidden_states)
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


class DreamXWorldTransformer3DModel(WanTransformer3DModel):
    _fsdp_shard_conditions = DreamXWorldConfig()._fsdp_shard_conditions
    _compile_conditions = DreamXWorldConfig()._compile_conditions
    _supported_attention_backends = DreamXWorldConfig(
    )._supported_attention_backends
    param_names_mapping = DreamXWorldConfig().param_names_mapping
    reverse_param_names_mapping = DreamXWorldConfig().reverse_param_names_mapping
    lora_param_names_mapping = DreamXWorldConfig().lora_param_names_mapping

    def __init__(self, config: DreamXWorldConfig, hf_config: dict[str,
                                                                  Any]) -> None:
        BaseDiT.__init__(self, config=config, hf_config=hf_config)
        self.quant_config = config.quant_config

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size
        self.text_len = config.text_len

        assert config.num_attention_heads % get_sp_world_size() == 0, f"The number of attention heads ({config.num_attention_heads}) must be divisible by the sequence parallel size ({get_sp_world_size()})"

        self.patch_embedding = PatchEmbed(in_chans=config.in_channels,
                                          embed_dim=inner_dim,
                                          patch_size=config.patch_size,
                                          flatten=False)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )
        self.blocks = nn.ModuleList([
            DreamXWorldTransformerBlock(
                inner_dim,
                config.ffn_dim,
                config.num_attention_heads,
                config.qk_norm,
                config.cross_attn_norm,
                config.eps,
                config.added_kv_proj_dim,
                self._supported_attention_backends,
                quant_config=config.quant_config,
                prefix=f"{config.prefix}.blocks.{i}",
                add_control_adapter=config.add_control_adapter,
                cam_method=config.cam_method,
                attn_compress=config.attn_compress,
                cam_self_attn_layers=config.cam_self_attn_layers,
                layer_idx=i)
            for i in range(config.num_layers)
        ])
        self.norm_out = LayerNormScaleShift(inner_dim,
                                            norm_type="layer",
                                            eps=config.eps,
                                            elementwise_affine=False,
                                            dtype=torch.float32,
                                            compute_dtype=torch.float32)
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.gradient_checkpointing = False
        self.__post_init__()

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor]
                | None = None,
                guidance=None,
                y_camera: dict[str, torch.Tensor] | None = None,
                **kwargs) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        if encoder_hidden_states is not None and not isinstance(
                encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_hidden_states_image,
                      list) and len(encoder_hidden_states_image) > 0:
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        d = self.hidden_size // self.num_attention_heads
        rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            (post_patch_num_frames, post_patch_height, post_patch_width),
            self.hidden_size,
            self.num_attention_heads,
            rope_dim_list,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
            rope_theta=10000)
        freqs_cis = (freqs_cos.to(hidden_states.device).float(),
                     freqs_sin.to(hidden_states.device).float())

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states, original_seq_len = sequence_model_parallel_shard(
            hidden_states, dim=1)

        if timestep.dim() == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len)
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states_image

        if current_platform.is_mps() or current_platform.is_npu():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)

        assert encoder_hidden_states.dtype == orig_dtype

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    timestep_proj, freqs_cis, original_seq_len, y_camera)
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states,
                                      timestep_proj, freqs_cis, original_seq_len,
                                      y_camera=y_camera)

        if temb.dim() == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) +
                            temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table +
                            temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(
            hidden_states, original_seq_len, dim=1)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames,
                                              post_patch_height,
                                              post_patch_width, p_t, p_h, p_w,
                                              -1)
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


EntryClass = DreamXWorldTransformer3DModel
