# SPDX-License-Identifier: Apache-2.0
"""PAN2 video transformer.

A mixture-of-transformers DiT: the video and text streams keep separate weights and meet only in one joint
attention per block, over ``[video; text]``. The text stream is modulated by a learned, timestep-free table.
Under sequence parallelism the video tokens are sharded. The dense transformer replicates the text tokens on every
rank. A sparse (VSA) student shards them too, as it was trained and validated: the text refiner runs on the whole
caption on every rank, then each rank keeps a shard of the caption, so the text stream's per-token layers run once per
token rather than once per rank, and the attention all-to-all carries both streams.

Parallel Decoding Distillation (PDD) students such as FastPAN2 widen ``proj_out`` to one head per interval of a fine
time grid (``fastvideo.layers.pdd``) and may attend block-sparse with VSA-H3 / VSA2, which adds a learned
``to_gate_compress`` per block.
"""

import contextlib
from collections.abc import Iterator, Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastvideo.attention import DistributedAttention, LocalAttention
from fastvideo.attention.layer import DistributedAttention_VSA
from fastvideo.attention.selector import get_attn_backend
from fastvideo.configs.models.dits.pan2 import PAN2VideoConfig
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather_with_unpad,
                                                    sequence_model_parallel_shard)
from fastvideo.distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size
from fastvideo.layers.layernorm import RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.pdd import PDDReplicatedLinear, fuse_pdd_heads
from fastvideo.layers.rotary_embedding import get_1d_rotary_pos_embed
from fastvideo.layers.visual_embedding import PatchEmbed, TimestepEmbedder
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.pan2_fusions import layer_norm_modulate, qk_rmsnorm_rope
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import get_compute_dtype


def _fusable_layer_norm(x: torch.Tensor) -> bool:
    """Whether ``layer_norm_modulate`` computes what the eager ``norm(x) * (1 + scale) + shift`` does here.

    The fused op rounds once from FP32, which is the eager result under autocast (FP32 LayerNorm and modulation, cast
    by the next linear layer) and in FP32; a low-precision tensor outside autocast rounds after every eager op instead.
    """
    return x.dtype == torch.float32 or torch.is_autocast_enabled(x.device.type)


class PAN2SwiGLU(nn.Module):
    """SwiGLU MLP of the text refiner; the fused input projection stores ``[value; gate]``."""

    def __init__(self, dim: int, inner_dim: int, prefix: str = ""):
        super().__init__()
        self.fc_in = ReplicatedLinear(dim, 2 * inner_dim, bias=False, prefix=f"{prefix}.fc_in")
        self.fc_out = ReplicatedLinear(inner_dim, dim, bias=False, prefix=f"{prefix}.fc_out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        value, gate = x.chunk(2, dim=-1)
        x, _ = self.fc_out(value * F.silu(gate))
        return x


class PAN2RefinerAttention(nn.Module):
    """Text-only self-attention of the refiner; the text is replicated on every sequence-parallel rank."""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = ReplicatedLinear(dim, 3 * num_heads * head_dim, bias=False, prefix=f"{prefix}.to_qkv")
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = ReplicatedLinear(num_heads * head_dim, dim, bias=False, prefix=f"{prefix}.to_out")
        self.attn = LocalAttention(num_heads=num_heads,
                                   head_size=head_dim,
                                   causal=False,
                                   supported_attention_backends=supported_attention_backends)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.unflatten(-1, (3, self.num_heads, -1)).unbind(2)
        query = self.norm_q(query).to(value.dtype)
        key = self.norm_k(key).to(value.dtype)
        hidden_states = self.attn(query, key, value)
        hidden_states, _ = self.to_out(hidden_states.flatten(2))
        return hidden_states


class PAN2TokenRefinerBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 mlp_ratio: float,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = PAN2RefinerAttention(dim,
                                         num_heads,
                                         head_dim,
                                         eps,
                                         supported_attention_backends=supported_attention_backends,
                                         prefix=f"{prefix}.attn")
        self.norm2 = RMSNorm(dim, eps=eps)
        self.ff = PAN2SwiGLU(dim, int(dim * mlp_ratio), prefix=f"{prefix}.ff")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class PAN2AdaLayerNormZero(nn.Module):
    """Timestep modulation of the video stream: modulated attention input plus the gate/shift/scale for the rest."""

    # Set by PAN2Transformer3DModel.set_fused_kernels.
    fused_kernels = False

    def __init__(self, dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.linear = ReplicatedLinear(dim, 6 * dim, bias=True, prefix=f"{prefix}.linear")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor,
                temb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb, _ = self.linear(F.silu(temb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        if self.fused_kernels and _fusable_layer_norm(x):
            x = layer_norm_modulate(x, 1 + scale_msa, shift_msa, self.norm.eps)
        else:
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class PAN2AdaLayerNormOut(nn.Module):

    # Set by PAN2Transformer3DModel.set_fused_kernels.
    fused_kernels = False

    def __init__(self, dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.linear = ReplicatedLinear(dim, 2 * dim, bias=True, prefix=f"{prefix}.linear")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        emb, _ = self.linear(F.silu(temb).to(x.dtype))
        scale, shift = emb.chunk(2, dim=-1)
        if self.fused_kernels and _fusable_layer_norm(x):
            return layer_norm_modulate(x, 1 + scale, shift, self.norm.eps)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class PAN2Attention(nn.Module):
    """Joint attention over ``[video; text]`` with per-stream projections and Q/K RMSNorm.

    RoPE applies to the video tokens only; ``DistributedAttention`` applies it after its all-to-all, where each rank
    holds the full (unpadded) video sequence for its heads. Under the VSA-H3 backend the video tokens attend
    block-sparse and the text tokens stay dense, with the attention metadata describing the whole ``[video; text]``
    sequence. With ``original_text_len`` the text arrives sharded like the video and joins it after the all-to-all;
    without it the text is replicated on every rank. With fused kernels the Q/K RMSNorm and RoPE run as one op before
    the all-to-all, on this rank's video shard with ``local_freqs_cis``.
    """

    # Set by PAN2Transformer3DModel.set_fused_kernels.
    fused_kernels = False

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 eps: float,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.num_heads = num_heads
        inner_dim = num_heads * head_dim

        self.to_qkv = ReplicatedLinear(dim, 3 * inner_dim, bias=True, prefix=f"{prefix}.to_qkv")
        self.norm_q = RMSNorm(head_dim, eps=eps)
        self.norm_k = RMSNorm(head_dim, eps=eps)
        self.to_out = ReplicatedLinear(inner_dim, dim, bias=True, prefix=f"{prefix}.to_out")

        self.add_qkv_proj = ReplicatedLinear(dim, 3 * inner_dim, bias=True, prefix=f"{prefix}.add_qkv_proj")
        self.norm_added_q = RMSNorm(head_dim, eps=eps)
        self.norm_added_k = RMSNorm(head_dim, eps=eps)
        self.to_add_out = ReplicatedLinear(inner_dim, dim, bias=True, prefix=f"{prefix}.to_add_out")

        # VSA carries a learned gate on its pooled-compression branch, built only when the VSA-H3 backend resolves so
        # the dense teacher's state dict is unchanged. A sparse student's checkpoint carries the trained gate; one
        # without it gets zeros from the loader (``ALLOWED_NEW_PARAM_PATTERNS``), and the branch contributes nothing.
        resolved_backend = get_attn_backend(head_dim,
                                            get_compute_dtype(),
                                            supported_attention_backends=supported_attention_backends)
        use_vsa = resolved_backend.get_name() == "VIDEO_SPARSE_ATTN_H3"
        attention_cls = DistributedAttention_VSA if use_vsa else DistributedAttention
        self.attn = attention_cls(num_heads=num_heads,
                                  head_size=head_dim,
                                  causal=False,
                                  supported_attention_backends=supported_attention_backends,
                                  prefix=f"{prefix}.attn")
        self.to_gate_compress: ReplicatedLinear | None = None
        if use_vsa:
            self.to_gate_compress = ReplicatedLinear(dim, inner_dim, bias=False, prefix=f"{prefix}.to_gate_compress")

    def _project(self, projection: ReplicatedLinear,
                 hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = projection(hidden_states)
        query, key, value = qkv.unflatten(-1, (3, self.num_heads, -1)).unbind(2)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
        original_seq_len: int,
        original_text_len: int | None = None,
        local_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value = self._project(self.to_qkv, hidden_states)
        encoder_query, encoder_key, encoder_value = self._project(self.add_qkv_proj, encoder_hidden_states)
        if self.fused_kernels and local_freqs_cis is not None:
            # RoPE is per token and per head, so it applies to this rank's shard before the all-to-all as well as to
            # the gathered sequence after it.
            query, key = qk_rmsnorm_rope(query, key, self.norm_q.weight, self.norm_k.weight,
                                         self.norm_q.variance_epsilon, *local_freqs_cis)
            encoder_query, encoder_key = qk_rmsnorm_rope(encoder_query, encoder_key, self.norm_added_q.weight,
                                                         self.norm_added_k.weight, self.norm_added_q.variance_epsilon)
            freqs_cis = None
        else:
            query = self.norm_q(query).to(value.dtype)
            key = self.norm_k(key).to(value.dtype)
            encoder_query = self.norm_added_q(encoder_query).to(encoder_value.dtype)
            encoder_key = self.norm_added_k(encoder_key).to(encoder_value.dtype)

        extra_attention_kwargs: dict[str, Any] = {}
        if self.to_gate_compress is not None:
            gate_compress, _ = self.to_gate_compress(hidden_states)
            extra_attention_kwargs["gate_compress"] = gate_compress.unflatten(-1, (self.num_heads, -1))
        if original_text_len is None:
            text_kwargs = dict(replicated_q=encoder_query, replicated_k=encoder_key, replicated_v=encoder_value)
        else:
            text_kwargs = dict(secondary_q=encoder_query,
                               secondary_k=encoder_key,
                               secondary_v=encoder_value,
                               secondary_seq_len=original_text_len)
        hidden_states, encoder_hidden_states = self.attn(query,
                                                         key,
                                                         value,
                                                         original_seq_len,
                                                         freqs_cis=freqs_cis,
                                                         **text_kwargs,
                                                         **extra_attention_kwargs)
        hidden_states, _ = self.to_out(hidden_states.flatten(2))
        encoder_hidden_states, _ = self.to_add_out(encoder_hidden_states.flatten(2))
        return hidden_states, encoder_hidden_states


class PAN2TransformerBlock(nn.Module):

    # Set by PAN2Transformer3DModel.set_fused_kernels.
    fused_kernels = False

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 head_dim: int,
                 mlp_ratio: float,
                 context_mlp_ratio: float,
                 eps: float = 1e-6,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...] | None = None,
                 prefix: str = ""):
        super().__init__()
        self.norm1 = PAN2AdaLayerNormZero(dim, eps=eps, prefix=f"{prefix}.norm1")
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.scale_shift_table_context = nn.Parameter(torch.zeros(6 * dim))

        self.attn = PAN2Attention(dim,
                                  num_heads,
                                  head_dim,
                                  eps,
                                  supported_attention_backends=supported_attention_backends,
                                  prefix=f"{prefix}.attn")

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = MLP(dim, int(dim * mlp_ratio), act_type="gelu_pytorch_tanh", prefix=f"{prefix}.ff")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = MLP(dim,
                              int(dim * context_mlp_ratio),
                              act_type="gelu_pytorch_tanh",
                              prefix=f"{prefix}.ff_context")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        original_seq_len: int,
        original_text_len: int | None = None,
        local_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
        (
            context_shift_msa,
            context_scale_msa,
            context_gate_msa,
            context_shift_mlp,
            context_scale_mlp,
            context_gate_mlp,
        ) = self.scale_shift_table_context.to(temb.dtype).chunk(6, dim=-1)
        fused = self.fused_kernels and _fusable_layer_norm(encoder_hidden_states)
        if fused:
            norm_encoder_hidden_states = layer_norm_modulate(encoder_hidden_states, 1 + context_scale_msa,
                                                             context_shift_msa, self.norm1_context.eps)
        else:
            norm_encoder_hidden_states = (self.norm1_context(encoder_hidden_states) *
                                          (1 + context_scale_msa) + context_shift_msa)

        attn_output, context_attn_output = self.attn(norm_hidden_states, norm_encoder_hidden_states, freqs_cis,
                                                     original_seq_len, original_text_len, local_freqs_cis)
        if fused:
            # One rounding per gated residual, as PAN2's fused training kernels compute it.
            hidden_states = torch.addcmul(hidden_states, attn_output, gate_msa.unsqueeze(1))
            encoder_hidden_states = torch.addcmul(encoder_hidden_states, context_attn_output, context_gate_msa)
            norm_hidden_states = layer_norm_modulate(hidden_states, 1 + scale_mlp, shift_mlp, self.norm2.eps)
            norm_encoder_hidden_states = layer_norm_modulate(encoder_hidden_states, 1 + context_scale_mlp,
                                                             context_shift_mlp, self.norm2_context.eps)
            hidden_states = torch.addcmul(hidden_states, self.ff(norm_hidden_states), gate_mlp.unsqueeze(1))
            encoder_hidden_states = torch.addcmul(encoder_hidden_states, self.ff_context(norm_encoder_hidden_states),
                                                  context_gate_mlp)
            return hidden_states, encoder_hidden_states

        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * context_gate_msa

        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (self.norm2_context(encoder_hidden_states) *
                                      (1 + context_scale_mlp) + context_shift_mlp)

        hidden_states = hidden_states + self.ff(norm_hidden_states) * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + self.ff_context(norm_encoder_hidden_states) * context_gate_mlp
        return hidden_states, encoder_hidden_states


class PAN2Transformer3DModel(BaseDiT):
    """PAN2 DiT. The input stacks the noisy latents, the conditioning latents and a one-channel conditioning mask
    along channels, so one model serves text-to-video and first-frame image-to-video."""

    _fsdp_shard_conditions = PAN2VideoConfig()._fsdp_shard_conditions
    _compile_conditions = PAN2VideoConfig()._compile_conditions
    _supported_attention_backends = PAN2VideoConfig()._supported_attention_backends
    param_names_mapping = PAN2VideoConfig().param_names_mapping
    reverse_param_names_mapping = PAN2VideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = PAN2VideoConfig().lora_param_names_mapping

    def __init__(self, config: PAN2VideoConfig, hf_config: dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, hf_config=hf_config)
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = inner_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.out_channels = config.out_channels
        self.patch_size = tuple(config.patch_size)
        self.rope_axes_dim = tuple(config.rope_axes_dim)
        if sum(self.rope_axes_dim) != config.attention_head_dim:
            raise ValueError(f"rope_axes_dim {self.rope_axes_dim} must sum to the head dim {config.attention_head_dim}")
        if config.num_attention_heads % get_sp_world_size() != 0:
            raise ValueError(f"The number of attention heads ({config.num_attention_heads}) must be divisible by the "
                             f"sequence parallel size ({get_sp_world_size()})")

        self.x_embedder = PatchEmbed(self.patch_size,
                                     config.in_channels,
                                     inner_dim,
                                     prefix=f"{config.prefix}.x_embedder")
        self.time_embedder = TimestepEmbedder(inner_dim, act_layer="silu", frequency_embedding_size=256)

        self.context_norm = RMSNorm(config.text_embed_dim, eps=1e-5)
        self.context_proj = ReplicatedLinear(config.text_embed_dim,
                                             inner_dim,
                                             bias=True,
                                             prefix=f"{config.prefix}.context_proj")
        # Layerwise offload hooks the first nn.ModuleList child, which must be the main blocks.
        self.transformer_blocks = nn.ModuleList([
            PAN2TransformerBlock(inner_dim,
                                 config.num_attention_heads,
                                 config.attention_head_dim,
                                 config.mlp_ratio,
                                 config.context_mlp_ratio,
                                 supported_attention_backends=self._supported_attention_backends,
                                 prefix=f"{config.prefix}.transformer_blocks.{i}") for i in range(config.num_layers)
        ])

        # The refiner attends over the text alone; the packed-sequence VSA backend must never be selected for it.
        refiner_backends = tuple(backend for backend in self._supported_attention_backends
                                 if backend != AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3)
        self.context_refiner = nn.ModuleList([
            PAN2TokenRefinerBlock(inner_dim,
                                  config.num_attention_heads,
                                  config.attention_head_dim,
                                  config.refiner_mlp_ratio,
                                  eps=1e-5,
                                  supported_attention_backends=refiner_backends,
                                  prefix=f"{config.prefix}.context_refiner.{i}")
            for i in range(config.num_refiner_layers)
        ])
        self.context_type_embedding = nn.Parameter(torch.zeros(inner_dim))

        # A sparse (VSA) student shards the refined text across the sequence-parallel ranks; the dense transformer keeps
        # it replicated, as PAN 2 has always been sampled.
        self.shard_text = any(
            isinstance(block.attn.attn, DistributedAttention_VSA) for block in self.transformer_blocks)

        self.norm_out = PAN2AdaLayerNormOut(inner_dim, eps=1e-6, prefix=f"{config.prefix}.norm_out")
        # PDD students widen ``proj_out`` to ``pdd_steps`` heads, one per interval of a fixed fine time grid. The
        # sampler fuses one block of heads per forward (``fuse_pdd_block``), so the fused output keeps the released
        # width.
        self.pdd_steps: int | None = getattr(config, "pdd_steps", None)
        patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2] * self.out_channels
        if self.pdd_steps is None:
            self.proj_out = ReplicatedLinear(inner_dim, patch_dim, bias=True, prefix=f"{config.prefix}.proj_out")
        else:
            # ``proj_out`` rows read (C, p_t, p_h, p_w), so the grid goes in front of the channel axis.
            self.proj_out = PDDReplicatedLinear(inner_dim,
                                                patch_dim,
                                                grid_size=self.pdd_steps,
                                                channel_layout="channel_major",
                                                bias=True,
                                                prefix=f"{config.prefix}.proj_out")

        self.gradient_checkpointing = False
        # See set_fused_kernels; the pipeline turns it on with PAN2PipelineConfig.dit_fused_kernels.
        self.fused_kernels = False
        self.__post_init__()

    @property
    def pdd_linears(self) -> dict[str, PDDReplicatedLinear]:
        """The widened ``{"video": proj_out}`` head."""
        if self.pdd_steps is None:
            # Not AttributeError: nn.Module.__getattr__ would swallow it and report a misleading missing attribute.
            raise RuntimeError("This PAN2 transformer has no PDD heads (pdd_steps is None)")
        return {"video": self.proj_out}

    def _pdd_linear_for_parameter(self, name: str) -> PDDReplicatedLinear | None:
        if self.pdd_steps is None:
            return None
        head, _, _ = name.rpartition(".")
        return self.proj_out if head == "proj_out" else None

    @property
    def _fastvideo_shard_cache_key_extra(self) -> dict[str, Any] | None:
        """Keep widened-head shards out of the base model's shard cache entry."""
        return self.inference_config_overrides() or None

    def inference_config_overrides(self) -> dict[str, Any]:
        """Extra ``config.json`` keys a deployable checkpoint of this model needs."""
        return {} if self.pdd_steps is None else {"pdd_steps": self.pdd_steps}

    def inference_export_expected_shape(self, output_key: str, base_shape: tuple[int, ...]) -> tuple[int, ...]:
        """The PDD head shape for a narrow or already-widened base."""
        linear = self._pdd_linear_for_parameter(output_key)
        if linear is None:
            return tuple(base_shape)
        base_rows = int(base_shape[0])
        if base_rows == linear.output_size:
            return tuple(base_shape)
        if base_rows != linear.head_output_size:
            raise ValueError(f"PDD export base tensor {output_key!r} has {base_rows} output rows; expected "
                             f"{linear.head_output_size} single-head rows or {linear.output_size} "
                             f"Grid-{linear.grid_size} rows")
        return (linear.output_size, *tuple(base_shape[1:]))

    def preprocess_loaded_weight(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Expand a narrow (released) head to the PDD width; already-widened heads pass through unchanged."""
        linear = self._pdd_linear_for_parameter(name)
        return tensor if linear is None else linear.expand_loaded_tensor(name, tensor)

    @contextlib.contextmanager
    def fuse_pdd_block(
        self,
        start: int,
        end: int,
        integration_weights: Mapping[str, torch.Tensor],
        precision_decoding: torch.dtype,
    ) -> Iterator[None]:
        """Fuse fine-grid block ``[start, end)`` on the widened head for the enclosed forwards."""
        with fuse_pdd_heads(self.pdd_linears, start, end, integration_weights, precision_decoding):
            yield

    def set_fused_kernels(self, enabled: bool) -> None:
        """Run the blocks' fused Q/K RMSNorm + RoPE and LayerNorm + modulation ops (``pan2_fusions``).

        The ops launch Triton kernels on CUDA and run their PyTorch reference elsewhere. The forward reproduces the
        eager arithmetic up to FP32 summation order, except that each gated residual rounds once instead of twice.
        """
        enabled = bool(enabled)
        for module in self.modules():
            if isinstance(module, (PAN2TransformerBlock, PAN2Attention, PAN2AdaLayerNormZero, PAN2AdaLayerNormOut)):
                module.fused_kernels = enabled
        self.fused_kernels = enabled

    @staticmethod
    def _local_rotary_pos_embed(freqs_cis: tuple[torch.Tensor, torch.Tensor],
                                local_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The RoPE rows of this rank's video shard (identity rotation on the sequence-parallel padding)."""
        cos, sin = freqs_cis
        start = get_sp_parallel_rank() * local_len
        pad = start + local_len - cos.shape[0]
        if pad > 0:
            cos = F.pad(cos, (0, 0, 0, pad), value=1.0)
            sin = F.pad(sin, (0, 0, 0, pad), value=0.0)
        return cos[start:start + local_len], sin[start:start + local_len]

    def _rotary_pos_embed(self, num_frames: int, height: int, width: int,
                          device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Half-dim cos/sin of the 3-axis ``(t, h, w)`` RoPE over the full (unsharded) video token grid."""
        axes = [torch.arange(size, device=device, dtype=torch.float32) for size in (num_frames, height, width)]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        grid[0] = grid[0] * self.config.rope_temporal_scale
        cos_sin = [
            get_1d_rotary_pos_embed(dim, grid[i].reshape(-1), theta=self.config.rope_theta, use_real=False)
            for i, dim in enumerate(self.rope_axes_dim)
        ]
        freqs_cos = torch.cat([cos for cos, _ in cos_sin], dim=1).float()
        freqs_sin = torch.cat([sin for _, sin in cos_sin], dim=1).float()
        return freqs_cos, freqs_sin

    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor | list[torch.Tensor],
                timestep: torch.LongTensor,
                encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
                guidance=None,
                **kwargs) -> torch.Tensor:
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        freqs_cis = self._rotary_pos_embed(post_patch_num_frames, post_patch_height, post_patch_width,
                                           hidden_states.device)

        hidden_states = self.x_embedder(hidden_states)
        hidden_states, original_seq_len = sequence_model_parallel_shard(hidden_states, dim=1)
        local_freqs_cis = None
        if self.fused_kernels:
            local_freqs_cis = self._local_rotary_pos_embed(freqs_cis, hidden_states.shape[1])

        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        temb = self.time_embedder(timestep).to(hidden_states.dtype)

        encoder_hidden_states, _ = self.context_proj(self.context_norm(encoder_hidden_states))
        for refiner_block in self.context_refiner:
            encoder_hidden_states = refiner_block(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + self.context_type_embedding
        original_text_len = None
        if self.shard_text and get_sp_world_size() > 1:
            # Each rank keeps a shard of the refined caption, as of the video; the text output is never gathered,
            # since only the video tokens leave the last block.
            encoder_hidden_states, original_text_len = sequence_model_parallel_shard(encoder_hidden_states, dim=1)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, freqs_cis, original_seq_len, original_text_len,
                    local_freqs_cis)
            else:
                hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb, freqs_cis,
                                                             original_seq_len, original_text_len, local_freqs_cis)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = sequence_model_parallel_all_gather_with_unpad(hidden_states, original_seq_len, dim=1)
        hidden_states, _ = self.proj_out(hidden_states)

        # proj_out emits channels before the patch offsets: (C, p_t, p_h, p_w). A PDD student's fused block keeps that
        # width (``fuse_pdd_block``).
        hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width,
                                              -1, p_t, p_h, p_w)
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = PAN2Transformer3DModel
