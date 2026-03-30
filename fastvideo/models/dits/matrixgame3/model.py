# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn

from fastvideo.attention import DistributedAttention
from fastvideo.configs.models.dits.matrixgame import MatrixGame3WanVideoConfig
from fastvideo.layers.layernorm import (
    FP32LayerNorm,
    LayerNormScaleShift,
    RMSNorm,
    ScaleResidual,
    ScaleResidualLayerNormScaleShift,
)
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.layers.visual_embedding import (
    PatchEmbed,
    TimestepEmbedder,
    ModulateProjection,
    WanCamControlPatchEmbedding,
)
from fastvideo.logger import init_logger
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wanvideo import (
    WanSelfAttention,
    WanI2VCrossAttention,
    WanT2VCrossAttention,
    WanImageEmbedding,
)
from fastvideo.platforms import AttentionBackendEnum

# Import ActionModule
from .action_module import MatrixGame3ActionModule

logger = init_logger(__name__)


def _build_rope_freqs(
    *,
    max_seq_len: int,
    head_dim: int,
    num_heads: int,
    sigma_theta: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    c = head_dim // 2
    c_t = c - 2 * (c // 3)
    c_h = c // 3
    c_w = c // 3

    if sigma_theta > 0:
        rope_epsilon = torch.linspace(-1, 1, num_heads, dtype=torch.float64, device=device)
        theta_hat = 10000.0 * (1 + sigma_theta * rope_epsilon)

        def build_axis_freqs(seq_len: int, c_part: int) -> torch.Tensor:
            exp = torch.arange(c_part, dtype=torch.float64, device=device) / c_part
            omega = 1.0 / torch.pow(theta_hat.unsqueeze(1), exp.unsqueeze(0))
            pos = torch.arange(seq_len, dtype=torch.float64, device=device)
            angles = pos.view(1, -1, 1) * omega.unsqueeze(1)
            return torch.polar(torch.ones_like(angles), angles)

        return torch.cat(
            [
                build_axis_freqs(max_seq_len, c_t),
                build_axis_freqs(max_seq_len, c_h),
                build_axis_freqs(max_seq_len, c_w),
            ],
            dim=2,
        )

    def build_axis_freqs(seq_len: int, c_part: int) -> torch.Tensor:
        exp = torch.arange(c_part, dtype=torch.float64, device=device) / c_part
        omega = 1.0 / torch.pow(torch.tensor(10000.0, dtype=torch.float64, device=device), exp)
        pos = torch.arange(seq_len, dtype=torch.float64, device=device)
        angles = pos[:, None] * omega[None, :]
        return torch.polar(torch.ones_like(angles), angles)

    return torch.cat(
        [
            build_axis_freqs(max_seq_len, c_t),
            build_axis_freqs(max_seq_len, c_h),
            build_axis_freqs(max_seq_len, c_w),
        ],
        dim=1,
    )


@torch.amp.autocast("cuda", enabled=False)
def _apply_rope_with_frame_indices(
    x: torch.Tensor,
    freqs: torch.Tensor,
    *,
    height: int,
    width: int,
    frame_indices: list[int] | tuple[int, ...] | torch.Tensor,
) -> torch.Tensor:
    num_heads = x.shape[2]
    half_dim = x.shape[-1] // 2
    c_t = half_dim - 2 * (half_dim // 3)
    c_h = half_dim // 3
    c_w = half_dim // 3

    if not torch.is_tensor(frame_indices):
        frame_indices = torch.tensor(frame_indices, device=freqs.device, dtype=torch.long)
    else:
        frame_indices = frame_indices.to(device=freqs.device, dtype=torch.long)

    f = int(frame_indices.numel())
    seq_len = f * height * width
    x_complex = torch.view_as_complex(x.to(torch.float32).reshape(x.shape[0], seq_len, num_heads, -1, 2))

    if freqs.dim() == 3:
        t_freqs = freqs[:, frame_indices, :c_t]
        h_freqs = freqs[:, :height, c_t:c_t + c_h]
        w_freqs = freqs[:, :width, c_t + c_h:c_t + c_h + c_w]
        freqs_i = torch.cat(
            [
                t_freqs.permute(1, 0, 2).view(f, 1, 1, num_heads, -1).expand(f, height, width, num_heads, -1),
                h_freqs.permute(1, 0, 2).view(1, height, 1, num_heads, -1).expand(f, height, width, num_heads, -1),
                w_freqs.permute(1, 0, 2).view(1, 1, width, num_heads, -1).expand(f, height, width, num_heads, -1),
            ],
            dim=-1,
        ).reshape(seq_len, num_heads, -1)
    else:
        t_freqs = freqs[frame_indices, :c_t]
        h_freqs = freqs[:height, c_t:c_t + c_h]
        w_freqs = freqs[:width, c_t + c_h:c_t + c_h + c_w]
        freqs_i = torch.cat(
            [
                t_freqs.view(f, 1, 1, -1).expand(f, height, width, -1),
                h_freqs.view(1, height, 1, -1).expand(f, height, width, -1),
                w_freqs.view(1, 1, width, -1).expand(f, height, width, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

    out = torch.view_as_real(x_complex * freqs_i.unsqueeze(0).to(x_complex.dtype)).flatten(3)
    return out.float()


class MatrixGame3TimeImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        image_embed_dim: int | None = None,
    ):
        super().__init__()

        self.time_embedder = TimestepEmbedder(
            dim, frequency_embedding_size=time_freq_dim, act_layer="silu"
        )
        self.time_modulation = ModulateProjection(
            dim, factor=6, act_layer="silu"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        temb = self.time_embedder(timestep, timestep_seq_len)
        timestep_proj = self.time_modulation(temb)

        if encoder_hidden_states_image is not None:
            assert self.image_embedder is not None
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        if encoder_hidden_states is None:
            batch_size = temb.shape[0] if temb.dim() > 1 else timestep.shape[0]
            encoder_hidden_states = torch.zeros(
                (batch_size, 0, temb.shape[-1]),
                device=temb.device,
                dtype=temb.dtype,
            )

        return (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        )


class MatrixGame3CrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens=None, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C] - typically 257 image tokens
            context_lens(Tensor): Shape [B]
            crossattn_cache(dict): Optional cache for k/v during inference
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q_input = x.to(self.to_q.weight.dtype)
        q = self.norm_q(self.to_q(q_input)[0]).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                context_input = context.to(self.to_k.weight.dtype)
                k = self.norm_k(self.to_k(context_input)[0]).view(b, -1, n, d)
                v = self.to_v(context_input)[0].view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            context_input = context.to(self.to_k.weight.dtype)
            k = self.norm_k(self.to_k(context_input)[0]).view(b, -1, n, d)
            v = self.to_v(context_input)[0].view(b, -1, n, d)

        # compute attention
        x = self.attn(q, k, v)

        # output
        x = x.flatten(2)
        x, _ = self.to_out(x)
        return x


class MatrixGame3TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        supported_attention_backends: tuple[AttentionBackendEnum, ...]
        | None = None,
        prefix: str = "",
        action_config: dict | None = None,
        block_id: int | None = None,
        use_memory: bool = False,
    ):
        super().__init__()
        action_config = action_config or {}
        self.use_memory = use_memory

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.to_q = ReplicatedLinear(dim, dim, bias=True)
        self.to_k = ReplicatedLinear(dim, dim, bias=True)
        self.to_v = ReplicatedLinear(dim, dim, bias=True)

        self.to_out = ReplicatedLinear(dim, dim, bias=True)
        self.attn1 = DistributedAttention(
            num_heads=num_heads,
            head_size=dim // num_heads,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            prefix=f"{prefix}.attn1",
        )
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
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        # 2. Cross-attention
        if added_kv_proj_dim is not None:
            # I2V
            self.attn2 = WanI2VCrossAttention(
                dim, num_heads, qk_norm=qk_norm, eps=eps
            )
        else:
            # T2V
            self.attn2 = MatrixGame3CrossAttention(
                dim, num_heads, qk_norm=qk_norm, eps=eps
            )
        self.cross_attn_residual_norm = ScaleResidualLayerNormScaleShift(
            dim,
            norm_type="layer",
            eps=eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )

        # 2.1. Action Module Integration
        enabled_blocks = set(action_config.get("blocks", [])) if action_config else set()
        self.use_action_module = len(action_config) > 0 and (block_id is None or block_id in enabled_blocks)
        if self.use_action_module:
            self.action_model = MatrixGame3ActionModule(**action_config)
        else:
            self.action_model = None

        # 3. Feed-forward
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh")
        self.mlp_residual = ScaleResidual()

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        if self.use_memory:
            self.cam_injector_layer1 = nn.Linear(dim, dim)
            self.cam_injector_layer2 = nn.Linear(dim, dim)
            self.cam_scale_layer = nn.Linear(dim, dim)
            self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs: torch.Tensor,
        # Action Module specific args
        grid_sizes: torch.Tensor,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        mouse_cond_memory: torch.Tensor | None = None,
        keyboard_cond_memory: torch.Tensor | None = None,
        plucker_emb: torch.Tensor | None = None,
        memory_length: int = 0,
        memory_latent_idx: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        predict_latent_idx: tuple[int, int] | list[int] | tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        orig_dtype = hidden_states.dtype

        if temb.dim() == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(
                6, dim=2
            )
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            e = self.scale_shift_table + temb.float()
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = e.chunk(6, dim=1)
        assert shift_msa.dtype == torch.float32

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states).float() * (1 + scale_msa) + shift_msa
        ).to(orig_dtype)
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

        # Apply rotary embeddings
        grid_frames = int(grid_sizes[0].item())
        grid_height = int(grid_sizes[1].item())
        grid_width = int(grid_sizes[2].item())

        if memory_length > 0:
            hw = grid_height * grid_width
            query_memory = query[:, :memory_length * hw]
            key_memory = key[:, :memory_length * hw]
            query_pred = query[:, memory_length * hw:]
            key_pred = key[:, memory_length * hw:]

            mem_indices = memory_latent_idx
            if mem_indices is None:
                mem_indices = list(range(memory_length))

            pred_indices = predict_latent_idx
            if isinstance(pred_indices, tuple) and len(pred_indices) == 2:
                pred_indices = list(range(pred_indices[0], pred_indices[1]))
            elif pred_indices is None:
                pred_indices = list(range(grid_frames - memory_length))

            query_memory = _apply_rope_with_frame_indices(
                query_memory, freqs, height=grid_height, width=grid_width, frame_indices=mem_indices
            )
            key_memory = _apply_rope_with_frame_indices(
                key_memory, freqs, height=grid_height, width=grid_width, frame_indices=mem_indices
            )
            query_pred = _apply_rope_with_frame_indices(
                query_pred, freqs, height=grid_height, width=grid_width, frame_indices=pred_indices
            )
            key_pred = _apply_rope_with_frame_indices(
                key_pred, freqs, height=grid_height, width=grid_width, frame_indices=pred_indices
            )
            query = torch.cat([query_memory, query_pred], dim=1)
            key = torch.cat([key_memory, key_pred], dim=1)
        else:
            pred_indices = predict_latent_idx
            if isinstance(pred_indices, tuple) and len(pred_indices) == 2:
                pred_indices = list(range(pred_indices[0], pred_indices[1]))

            if pred_indices is None:
                pred_indices = list(range(grid_frames))
            query = _apply_rope_with_frame_indices(
                query, freqs, height=grid_height, width=grid_width, frame_indices=pred_indices
            )
            key = _apply_rope_with_frame_indices(
                key, freqs, height=grid_height, width=grid_width, frame_indices=pred_indices
            )

        attn_output, _ = self.attn1(query, key, value)
        attn_output = attn_output.flatten(2)
        attn_output = attn_output.to(self.to_out.weight.dtype)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        hidden_states = hidden_states.float() + attn_output.float() * gate_msa.float()
        if self.use_memory and plucker_emb is not None:
            c2ws_hidden_states = self.cam_injector_layer2(
                torch.nn.functional.silu(self.cam_injector_layer1(plucker_emb))
            )
            c2ws_hidden_states = c2ws_hidden_states + plucker_emb
            cam_scale = self.cam_scale_layer(c2ws_hidden_states)
            cam_shift = self.cam_shift_layer(c2ws_hidden_states)
            hidden_states = (1.0 + cam_scale) * hidden_states + cam_shift
        norm_hidden_states = self.self_attn_residual_norm.norm(hidden_states)

        # 2. Cross-attention
        attn_output = self.attn2(
            norm_hidden_states, context=encoder_hidden_states, context_lens=None
        )

        # - use_memory / action path: x = norm3(x); x = x + cross_attn(x)
        # - plain path:              x = x + cross_attn(norm3(x))
        cross_attn_residual_base = (
            norm_hidden_states if (mouse_cond is not None or self.use_memory) else hidden_states
        )
        hidden_states = cross_attn_residual_base + attn_output
        norm_hidden_states = self.cross_attn_residual_norm.norm(
            hidden_states.float()
        )
        norm_hidden_states = norm_hidden_states * (1 + c_scale_msa) + c_shift_msa

        # ================= Action Module =================
        if self.action_model is not None:
            if mouse_cond is not None or keyboard_cond is not None:
                action_dtype = self.ffn.fc_in.weight.dtype
                # grid_sizes is expected to be [F, H, W]
                # ActionModule implementation takes hidden_states directly
                hidden_states = self.action_model(
                    hidden_states.to(action_dtype),
                    int(grid_sizes[0]),
                    int(grid_sizes[1]),
                    int(grid_sizes[2]),
                    mouse_cond,
                    keyboard_cond,
                    mouse_cond_memory=mouse_cond_memory,
                    keyboard_cond_memory=keyboard_cond_memory,
                    num_frame_per_block=int(grid_sizes[0]),
                )

                norm_hidden_states = self.cross_attn_residual_norm.norm(
                    hidden_states.float()
                )
                norm_hidden_states = norm_hidden_states * (1 + c_scale_msa) + c_shift_msa
        # =================================================

        # 3. Feed-forward
        ff_output = self.ffn(norm_hidden_states.to(self.ffn.fc_in.weight.dtype))
        hidden_states = self.mlp_residual(hidden_states, ff_output, c_gate_msa)
        hidden_states = hidden_states.to(orig_dtype)

        return hidden_states


_DEFAULT_MATRIXGAME3_CONFIG = MatrixGame3WanVideoConfig()


class MatrixGame3WanModel(BaseDiT):
    # Marker for action input support (Matrix-Game)
    supports_action_input = True

    _fsdp_shard_conditions = _DEFAULT_MATRIXGAME3_CONFIG._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_MATRIXGAME3_CONFIG._compile_conditions
    _supported_attention_backends = (
        _DEFAULT_MATRIXGAME3_CONFIG._supported_attention_backends
    )
    param_names_mapping = _DEFAULT_MATRIXGAME3_CONFIG.param_names_mapping
    reverse_param_names_mapping = (
        _DEFAULT_MATRIXGAME3_CONFIG.reverse_param_names_mapping
    )
    lora_param_names_mapping = (
        _DEFAULT_MATRIXGAME3_CONFIG.lora_param_names_mapping
    )

    def __init__(
        self,
        config: MatrixGame3WanVideoConfig,
        hf_config: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.out_channels
        self.patch_size = config.patch_size

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=config.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = MatrixGame3TimeImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            image_embed_dim=(config.image_dim if config.image_dim > 0 else None),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, inner_dim),
        )
        self.use_memory = getattr(config, "use_memory", False)
        self.sigma_theta = float(getattr(config, "sigma_theta", 0.0))
        head_dim = inner_dim // self.num_attention_heads
        self.register_buffer(
            "freqs",
            _build_rope_freqs(
                max_seq_len=(2048 if self.use_memory else 1024),
                head_dim=head_dim,
                num_heads=self.num_attention_heads,
                sigma_theta=(self.sigma_theta if self.use_memory else 0.0),
            ),
            persistent=False,
        )
        self.camera_patch_embedding = None
        self.c2ws_hidden_states_layer1 = None
        self.c2ws_hidden_states_layer2 = None
        if self.use_memory:
            camera_in_channels = getattr(config, "camera_embed_in_channels", 1536)
            self.camera_patch_embedding = WanCamControlPatchEmbedding(
                patch_size=config.patch_size,
                in_chans=camera_in_channels,
                embed_dim=inner_dim,
            )
            self.c2ws_hidden_states_layer1 = nn.Linear(inner_dim, inner_dim)
            self.c2ws_hidden_states_layer2 = nn.Linear(inner_dim, inner_dim)

        # 2.1. Get action config
        self.action_config = getattr(config, "action_config", {})

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                MatrixGame3TransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{getattr(config, 'prefix', 'Wan')}.blocks.{i}",
                    action_config=self.action_config,
                    block_id=i,
                    use_memory=self.use_memory,
                )
                for i in range(config.num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, config.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor
        | list[torch.Tensor]
        | None = None,
        # Action inputs
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        x_memory: torch.Tensor | None = None,
        timestep_memory: torch.Tensor | None = None,
        mouse_cond_memory: torch.Tensor | None = None,
        keyboard_cond_memory: torch.Tensor | None = None,
        c2ws_plucker_emb: torch.Tensor | None = None,
        memory_latent_idx: list[int] | tuple[int, ...] | torch.Tensor | None = None,
        predict_latent_idx: tuple[int, int] | list[int] | tuple[int, ...] | torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None and not isinstance(
            encoder_hidden_states, torch.Tensor
        ):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        memory_length = 0
        if x_memory is not None:
            memory_length = x_memory.shape[2]
            hidden_states = torch.cat([x_memory.to(hidden_states.dtype), hidden_states], dim=2)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        max_frame_index = post_patch_num_frames
        if isinstance(predict_latent_idx, tuple) and len(predict_latent_idx) == 2:
            max_frame_index = max(max_frame_index, int(predict_latent_idx[1]))
        elif predict_latent_idx is not None:
            max_frame_index = max(max_frame_index, int(max(predict_latent_idx)) + 1)
        if memory_latent_idx is not None and len(memory_latent_idx) > 0:
            max_frame_index = max(max_frame_index, int(max(memory_latent_idx)) + 1)

        freqs = self.freqs.to(hidden_states.device)
        max_seq_len = freqs.shape[1] if freqs.dim() == 3 else freqs.shape[0]
        if max_frame_index > max_seq_len or post_patch_height > max_seq_len or post_patch_width > max_seq_len:
            raise ValueError(
                f"got frames={max_frame_index}, height={post_patch_height}, width={post_patch_width}, "
                f"max_seq_len={max_seq_len}"
            )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        plucker_emb = None
        if c2ws_plucker_emb is not None and self.camera_patch_embedding is not None:
            if memory_length > 0 and c2ws_plucker_emb.shape[2] == post_patch_num_frames - memory_length:
                zeros = torch.zeros(c2ws_plucker_emb.shape[0],
                                    c2ws_plucker_emb.shape[1],
                                    memory_length,
                                    c2ws_plucker_emb.shape[3],
                                    c2ws_plucker_emb.shape[4],
                                    device=c2ws_plucker_emb.device,
                                    dtype=c2ws_plucker_emb.dtype)
                c2ws_plucker_emb = torch.cat([zeros, c2ws_plucker_emb], dim=2)
            target_frames = post_patch_num_frames * p_t
            target_height = post_patch_height * p_h
            target_width = post_patch_width * p_w
            if c2ws_plucker_emb.shape[2] > target_frames or c2ws_plucker_emb.shape[3] > target_height or c2ws_plucker_emb.shape[4] > target_width:
                c2ws_plucker_emb = c2ws_plucker_emb[:, :, :target_frames, :target_height, :target_width]
            if c2ws_plucker_emb.shape[2:] != (target_frames, target_height, target_width):
                raise ValueError(
                    "Camera conditioning must align with the patchified latent grid, "
                    f"got camera shape {tuple(c2ws_plucker_emb.shape[2:])} vs target "
                    f"{(target_frames, target_height, target_width)}"
                )
            plucker_emb = self.camera_patch_embedding(
                c2ws_plucker_emb.to(hidden_states.dtype)
            )
            plucker_hidden = self.c2ws_hidden_states_layer2(
                torch.nn.functional.silu(self.c2ws_hidden_states_layer1(plucker_emb))
            )
            plucker_emb = plucker_emb + plucker_hidden

        timestep_tokens = timestep
        if timestep_tokens.dim() == 0:
            timestep_tokens = timestep_tokens.unsqueeze(0)
        if timestep_tokens.dim() == 1:
            timestep_tokens = timestep_tokens.unsqueeze(1).repeat(
                1, post_patch_num_frames * post_patch_height * post_patch_width
            )
        elif timestep_tokens.dim() == 2 and timestep_tokens.shape[1] == num_frames:
            timestep_tokens = (
                timestep_tokens[:, :, None, None]
                .repeat(1, 1, post_patch_height, post_patch_width)
                .reshape(timestep_tokens.shape[0], -1)
            )
        if memory_length > 0:
            if timestep_memory is None:
                raise ValueError("MatrixGame3 requires timestep_memory when x_memory is provided")
            timestep_tokens = torch.cat(
                [timestep_memory.to(timestep_tokens.dtype), timestep_tokens], dim=1
            )

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep_tokens.flatten(),
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=timestep_tokens.shape[1],
        )
        if timestep_proj.dim() == 3 and timestep_proj.shape[-1] % 6 == 0:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_hidden_states = encoder_hidden_states[0]
            elif encoder_hidden_states.ndim == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
            if (
                encoder_hidden_states.ndim == 3
                and encoder_hidden_states.shape[-1] == self.text_embedding[0].in_features
            ):
                encoder_hidden_states = self.text_embedding(
                    encoder_hidden_states.to(hidden_states.dtype)
                )
        else:
            # encoder_hidden_states is None (e.g. no text encoder)
            # MatrixGame uses image-action cross-attn.
            pass

        if encoder_hidden_states_image is not None:
            if encoder_hidden_states is not None:
                encoder_hidden_states = torch.concat(
                    [encoder_hidden_states_image, encoder_hidden_states], dim=1
                )
            else:
                encoder_hidden_states = encoder_hidden_states_image

        # This is [F, H, W] for the ActionModule
        grid_sizes = torch.tensor(
            [post_patch_num_frames, post_patch_height, post_patch_width],
            device=hidden_states.device,
        )

        # Blocks
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs,
                    grid_sizes=grid_sizes,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                    mouse_cond_memory=mouse_cond_memory,
                    keyboard_cond_memory=keyboard_cond_memory,
                    plucker_emb=plucker_emb,
                    memory_length=memory_length,
                    memory_latent_idx=memory_latent_idx,
                    predict_latent_idx=predict_latent_idx,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    freqs,
                    grid_sizes=grid_sizes,
                    mouse_cond=mouse_cond,
                    keyboard_cond=keyboard_cond,
                    mouse_cond_memory=mouse_cond_memory,
                    keyboard_cond_memory=keyboard_cond_memory,
                    plucker_emb=plucker_emb,
                    memory_length=memory_length,
                    memory_latent_idx=memory_latent_idx,
                    predict_latent_idx=predict_latent_idx,
                )

        # Output
        if temb.dim() == 3:
            shift, scale = (
                self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)
            ).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(
                2, dim=1
            )
        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if memory_length > 0:
            output = output[:, :, memory_length:]

        return output
