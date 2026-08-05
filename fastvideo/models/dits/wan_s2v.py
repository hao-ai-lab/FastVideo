# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V-14B: audio-driven video generation.

Ported from the official native implementation (``Wan-Video/Wan2.2``,
``wan/modules/s2v/``). There is no diffusers implementation of S2V -- the PR
that would have added one (huggingface/diffusers#12258) was closed unmerged --
so the official repo is the only reference, and the checkpoint ships in native
Wan naming (``blocks.0.self_attn.q``) rather than diffusers naming.

Three things make S2V structurally different from every other Wan variant here,
and none of them can be expressed by configuring ``WanTransformer3DModel``:

1. **Heterogeneous sequence.** Tokens are ``[video | reference image | motion]``.
   Only the leading ``video_len`` video tokens are denoised and returned; the
   rest are conditioning context carried through the tower.
2. **Two-segment modulation.** With ``zero_timestep``, video tokens are modulated
   by the real timestep while ref/motion tokens are modulated by a fixed zero
   timestep -- they are already clean, so they must not be treated as noisy.
   Every modulation site in the block therefore splits at ``seg_idx``.
3. **Precomputed heterogeneous RoPE.** Each span gets its own positional range
   (motion frames sit at *negative* time offsets); frequencies are built once
   for the whole sequence rather than derived from a single grid.

Sequence parallelism is not wired up in this first version (upstream shards
``pre_compute_freqs`` alongside the hidden states); single-GPU and FSDP-style
weight sharding work. See the S2V section of the support matrix.
"""
import math
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from torch import nn

from fastvideo.attention import LocalAttention
from fastvideo.configs.models.dits.wan_s2v import WanS2VConfig
from fastvideo.layers.layernorm import FP32LayerNorm, RMSNorm
from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.layers.mlp import MLP
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.wan_s2v_audio import (S2V_ATTENTION_BACKENDS, AudioInjector, CausalAudioEncoder)

# Reference-image tokens are parked at this time index so they can never collide
# with a real video position in RoPE.
REF_TIME_INDEX = 30


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).float()


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_freqs(head_dim: int, max_seq_len: int = 1024) -> torch.Tensor:
    """(time, height, width) RoPE tables for one head; time absorbs the remainder of head_dim."""
    spatial = 2 * (head_dim // 6)
    return torch.cat([rope_params(max_seq_len, band) for band in (head_dim - 2 * spatial, spatial, spatial)], dim=1)


def _sample_positions(start: int, extent: int, count: int) -> list[int]:
    """``count`` integer positions evenly covering ``extent`` slots from ``start``; negative runs backwards."""
    step = 1 if extent >= 0 else -1
    return np.linspace(start, start + extent - step, count).astype(int).tolist()


def rope_precompute(x: torch.Tensor, grid_sizes: list, freqs: torch.Tensor) -> torch.Tensor:
    """Build per-token complex RoPE frequencies for a heterogeneous sequence.

    ``grid_sizes`` is a list of spans laid out back-to-back in sequence order;
    each span is ``[start, end, extent]`` where every element is a ``[B, 3]``
    tensor of (frame, height, width). A negative start frame means the span sits
    in the *past* (motion frames), which is encoded by walking the time axis
    backwards and conjugating the temporal band rather than by indexing
    negatively.
    """
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2
    freqs_t, freqs_h, freqs_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))

    offset = 0
    for span_start, span_end, extent in grid_sizes:
        seq_len = 0
        for i in range(span_start.shape[0]):
            f_o, h_o, w_o = span_start[i]
            t_f, t_h, t_w = extent[i]
            seq_f, seq_h, seq_w = (int(v) for v in span_end[i] - span_start[i])
            seq_len = seq_f * seq_h * seq_w
            if seq_len <= 0 or t_f <= 0:
                continue
            past = f_o < 0
            direction = -1 if past else 1
            f_idx = _sample_positions(direction * f_o.item(), direction * t_f.item(), seq_f)
            band_t = freqs_t[f_idx].conj() if past else freqs_t[f_idx]
            output[i, offset:offset + seq_len] = torch.cat([
                band_t.view(seq_f, 1, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                freqs_h[_sample_positions(h_o.item(), t_h.item(), seq_h)].view(1, seq_h, 1, -1).expand(
                    seq_f, seq_h, seq_w, -1),
                freqs_w[_sample_positions(w_o.item(), t_w.item(), seq_w)].view(1, 1, seq_w, -1).expand(
                    seq_f, seq_h, seq_w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)
        offset += seq_len
    return output


def rope_apply(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate q/k of shape [B, L, N, D] by precomputed complex frequencies."""
    x_complex = torch.view_as_complex(x.to(torch.float64).unflatten(3, (-1, 2)))
    return torch.view_as_real(x_complex * freqs[:, :x.size(1)]).flatten(3).float()


class WanS2VSelfAttention(nn.Module):
    """Self-attention over the full heterogeneous sequence, with precomputed RoPE."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = ReplicatedLinear(dim, dim)
        self.to_k = ReplicatedLinear(dim, dim)
        self.to_v = ReplicatedLinear(dim, dim)
        self.to_out = ReplicatedLinear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = LocalAttention(num_heads=num_heads, head_size=self.head_dim, dropout_rate=0,
                                   softmax_scale=None, causal=False,
                                   supported_attention_backends=S2V_ATTENTION_BACKENDS)

    def qkv(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.to_q(x)[0]).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)[0]).view(b, -1, n, d)
        v = self.to_v(context)[0].view(b, -1, n, d)
        return q, k, v

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x, x)
        q, k = rope_apply(q, freqs).to(v.dtype), rope_apply(k, freqs).to(v.dtype)
        return self.to_out(self.attn(q, k, v).flatten(2))[0]


class WanS2VCrossAttention(WanS2VSelfAttention):
    """Text cross-attention: same projections as self-attention, no RoPE."""

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.to_out(self.attn(*self.qkv(x, context)).flatten(2))[0]


def _segment_affine(x: torch.Tensor, seg: list[int], scale: torch.Tensor,
                    shift: torch.Tensor | None = None) -> torch.Tensor:
    """Apply a per-segment affine to a heterogeneous sequence.

    ``scale``/``shift`` carry a segment axis of size 2: segment 0 covers tokens
    ``[0, seg[1])`` (video, real timestep) and segment 1 covers the ref+motion
    tail (zero timestep). Modulation passes ``1 + scale``; gating passes the
    gate alone and no shift.
    """
    parts = [x[:, seg[i]:seg[i + 1]] * scale[:, i:i + 1] for i in range(2)]
    if shift is not None:
        parts = [part + shift[:, i:i + 1] for i, part in enumerate(parts)]
    return torch.cat(parts, dim=1)


class WanS2VAttentionBlock(nn.Module):
    """One of the 40 blocks.

    Cannot inherit ``WanTransformerBlock``: every modulation site here is
    segment-aware (see the module docstring).
    """

    def __init__(self, dim: int, ffn_dim: int, num_heads: int, qk_norm: bool = True,
                 cross_attn_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        # fp32 like norm1/norm2: upstream's WanLayerNorm computes in fp32 and casts
        # back, and FastVideo's own Wan port does the same for this norm.
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.self_attn = WanS2VSelfAttention(dim, num_heads, qk_norm, eps)
        self.cross_attn = WanS2VCrossAttention(dim, num_heads, qk_norm, eps)
        self.ffn = MLP(dim, ffn_dim, act_type="gelu_pytorch_tanh", bias=True)
        # Native Wan calls this "modulation"; diffusers calls it scale_shift_table.
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: tuple[torch.Tensor, int], context: torch.Tensor,
                freqs: torch.Tensor) -> torch.Tensor:
        e_tensor, seg_idx = e
        seg = [0, min(max(0, int(seg_idx)), x.size(1)), x.size(1)]
        # modulation is [1, 6, dim] and e_tensor [B, 6, 2, dim]; the extra axis is
        # the segment axis, so each chunk comes out as [B, 2, dim].
        shift_msa, scale_msa, gate_msa, c_shift, c_scale, c_gate = [
            chunk.squeeze(1) for chunk in (self.modulation.unsqueeze(2) + e_tensor).float().chunk(6, dim=1)
        ]

        y = self.self_attn(_segment_affine(self.norm1(x).float(), seg, 1 + scale_msa, shift_msa).to(x.dtype), freqs)
        x = x + _segment_affine(y.float(), seg, gate_msa).to(x.dtype)

        x = x + self.cross_attn(self.norm3(x), context)

        y = self.ffn(_segment_affine(self.norm2(x).float(), seg, 1 + c_scale, c_shift).to(x.dtype))
        return x + _segment_affine(y.float(), seg, c_gate).to(x.dtype)


class HeadS2V(nn.Module):
    """Final norm + projection back to patch space."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation + e.unsqueeze(1)).float().chunk(2, dim=1)
        return self.head((self.norm(x).float() * (1 + scale) + shift).to(x.dtype))


class FramePackMotioner(nn.Module):
    """Compress past motion frames at three temporal scales.

    Recent frames keep full detail (``proj``), older ones are downsampled 2x
    (``proj_2x``) and 4x (``proj_4x``) -- more history for fewer tokens, the
    same trade a video codec makes. Buckets are [nearest, mid, farthest].
    """

    def __init__(self, inner_dim: int = 5120, num_heads: int = 40,
                 zip_frame_buckets: tuple[int, int, int] = (1, 2, 16),
                 drop_mode: str = "drop") -> None:
        super().__init__()
        assert inner_dim % num_heads == 0 and (inner_dim // num_heads) % 2 == 0
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        # Plain ints, not a tensor: the loader builds this model under
        # torch.device("meta"), where any tensor attribute becomes unreadable
        # (.item()/.sum() raise) and only registered buffers can be rematerialised.
        self.zip_frame_buckets = tuple(zip_frame_buckets)
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.drop_mode = drop_mode
        # Non-persistent: derived from config, never stored in the checkpoint.
        self.register_buffer("freqs", rope_freqs(inner_dim // num_heads), persistent=False)

    def forward(self, motion_latents: list[torch.Tensor],
                add_last_motion: int = 2) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        total_frames = sum(self.zip_frame_buckets)
        head_dim = self.inner_dim // self.num_heads
        tokens: list[torch.Tensor] = []
        ropes: list[torch.Tensor] = []
        for m in motion_latents:
            lat_h, lat_w = m.shape[2], m.shape[3]
            padded = torch.zeros(self.proj.in_channels, total_frames, lat_h, lat_w, device=m.device, dtype=m.dtype)
            overlap = min(total_frames, m.shape[1])
            if overlap > 0:
                padded[:, -overlap:] = m[:, -overlap:]
            if add_last_motion < 2 and self.drop_mode != "drop":
                # "padd" mode keeps the token count fixed and zeroes the dropped buckets instead.
                zero_end = sum(self.zip_frame_buckets[:len(self.zip_frame_buckets) - add_last_motion - 1])
                padded[:, -zero_end:] = 0

            # split() consumes the buckets farthest-first, hence the reversed order.
            far, mid, near = padded.unsqueeze(0).split(self.zip_frame_buckets[::-1], dim=2)
            parts = [proj(latents).flatten(2).transpose(1, 2)
                     for proj, latents in ((self.proj, near), (self.proj_2x, mid), (self.proj_4x, far))]
            if add_last_motion < 2 and self.drop_mode == "drop":
                parts[0] = parts[0][:, :0]
                if add_last_motion < 1:
                    parts[1] = parts[1][:, :0]
            motion_lat = torch.cat(parts, dim=1)

            tokens.append(motion_lat)
            ropes.append(rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, head_dim),
                self._rope_grids(add_last_motion, lat_h, lat_w), self.freqs.to(m.device)))
        return tokens, ropes

    def _rope_grids(self, add_last_motion: int, lat_h: int, lat_w: int) -> list:
        """Positional spans for the three buckets, all at negative (past) time offsets."""
        z = self.zip_frame_buckets
        grids: list = []
        # Per bucket: temporal span after downsampling, spatial downsample factor.
        # The *extent* (third tensor) stays at the 1x grid for every bucket: it is
        # the range positions are spread over, so downsampled buckets must still
        # span the whole frame. Shrinking it too would collapse the 2x/4x history
        # into the top-left corner of the positional grid.
        for i, (span, div) in enumerate([(z[0], 2), (z[1] // 2, 4), (z[2] // 4, 8)]):
            if self.drop_mode == "drop" and add_last_motion < 2 - i:
                continue
            start = -sum(z[:i + 1])
            grids.append([
                torch.tensor([start, 0, 0]).unsqueeze(0),
                torch.tensor([start + span, lat_h // div, lat_w // div]).unsqueeze(0),
                torch.tensor([z[i], lat_h // 2, lat_w // 2]).unsqueeze(0),
            ])
        return grids


class WanS2VTransformer3DModel(BaseDiT):
    """Wan2.2-S2V-14B transformer."""

    _fsdp_shard_conditions = WanS2VConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = WanS2VConfig().arch_config._compile_conditions
    param_names_mapping = WanS2VConfig().arch_config.param_names_mapping
    reverse_param_names_mapping: dict = {}
    _supported_attention_backends = S2V_ATTENTION_BACKENDS

    def __init__(self, config: WanS2VConfig, hf_config: dict[str, Any], **kwargs) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch = config.arch_config
        dim = arch.hidden_size
        self.hidden_size = dim
        self.num_attention_heads = arch.num_attention_heads
        self.rope_max_seq_len = arch.rope_max_seq_len
        self.num_channels_latents = arch.num_channels_latents
        self.out_channels = arch.out_channels
        self.patch_size = arch.patch_size
        self.text_len = arch.text_len
        self.freq_dim = arch.freq_dim
        self.zero_timestep = arch.zero_timestep
        self.enable_adain = arch.enable_adain
        self.adain_mode = arch.adain_mode
        self.add_last_motion = arch.add_last_motion

        self.patch_embedding = nn.Conv3d(arch.in_channels, dim, kernel_size=arch.patch_size, stride=arch.patch_size)
        self.text_embedding = MLP(arch.text_dim, dim, output_dim=dim, act_type="gelu_pytorch_tanh", bias=True)
        self.time_embedding = MLP(arch.freq_dim, dim, output_dim=dim, act_type="silu", bias=True)
        # Named (not indexed) so the child is `time_projection.linear`, which is
        # what param_names_mapping targets; nn.Sequential would name it ".1".
        self.time_projection = nn.Sequential(
            OrderedDict([("act", nn.SiLU()), ("linear", ReplicatedLinear(dim, dim * 6))]))

        self.blocks = nn.ModuleList([
            WanS2VAttentionBlock(dim, arch.ffn_dim, arch.num_attention_heads, qk_norm=True,
                                 cross_attn_norm=arch.cross_attn_norm, eps=arch.eps)
            for _ in range(arch.num_layers)
        ])
        self.head = HeadS2V(dim, arch.out_channels, arch.patch_size, arch.eps)

        self.cond_encoder = None
        if arch.cond_dim > 0:
            self.cond_encoder = nn.Conv3d(arch.cond_dim, dim, kernel_size=arch.patch_size, stride=arch.patch_size)
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=arch.audio_dim, num_layers=25, out_dim=dim, num_token=arch.num_audio_token,
            need_global=arch.enable_adain)
        self.audio_injector = AudioInjector(
            dim=dim, num_heads=arch.num_attention_heads, inject_layers=arch.audio_inject_layers,
            enable_adain=arch.enable_adain, adain_dim=dim, eps=arch.eps)
        # 3 token kinds: 0 = noisy video, 1 = reference image, 2 = motion.
        self.trainable_cond_mask = nn.Embedding(3, dim)
        if arch.enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=dim, num_heads=arch.num_attention_heads, zip_frame_buckets=(1, 2, 16),
                drop_mode=arch.framepack_drop_mode)

        # Non-persistent: derived from config, absent from the checkpoint.
        self.register_buffer("freqs", rope_freqs(dim // arch.num_attention_heads, arch.rope_max_seq_len),
                             persistent=False)

    def materialize_non_persistent_buffers(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        """Rebuild the RoPE tables after meta-device construction.

        TransformerLoader builds the DiT under ``torch.device("meta")`` and then
        streams checkpoint weights in. Non-persistent buffers are not in the
        checkpoint, so they stay on meta until this hook (called by fsdp_load)
        recreates them with real storage. Complex dtype is deliberate -- these
        are rotation factors, not activations, and must not follow the model dtype.
        """
        head_dim = self.hidden_size // self.num_attention_heads
        if self.freqs.is_meta:
            self.freqs = rope_freqs(head_dim, self.rope_max_seq_len).to(device)
        packer = getattr(self, "frame_packer", None)
        if packer is not None and packer.freqs.is_meta:
            packer.freqs = rope_freqs(packer.inner_dim // packer.num_heads).to(device)

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        c = self.out_channels
        out = []
        for u, v in zip(x, grid_sizes.tolist(), strict=False):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            out.append(u.reshape(c, *[i * j for i, j in zip(v, self.patch_size, strict=False)]))
        return out

    def _embed_audio(self, audio_input: torch.Tensor,
                     motion_frames: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """wav2vec2 stack -> (per-frame tokens, per-frame global summary).

        The leading ``motion_frames[0]`` samples are repeated so that the audio
        covers the motion-frame prefix, then the prefix is dropped again in
        latent space (``motion_frames[1]``) to realign with the video latents.
        """
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        res = self.casual_audio_encoder(audio_input)
        if self.enable_adain:
            audio_emb_global, audio_emb = res
            return audio_emb[:, motion_frames[1]:], audio_emb_global[:, motion_frames[1]:]
        return res[:, motion_frames[1]:], None

    def _timestep_embedding(self, t: torch.Tensor,
                            video_len: int) -> tuple[torch.Tensor, tuple[torch.Tensor, int]]:
        """Return (head_emb, (block_emb, seg_idx)).

        With ``zero_timestep`` an extra t=0 entry is embedded and used for the
        ref/motion segment: those tokens are already clean, so modulating them
        with the noisy timestep would corrupt the conditioning.

        The projections themselves run in the model dtype and the results are
        upcast, matching ``wanvideo.py``. Upstream instead forces the whole
        block to fp32; all downstream modulation arithmetic here is fp32 either
        way, so only the two GEMMs differ.
        """
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(t.device)).float()
        e0 = self.time_projection(e)[0].unflatten(1, (6, self.hidden_size)).float()
        if not self.zero_timestep:
            return e, (e0.unsqueeze(2).repeat(1, 1, 2, 1), 0)
        e, zero_e0, e0 = e[:-1], e0[-1:], e0[:-1]
        e0 = torch.cat([e0.unsqueeze(2), zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)], dim=2)
        return e, (e0, video_len)

    @staticmethod
    def _as_sample_list(value: torch.Tensor | list[torch.Tensor] | None) -> list[torch.Tensor] | None:
        """Accept either a batched [B, C, T, H, W] tensor or a list of [C, T, H, W].

        DenoisingStage passes batched tensors; the reference implementation (and
        our own tests) pass lists, because each sample may have a different
        frame count. Normalising here keeps both callers working.
        """
        if value is None or isinstance(value, list):
            return value
        return list(value) if value.dim() == 5 else [value]

    def forward(self, hidden_states: torch.Tensor | list[torch.Tensor],
                encoder_hidden_states: torch.Tensor | list[torch.Tensor], timestep: torch.Tensor,
                ref_latents: torch.Tensor | list[torch.Tensor] | None = None,
                motion_latents: torch.Tensor | list[torch.Tensor] | None = None,
                cond_states: torch.Tensor | list[torch.Tensor] | None = None,
                audio_input: torch.Tensor | None = None,
                motion_frames: tuple[int, int] = (17, 5), add_last_motion: int = 2,
                drop_motion_frames: bool = False, **kwargs) -> list[torch.Tensor]:
        """Denoise one step of an audio-driven video.

        hidden_states  [B, C, T, H, W] or list of [C, T, H, W] noisy video latents
        ref_latents    reference-image latents; required, the model is image-conditioned
        motion_latents previously generated frames; None on the first clip
        cond_states    pose/control latents, or None when unused
        audio_input    [B, 25, C_a, T_a] stacked wav2vec2 hidden states; required

        Returns a batched [B, C, T, H, W] fp32 tensor when ``hidden_states`` came
        in batched (the DenoisingStage path -- CFG arithmetic and scheduler.step
        need a tensor), or a list of per-sample tensors when it came in as a
        list (the reference-implementation path).
        """
        batched_input = isinstance(hidden_states, torch.Tensor)
        hidden_states = self._as_sample_list(hidden_states)
        ref_latents = self._as_sample_list(ref_latents)
        motion_latents = self._as_sample_list(motion_latents)
        cond_states = self._as_sample_list(cond_states)
        if isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = list(encoder_hidden_states)
        elif isinstance(encoder_hidden_states, list) and encoder_hidden_states and \
                encoder_hidden_states[0].dim() == 3:
            # DenoisingStage passes a list with one batched [B, L, C] tensor per
            # text encoder (same convention wanvideo.py unwraps with [0]).
            encoder_hidden_states = list(encoder_hidden_states[0])
        if audio_input is None or ref_latents is None:
            raise ValueError(
                "Wan-S2V needs both a reference image and audio. Got "
                f"ref_latents={'set' if ref_latents is not None else 'None'}, "
                f"audio_input={'set' if audio_input is not None else 'None'}. The pipeline supplies "
                "these from batch.image_latent and batch.audio_embeds.")
        if motion_latents is None:
            # First clip: no history yet. The official runner drops motion tokens
            # here (drop_first_motion=True in its config); the zeros only exist so
            # the frame packer has an input to (then discard) -- their values are
            # never attended to.
            drop_motion_frames = True
            motion_latents = [torch.zeros_like(u[:, :1]) for u in hidden_states]
        if cond_states is None and self.cond_encoder is not None:
            # The reference always encodes a cond tensor (zeros when unused), and
            # cond_encoder has a bias -- skipping it entirely would shift every
            # video token relative to the official implementation.
            cond_states = [torch.zeros_like(u) for u in hidden_states]

        add_last_motion = int(self.add_last_motion) * add_last_motion
        audio_emb, audio_emb_global = self._embed_audio(audio_input, motion_frames)
        freqs = self.freqs.to(self.patch_embedding.weight.device)

        # 1. video tokens (+ pose conditioning added in patch space)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in hidden_states]
        if cond_states is not None and self.cond_encoder is not None:
            x = [x_ + self.cond_encoder(c.unsqueeze(0)) for x_, c in zip(x, cond_states, strict=False)]
        original_grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        video_len = x[0].size(1)
        grid_sizes = [[torch.zeros_like(original_grid_sizes), original_grid_sizes, original_grid_sizes]]

        # 2. reference-image tokens, parked far away in time (see REF_TIME_INDEX)
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        bsz, h, w = len(ref), ref[0].shape[3], ref[0].shape[4]
        grid_sizes.append([
            torch.tensor([REF_TIME_INDEX, 0, 0]).unsqueeze(0).repeat(bsz, 1),
            torch.tensor([REF_TIME_INDEX + 1, h, w]).unsqueeze(0).repeat(bsz, 1),
            torch.tensor([1, h, w]).unsqueeze(0).repeat(bsz, 1),
        ])
        x = [torch.cat([u, r.flatten(2).transpose(1, 2)], dim=1) for u, r in zip(x, ref, strict=False)]

        # 3. token-kind mask: 0 video, 1 reference (2 = motion, tagged on append)
        mask = [torch.zeros([1, u.shape[1]], dtype=torch.long, device=u.device) for u in x]
        for m in mask:
            m[:, video_len:] = 1

        # 4. RoPE for video+ref, then append motion tokens (which carry their own)
        stacked = torch.cat(x)
        rope = rope_precompute(
            stacked.detach().view(stacked.size(0), stacked.size(1), self.num_attention_heads,
                                  self.hidden_size // self.num_attention_heads), grid_sizes, freqs)
        x, rope = [u.unsqueeze(0) for u in stacked], [u.unsqueeze(0) for u in rope]
        x, rope, mask = self._inject_motion(x, rope, mask, motion_latents, drop_motion_frames, add_last_motion)

        x = torch.cat(x, dim=0)
        rope = torch.cat(rope, dim=0)
        x = x + self.trainable_cond_mask(torch.cat(mask, dim=0)).to(x.dtype)

        # 5. conditioning embeddings
        e, block_e = self._timestep_embedding(timestep, video_len)
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in encoder_hidden_states
            ]))

        # 6. the tower, with audio re-injected at 12 of the 40 blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, block_e, context, rope)
            x = self.audio_injector(x, idx, audio_emb, audio_emb_global, video_len, self.adain_mode)

        # 7. only the video span is denoised output; ref/motion are context
        out = [u.float() for u in self.unpatchify(self.head(x[:, :video_len], e), original_grid_sizes)]
        # Batched callers (DenoisingStage) need a tensor back for CFG arithmetic
        # and scheduler.step; list callers may have per-sample shapes, keep lists.
        return torch.stack(out) if batched_input else out

    def _inject_motion(self, x: list[torch.Tensor], rope: list[torch.Tensor], mask: list[torch.Tensor],
                       motion_latents: list[torch.Tensor], drop_motion_frames: bool,
                       add_last_motion: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Append compressed motion-frame tokens and tag them as kind 2."""
        if drop_motion_frames and getattr(self, "frame_packer", None) is None:
            # Only the framepack motion path is implemented; without it, dropped
            # motion is simply no tokens. Asking for *kept* motion without a
            # packer is a config error and should fail below, not silently skip.
            return x, rope, mask
        mot, mot_rope = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            mot, mot_rope = [m[:, :0] for m in mot], [m[:, :0] for m in mot_rope]
        if not mot or mot[0].size(1) == 0:
            return x, rope, mask
        x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot, strict=False)]
        rope = [torch.cat([u, m], dim=1) for u, m in zip(rope, mot_rope, strict=False)]
        mask = [
            torch.cat([m, 2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype)], dim=1)
            for m, u in zip(mask, x, strict=False)
        ]
        return x, rope, mask
