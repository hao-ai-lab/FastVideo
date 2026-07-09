# SPDX-License-Identifier: Apache-2.0
"""Training-time attention spec for causal video DiTs.

The rolling KV cache used at inference (``sink_size`` pinned frames plus a
trailing ``local_attn_size - sink_size`` frame window, optionally with
relativistic RoPE re-indexing) must be mirrored by the full-sequence attention
pattern used in teacher-forcing / causal-distillation training, otherwise the
student trains against a context layout it never sees when streaming.

This module is the single source of truth for that pattern:

- frame-level visibility rules shared by the FlexAttention mask builders, the
  Triton kernel and the pure-PyTorch reference implementation;
- ``CausalTrainAttentionPlan``: a lightweight, cacheable description of one
  training attention layout (``blockwise`` or ``teacher_forcing``);
- the relativistic sink RoPE correction. Within the rolling window,
  relativistic re-indexing shifts every position by the same amount, so
  absolute training RoPE already matches it exactly (RoPE phases depend only
  on position differences). The only pairs whose phase differs are
  query -> sink pairs once the window has scrolled past the sink: streaming
  inference re-indexes the sink to a fixed small distance while absolute
  training RoPE lets it recede. Matching that in training requires the query
  to be rotated back by ``delta = max(0, block_end - local_attn_size)``
  frames for sink columns only - a per-query-block key repositioning that
  FlexAttention cannot express. The Triton / reference implementations apply
  it exactly via ``delta_cos`` / ``delta_sin``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)


def blockwise_frame_visible(
    q_frame: int,
    kv_frame: int,
    *,
    num_frame_per_block: int,
    local_attn_size: int,
    sink_size: int,
) -> bool:
    """Frame-level visibility for the blockwise-causal (single-sequence) layout.

    ``q_frame`` attends everything up to the end of its own block, restricted
    (when ``local_attn_size != -1``) to the sink frames plus the trailing
    ``local_attn_size - sink_size`` frame window - the same set a streaming
    step reads from the rolled KV cache.
    """
    block_end = (q_frame // num_frame_per_block + 1) * num_frame_per_block
    if kv_frame >= block_end:
        return False
    if local_attn_size == -1:
        return True
    rolling = max(0, int(local_attn_size) - int(sink_size))
    if kv_frame >= block_end - rolling:
        return True
    return kv_frame < sink_size


def teacher_forcing_frame_visible(
    q_half: int,
    q_frame: int,
    kv_half: int,
    kv_frame: int,
    *,
    num_frame_per_block: int,
    local_attn_size: int,
    sink_size: int,
) -> bool:
    """Frame-level visibility for the ``[clean | noisy]`` teacher-forcing layout.

    ``*_half`` is 0 for the clean half and 1 for the noisy half. Clean rows are
    blockwise-causal over the clean half (they mirror the context write-back
    forward at inference); a noisy row attends its own noisy block plus the
    clean context of strictly-previous blocks. With a rolling window the clean
    context is restricted to the sink plus the trailing window, whose budget
    includes the noisy block itself (it occupies the newest ``num_frame_per_block``
    cache slots while being denoised at inference).
    """
    block = q_frame // num_frame_per_block
    block_end = (block + 1) * num_frame_per_block
    if q_half == 0:
        if kv_half != 0:
            return False
        return blockwise_frame_visible(
            q_frame,
            kv_frame,
            num_frame_per_block=num_frame_per_block,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
    if kv_half == 1:
        return kv_frame // num_frame_per_block == block
    context_end = block_end - num_frame_per_block
    if kv_frame >= context_end:
        return False
    if local_attn_size == -1:
        return True
    rolling = max(0, int(local_attn_size) - int(sink_size))
    if kv_frame >= block_end - rolling:
        return True
    return kv_frame < sink_size


def sink_delta_frames(
    num_frames: int,
    *,
    num_frame_per_block: int,
    local_attn_size: int,
) -> list[int]:
    """Relativistic sink phase offset per query frame-block, in frames.

    ``delta[b] = max(0, block_end - local_attn_size)``: once the rolling
    window has scrolled past the sink, streaming inference re-indexes the sink
    to sit ``local_attn_size`` frames behind the newest block end instead of
    at its absolute distance.
    """
    num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block
    return [
        max(0, (b + 1) * num_frame_per_block - int(local_attn_size))
        for b in range(num_blocks)
    ]


def build_sink_delta_tables(
    *,
    num_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
    sink_size: int,
    hidden_size: int,
    num_attention_heads: int,
    rope_theta: float = 10000,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Per-query-block rotation tables for the relativistic sink correction.

    Returns ``(delta_cos, delta_sin)`` of shape ``[num_blocks, head_dim]`` in
    float32, or ``None`` when every delta is zero (window never scrolls past
    the sink) or the correction does not apply. ``rope(x, pos + delta)``
    equals the rope rotation at temporal position ``delta`` (h = w = 0, so the
    spatial dims get an identity rotation) applied to ``rope(x, pos)``;
    sampling one rope row per delta reuses the model's exact frequencies.
    """
    if local_attn_size == -1 or sink_size <= 0:
        return None
    deltas = sink_delta_frames(
        num_frames,
        num_frame_per_block=num_frame_per_block,
        local_attn_size=local_attn_size,
    )
    max_delta = max(deltas)
    if max_delta == 0:
        return None
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    d = hidden_size // num_attention_heads
    rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]
    cos, sin = get_rotary_pos_embed(
        (max_delta + 1, 1, 1),
        hidden_size,
        num_attention_heads,
        rope_dim_list,
        rope_theta=rope_theta,
        dtype=dtype,
    )
    idx = torch.tensor(deltas, dtype=torch.long)
    delta_cos = cos.index_select(0, idx).to(device=device, dtype=torch.float32)
    delta_sin = sin.index_select(0, idx).to(device=device, dtype=torch.float32)
    return delta_cos, delta_sin


@dataclass
class CausalTrainAttentionPlan:
    """Cacheable description of one full-sequence training attention layout."""

    kind: str  # "blockwise" | "teacher_forcing"
    impl: str  # "triton" | "reference"
    num_frames: int  # latent frames (per half for teacher_forcing)
    frame_seqlen: int
    num_frame_per_block: int
    local_attn_size: int
    sink_size: int
    sm_scale: float
    # Relativistic sink correction ([num_blocks, head_dim] float32 each);
    # None means the correction is a no-op and absolute RoPE is exact.
    delta_cos: torch.Tensor | None = None
    delta_sin: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("blockwise", "teacher_forcing"):
            raise ValueError(f"Unknown plan kind: {self.kind!r}")
        if self.impl not in ("triton", "reference"):
            raise ValueError(f"Unknown plan impl: {self.impl!r}")
        if (self.delta_cos is None) != (self.delta_sin is None):
            raise ValueError("delta_cos and delta_sin must be set together")

    @property
    def seq_len(self) -> int:
        halves = 2 if self.kind == "teacher_forcing" else 1
        return self.num_frames * self.frame_seqlen * halves

    def without_delta(self) -> CausalTrainAttentionPlan:
        """Mask-only copy for attention streams that do not carry RoPE (PRoPE)."""
        if self.delta_cos is None:
            return self
        return replace(self, delta_cos=None, delta_sin=None)


def _rotate_half_gptj(x: torch.Tensor) -> torch.Tensor:
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack([-x_imag, x_real], dim=-1).flatten(-2)


def apply_sink_delta_to_query(
    q: torch.Tensor,
    delta_cos: torch.Tensor,
    delta_sin: torch.Tensor,
) -> torch.Tensor:
    """Rotate roped queries back by their block's sink delta (``R(-delta)``).

    ``score = <rope(k, f + delta), rope(q, t)> = <rope(k, f), R(-delta) rope(q, t)>``,
    so the per-query-block sink repositioning is applied on the query side.

    Args:
        q: ``[..., L, D]`` roped queries.
        delta_cos / delta_sin: ``[L, D]`` rows already gathered per query token.
    """
    return (q.float() * delta_cos - _rotate_half_gptj(q.float()) * delta_sin).type_as(q)


def _plan_row_geometry(
    plan: CausalTrainAttentionPlan,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token (half, frame, block) indices for a plan's sequence layout."""
    tokens_per_half = plan.num_frames * plan.frame_seqlen
    idx = torch.arange(plan.seq_len, device=device)
    half = (idx // tokens_per_half).clamp(max=1)
    frame = (idx % tokens_per_half) // plan.frame_seqlen
    block = frame // plan.num_frame_per_block
    return half, frame, block


def build_plan_visibility(
    plan: CausalTrainAttentionPlan,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Dense boolean visibility matrix ``[seq_len, seq_len]`` for a plan."""
    device = torch.device(device)
    half, frame, block = _plan_row_geometry(plan, device)
    q_half, kv_half = half[:, None], half[None, :]
    q_frame, kv_frame = frame[:, None], frame[None, :]
    q_block, kv_block = block[:, None], block[None, :]
    block_end = (q_block + 1) * plan.num_frame_per_block

    if plan.local_attn_size == -1:
        in_window = torch.ones_like(kv_frame < 0)
        in_sink = torch.zeros_like(in_window)
    else:
        rolling = max(0, plan.local_attn_size - plan.sink_size)
        in_window = kv_frame >= (block_end - rolling)
        in_sink = kv_frame < plan.sink_size
    windowed = in_window | in_sink

    if plan.kind == "blockwise":
        return (kv_frame < block_end) & windowed

    clean_rows = (q_half == 0) & (kv_half == 0) & (kv_frame < block_end) & windowed
    noisy_self = (q_half == 1) & (kv_half == 1) & (kv_block == q_block)
    context_end = block_end - plan.num_frame_per_block
    noisy_context = ((q_half == 1) & (kv_half == 0) &
                     (kv_frame < context_end) & windowed)
    return clean_rows | noisy_self | noisy_context


def reference_causal_train_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: CausalTrainAttentionPlan,
) -> torch.Tensor:
    """O(L^2) float32 reference for the plan semantics.

    Args:
        q / k / v: ``[B, H, L, D]``.
    Returns:
        ``[B, H, L, D]`` in ``v.dtype``.
    """
    seq_len = q.shape[-2]
    if seq_len != plan.seq_len:
        raise ValueError(f"Plan expects seq_len={plan.seq_len}, got {seq_len}")
    device = q.device
    visible = build_plan_visibility(plan, device=device)

    scores = torch.einsum("bhld,bhmd->bhlm", q.float(), k.float()) * plan.sm_scale

    if plan.delta_cos is not None:
        _, _, block = _plan_row_geometry(plan, device)
        delta_cos = plan.delta_cos.to(device=device)[block]
        delta_sin = plan.delta_sin.to(device=device)[block]
        q_delta = apply_sink_delta_to_query(q.float(), delta_cos, delta_sin)
        sink_scores = torch.einsum("bhld,bhmd->bhlm", q_delta, k.float()) * plan.sm_scale
        sink_tokens = plan.sink_size * plan.frame_seqlen
        sink_col = torch.arange(seq_len, device=device)[None, None, None, :] < sink_tokens
        scores = torch.where(sink_col, sink_scores, scores)

    scores = scores.masked_fill(~visible, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhlm,bhmd->bhld", probs, v.float())
    return out.type_as(v)


def run_causal_train_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: CausalTrainAttentionPlan,
) -> torch.Tensor:
    """Dispatch a plan to its implementation. q/k/v: ``[B, H, L, D]``."""
    if plan.impl == "triton":
        from fastvideo.attention.kernels.block_causal_sink import (
            block_causal_sink_attention, )
        return block_causal_sink_attention(q, k, v, plan)
    return reference_causal_train_attention(q, k, v, plan)


def validate_causal_attention_geometry(
    *,
    local_attn_size: int,
    sink_size: int,
    num_frame_per_block: int,
    where: str,
) -> None:
    """Shared validation for the sink + rolling-window geometry."""
    if sink_size < 0:
        raise ValueError(f"{where}: sink_size must be non-negative, got {sink_size}")
    if local_attn_size == -1 or sink_size == 0:
        return
    if sink_size + num_frame_per_block > local_attn_size:
        raise ValueError(
            f"{where}: local_attn_size ({local_attn_size}) must cover the sink "
            f"plus at least one frame block (sink_size={sink_size} + "
            f"num_frame_per_block={num_frame_per_block}); the rolling KV cache "
            "otherwise cannot hold a new block after pinning the sink")


def approx_relativistic_delta_max(
    *,
    num_frames: int,
    num_frame_per_block: int,
    local_attn_size: int,
) -> int:
    """Largest sink phase offset the training sequence would need."""
    if local_attn_size == -1:
        return 0
    deltas = sink_delta_frames(
        num_frames,
        num_frame_per_block=num_frame_per_block,
        local_attn_size=local_attn_size,
    )
    return max(deltas) if deltas else 0


__all__ = [
    "CausalTrainAttentionPlan",
    "apply_sink_delta_to_query",
    "approx_relativistic_delta_max",
    "blockwise_frame_visible",
    "build_plan_visibility",
    "build_sink_delta_tables",
    "reference_causal_train_attention",
    "run_causal_train_attention",
    "sink_delta_frames",
    "teacher_forcing_frame_visible",
    "validate_causal_attention_geometry",
]
