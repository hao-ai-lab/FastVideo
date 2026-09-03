# SPDX-License-Identifier: Apache-2.0
"""Ring Attention (optionally combined with Ulysses as the USP hybrid).

Owns every Ring/USP-specific decision: construction-time validation of a
layer's shape against the process-wide Ring topology, the Ulysses-within-ring
all-to-all, Ring-local RoPE slicing, and the direct call into the vendored
Ring FlashAttention kernel -- bypassing the local-kernel-backend abstraction
(``fastvideo.attention.selector`` / ``AttentionImpl``) entirely, since that
abstraction resolves a single-GPU math kernel once per component, while Ring
is a distributed strategy layered above it: it picks *which* backend runs
per shard (today hard-restricted to FLASH_ATTN) and changes RoPE application,
so it cannot also be a member of that registry without a circular
relationship.

``fastvideo.attention.layer.DistributedAttention`` delegates to an instance
of this class exactly the way it delegates local-kernel choice to
``self.attn_impl`` -- Ring is a second delegate with the same shape, not a
code path inlined into the attention layer.
"""

from __future__ import annotations

import torch

from fastvideo.attention.ring import ring_flash_attn_func
from fastvideo.distributed.communication_op import ulysses_all_to_all_4D
from fastvideo.distributed.parallel_state import (get_ring_group, get_ring_rank, get_ring_size, get_sp_world_size,
                                                  get_ulysses_group)
from fastvideo.layers.rotary_embedding import _apply_rotary_emb
from fastvideo.platforms import AttentionBackendEnum


class RingAttention:
    """Ring Attention (and its USP hybrid with Ulysses) for one attention layer."""

    @classmethod
    def create_if_enabled(
        cls,
        *,
        num_heads: int,
        num_kv_heads: int,
        softmax_scale: float,
        causal: bool,
        backend: AttentionBackendEnum,
    ) -> RingAttention | None:
        """Return a configured ``RingAttention``, or ``None`` if Ring Attention
        is disabled (``ring_size == 1``) for the current process.

        This is the only place outside this module that reads the
        process-wide Ring topology (``DistributedAttention`` itself no longer
        does) -- everything else about Ring is decided here, at construction
        time, from the layer's own shape.
        """
        if get_ring_size() <= 1:
            return None
        return cls(num_heads=num_heads,
                   num_kv_heads=num_kv_heads,
                   softmax_scale=softmax_scale,
                   causal=causal,
                   backend=backend)

    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        softmax_scale: float,
        causal: bool,
        backend: AttentionBackendEnum,
    ) -> None:
        self.ring_size = get_ring_size()
        sp_world_size = get_sp_world_size()
        if sp_world_size % self.ring_size != 0:
            raise RuntimeError("Ring Attention requires ring_size to evenly divide the SP world size. "
                               f"Got ring_size={self.ring_size}, sp_world_size={sp_world_size}.")
        if backend != AttentionBackendEnum.FLASH_ATTN:
            raise NotImplementedError("The initial Ring Attention implementation only supports the "
                                      f"FlashAttention backend, got backend={backend}.")
        if num_heads != num_kv_heads:
            raise NotImplementedError("The initial Ring Attention implementation does not support GQA. "
                                      f"num_heads={num_heads}, num_kv_heads={num_kv_heads}.")
        ulysses_size = sp_world_size // self.ring_size
        if ulysses_size > 1 and num_heads % ulysses_size != 0:
            raise NotImplementedError("The Ring+Ulysses (USP) hybrid requires num_heads to be divisible by the Ulysses "
                                      f"subgroup size. Got num_heads={num_heads}, ulysses_size={ulysses_size} "
                                      f"(sp_world_size={sp_world_size} // ring_size={self.ring_size}).")
        self.softmax_scale = softmax_scale
        self.causal = causal

    def _validate_ring_inputs(
        self,
        q: torch.Tensor,
        *,
        training: bool,
        original_seq_len: int | None,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
    ) -> None:
        if training:
            raise NotImplementedError("Ring Attention training/backward is not supported in the initial "
                                      "FastVideo integration.")

        if replicated_q is not None or replicated_k is not None or replicated_v is not None:
            raise NotImplementedError("Ring Attention does not yet support replicated Q/K/V tokens "
                                      "(e.g. text tokens concatenated onto the visual sequence).")

        if self.causal:
            raise NotImplementedError("The initial Ring Attention integration only supports non-causal "
                                      "self-attention.")

        if original_seq_len is not None:
            local_seq_len = q.shape[1]
            global_seq_len = local_seq_len * get_sp_world_size()
            if original_seq_len != global_seq_len:
                raise NotImplementedError(
                    "Ring Attention does not yet support sequence-parallel padding or uneven shards. "
                    f"original_seq_len={original_seq_len}, local_seq_len={local_seq_len}, "
                    f"sp_world_size={get_sp_world_size()} (expected original_seq_len == "
                    f"local_seq_len * sp_world_size == {global_seq_len}).")

        if q.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"Ring Attention requires fp16/bf16 inputs (the underlying FlashAttention kernel does not "
                f"support fp32), got {q.dtype}. Cast Q/K/V before calling this layer instead of relying on "
                f"a silent cast here.")

    @staticmethod
    def _slice_local_rope(
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        local_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice global RoPE tables down to this rank's contiguous token range.

        ``freqs_cis`` (as produced by ``get_rotary_pos_embed``) covers the full,
        unsharded global sequence with shape ``[global_seq_len, head_size]``.
        Ulysses can apply it unsliced because the all-to-all gathers the full
        sequence before RoPE is applied. Ring Attention (pure, or the Ring
        side of the Ring+Ulysses/USP hybrid) never gathers the sequence, so
        each rank must rotate only its own contiguous chunk.

        ``local_seq_len`` here is the length of that Ring chunk (equal to the
        SP shard length in pure Ring, since ring rank == SP rank there; equal
        to ``sp_shard_len * ulysses_size`` in the hybrid, since each Ring
        chunk is first assembled from ``ulysses_size`` contiguous SP shards
        by the pre-Ring Ulysses all-to-all). ``get_ring_rank()`` indexes that
        chunk directly, so it is used regardless of which case is active
        (it degenerates to the SP rank in pure Ring, and to 0 when Ring is
        disabled entirely).
        """
        rank = get_ring_rank()
        start = rank * local_seq_len
        end = start + local_seq_len

        cos, sin = freqs_cis
        if cos.shape[0] < end or sin.shape[0] < end:
            raise ValueError("RoPE tables are shorter than the required global token range. "
                             f"rank={rank}, local_seq_len={local_seq_len}, required_end={end}, "
                             f"cos_shape={tuple(cos.shape)}, sin_shape={tuple(sin.shape)}.")

        return cos[start:end], sin[start:end]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        training: bool,
        original_seq_len: int | None,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, None]:
        """Ring Attention, optionally combined with Ulysses (USP hybrid).

        When ``ulysses_size == 1`` (``ring_size == sp_world_size``) this is
        pure Ring Attention: every SP rank is a Ring rank, and the Ulysses
        all-to-all below is a no-op. When ``ulysses_size > 1``, each SP
        replica is arranged as a ``ring_size x ulysses_size`` mesh: the
        Ulysses all-to-all first redistributes heads -> sequence *within*
        this rank's Ulysses subgroup, assembling one full contiguous "Ring
        chunk" (``ulysses_size`` SP shards' worth of tokens) held on a
        reduced number of heads. Ring Attention then runs across the
        ``ring_size`` Ring ranks that each hold a different chunk but the
        same head subset, and a final Ulysses all-to-all redistributes
        sequence -> heads back to the original per-rank shard shape.
        """
        self._validate_ring_inputs(
            q,
            training=training,
            original_seq_len=original_seq_len,
            replicated_q=replicated_q,
            replicated_k=replicated_k,
            replicated_v=replicated_v,
        )

        batch_size, local_seq_len, _, _ = q.shape

        # Ulysses step (skipped entirely when ulysses_size == 1, i.e. pure
        # Ring): redistribute heads -> sequence within this rank's Ulysses
        # subgroup so that each Ring rank holds one full, contiguous Ring
        # chunk. The stack-into-one-all-to-all-then-chunk below costs a real
        # copy of Q/K/V, so only pay it when there is an actual Ulysses
        # subgroup to redistribute within.
        if get_ulysses_group() is not None:
            qkv = torch.cat([q, k, v], dim=0)
            qkv = ulysses_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
            q, k, v = qkv.chunk(3, dim=0)

        ring_local_seq_len = q.shape[1]

        if freqs_cis is not None:
            local_cos, local_sin = self._slice_local_rope(freqs_cis, ring_local_seq_len)
            q = _apply_rotary_emb(q, local_cos, local_sin, is_neox_style=False)
            k = _apply_rotary_emb(k, local_cos, local_sin, is_neox_style=False)

        # Ring Attention calls the distributed ring kernel directly rather than
        # a local-kernel-backend's ``AttentionImpl.forward``: that runs plain
        # (non-distributed) FlashAttention over whatever it is given, which
        # would silently compute attention within the local shard only.
        # ``preprocess_qkv`` / ``postprocess_output`` are identity for the
        # FlashAttention backend (the only backend Ring supports, enforced in
        # ``__init__``), so skipping them here does not diverge from the
        # Ulysses path.
        ring_group = get_ring_group()
        if ring_group is None:
            raise RuntimeError("Ring Attention is enabled, but the Ring process group is not initialized.")

        output = ring_flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
            group=ring_group.device_group,
        )

        # Ulysses step back (no-op when ulysses_size == 1): redistribute
        # sequence -> heads to restore the original per-rank shard shape.
        output = ulysses_all_to_all_4D(output, scatter_dim=1, gather_dim=2)

        if output.shape[:2] != (batch_size, local_seq_len):
            raise RuntimeError("Ring Attention must preserve the local sequence shard. Expected output "
                               f"prefix {(batch_size, local_seq_len)}, got {tuple(output.shape)}.")

        return output, None
