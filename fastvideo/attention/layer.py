# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn

from fastvideo.attention.ring import ring_flash_attn_func
from fastvideo.attention.selector import backend_name_to_enum, get_attn_backend
from fastvideo.distributed.communication_op import (sequence_model_parallel_all_gather,
                                                    sequence_model_parallel_all_to_all_4D)
from fastvideo.distributed.parallel_state import (get_ring_group, get_ring_size, get_sp_parallel_rank,
                                                  get_sp_world_size)
from fastvideo.forward_context import ForwardContext, get_forward_context
from fastvideo.platforms import AttentionBackendEnum
from fastvideo.utils import get_compute_dtype
from fastvideo.layers.rotary_embedding import _apply_rotary_emb


def _attention_compile_disabled() -> bool:
    """Whether to keep attention ``forward`` out of the torch.compile graph.

    Defaults to ``True`` (the historical behavior: attention runs eager via
    ``torch.compiler.disable``). Set ``FASTVIDEO_DISABLE_ATTENTION_COMPILE=0``
    to let attention be traced/compiled into the surrounding graph.
    """
    val = os.environ.get("FASTVIDEO_DISABLE_ATTENTION_COMPILE")
    if val is None:
        return True
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _maybe_compiler_disable(fn):
    """Apply ``torch.compiler.disable`` unless disabled via env var."""
    if _attention_compile_disabled():
        return torch.compiler.disable(fn)
    return fn


class DistributedAttention(nn.Module):
    """Distributed attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 prefix: str = "",
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(head_size, dtype, supported_attention_backends=supported_attention_backends)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  causal=causal,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  prefix=f"{prefix}.impl",
                                  **extra_impl_args)
        # Register attn_impl as submodule if it has learnable parameters (e.g., SLA's proj_l)
        # This ensures its parameters are included in state_dict() for saving/loading
        if isinstance(self.attn_impl, nn.Module):
            self.add_module('attn_impl', self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype
        self.causal = causal

        # Ring Attention is a process-wide parallelism setting (like SP itself),
        # so it is read from the distributed runtime rather than threaded
        # through every model's constructor.
        self.ring_size = get_ring_size()
        self.use_ring_attention = self.ring_size > 1
        if self.use_ring_attention:
            if self.ring_size != get_sp_world_size():
                raise RuntimeError("Pure Ring Attention requires ring_size == SP world size. "
                                   f"Got ring_size={self.ring_size}, sp_world_size={get_sp_world_size()}.")
            if self.backend != AttentionBackendEnum.FLASH_ATTN:
                raise NotImplementedError("The initial Ring Attention implementation only supports the "
                                          f"FlashAttention backend, got backend={self.backend}.")
            if num_heads != num_kv_heads:
                raise NotImplementedError("The initial Ring Attention implementation does not support GQA. "
                                          f"num_heads={num_heads}, num_kv_heads={num_kv_heads}.")

    @_maybe_compiler_disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int | None = None,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.

        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            original_seq_len (int): Original (unpadded) full sequence length
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        if self.use_ring_attention:
            return self._forward_ring_attention(
                q=q,
                k=k,
                v=v,
                original_seq_len=original_seq_len,
                replicated_q=replicated_q,
                replicated_k=replicated_k,
                replicated_v=replicated_v,
                freqs_cis=freqs_cis,
            )

        return self._forward_ulysses_attention(
            q=q,
            k=k,
            v=v,
            original_seq_len=original_seq_len,
            replicated_q=replicated_q,
            replicated_k=replicated_k,
            replicated_v=replicated_v,
            freqs_cis=freqs_cis,
        )

    def _forward_ulysses_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int | None,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, _, num_heads, _ = q.shape
        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3*batch, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)

        # After all-to-all, each rank has the full sequence but only a subset of heads.
        # Trim away SP padding for attention compute, then pad back before returning.
        original_seq_len = original_seq_len or qkv.shape[1]
        pad_seq_len = qkv.shape[1] - original_seq_len
        qkv = qkv[:, :original_seq_len, :, :]

        if freqs_cis is not None:
            cos, sin = freqs_cis
            qkv[:batch_size * 2] = _apply_rotary_emb(qkv[:batch_size * 2], cos, sin, is_neox_style=False)
        # Apply backend-specific preprocess_qkv
        qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat([replicated_q, replicated_k, replicated_v],
                                       dim=0)  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[:, :, local_rank * heads_per_rank:(local_rank + 1) * heads_per_rank]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            split_idx = original_seq_len
            replicated_output = output[:, split_idx:]
            output = output[:, :split_idx]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(replicated_output.contiguous(), dim=2)
        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))

        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)

        return output, replicated_output

    def _validate_ring_inputs(
        self,
        q: torch.Tensor,
        original_seq_len: int | None,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
    ) -> None:
        if self.training:
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

    @staticmethod
    def _slice_local_rope(
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        local_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice global RoPE tables down to this rank's contiguous token range.

        ``freqs_cis`` (as produced by ``get_rotary_pos_embed``) covers the full,
        unsharded global sequence with shape ``[global_seq_len, head_size]``.
        Ulysses can apply it unsliced because the all-to-all gathers the full
        sequence before RoPE is applied; Ring Attention never gathers the
        sequence, so each rank must rotate only its own token range.
        """
        rank = get_sp_parallel_rank()
        start = rank * local_seq_len
        end = start + local_seq_len

        cos, sin = freqs_cis
        if cos.shape[0] < end or sin.shape[0] < end:
            raise ValueError("RoPE tables are shorter than the required global token range. "
                             f"rank={rank}, local_seq_len={local_seq_len}, required_end={end}, "
                             f"cos_shape={tuple(cos.shape)}, sin_shape={tuple(sin.shape)}.")

        return cos[start:end], sin[start:end]

    def _forward_ring_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int | None,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, None]:
        self._validate_ring_inputs(
            q=q,
            original_seq_len=original_seq_len,
            replicated_q=replicated_q,
            replicated_k=replicated_k,
            replicated_v=replicated_v,
        )

        batch_size, local_seq_len, _, _ = q.shape

        if freqs_cis is not None:
            local_cos, local_sin = self._slice_local_rope(freqs_cis, local_seq_len)
            q = _apply_rotary_emb(q, local_cos, local_sin, is_neox_style=False)
            k = _apply_rotary_emb(k, local_cos, local_sin, is_neox_style=False)

        # Ring Attention calls the distributed ring kernel directly rather than
        # ``self.attn_impl.forward``: the latter runs plain (non-distributed)
        # FlashAttention over whatever it is given, which would silently
        # compute attention within the local shard only. ``preprocess_qkv`` /
        # ``postprocess_output`` are identity for the FlashAttention backend
        # (the only backend Ring supports, enforced in ``__init__``), so
        # skipping them here does not diverge from the Ulysses path.
        ring_group = get_ring_group()
        if ring_group is None:
            raise RuntimeError("Ring Attention is enabled, but the Ring process group is not initialized.")

        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        output = ring_flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=self.softmax_scale,
            causal=False,
            group=ring_group,
        )

        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)

        if output.shape[:2] != (batch_size, local_seq_len):
            raise RuntimeError("Ring Attention must preserve the local sequence shard. Expected output "
                               f"prefix {(batch_size, local_seq_len)}, got {tuple(output.shape)}.")

        return output, None


class DistributedAttention_VSA(DistributedAttention):
    """Distributed attention layer with VSA support.
    """

    @_maybe_compiler_disable
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_seq_len: int,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.
        
        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            original_seq_len (int): Original (unpadded) full sequence length
            gate_compress (torch.Tensor): Gate compress tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check text tokens are not supported for VSA now
        assert replicated_q is None and replicated_k is None and replicated_v is None, "Replicated QKV is not supported for VSA now"
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        batch_size, seq_len, num_heads, head_dim = q.shape
        # Stack QKV
        qkvg = torch.cat([q, k, v, gate_compress], dim=0)  # [4*batch, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        # Before: [4*batch, shard_seq_len, num_heads, head_dim]
        # After:  [4*batch, full_seq_len, shard_num_heads, head_dim]
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)

        # After all-to-all, each rank has the full sequence but only a subset of heads
        pad_seq_len = qkvg.shape[1] - original_seq_len
        qkvg = qkvg[:, :original_seq_len, :, :]

        if freqs_cis is not None:
            cos, sin = freqs_cis
            qkvg[:batch_size * 2] = _apply_rotary_emb(qkvg[:batch_size * 2], cos, sin, is_neox_style=False)

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        q, k, v, gate_compress = qkvg.chunk(4, dim=0)
        output = self.attn_impl.forward(q, k, v, gate_compress, ctx_attn_metadata)  # type: ignore[call-arg]

        # Redistribute back if using sequence parallelism
        replicated_output = None

        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = torch.nn.functional.pad(output, (0, 0, 0, 0, 0, pad_seq_len))

        output = sequence_model_parallel_all_to_all_4D(output, scatter_dim=1, gather_dim=2)
        return output, replicated_output


class LocalAttention(nn.Module):
    """Attention layer.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 num_kv_heads: int | None = None,
                 softmax_scale: float | None = None,
                 causal: bool = False,
                 supported_attention_backends: tuple[AttentionBackendEnum, ...]
                 | None = None,
                 default_backend: AttentionBackendEnum | None = None,
                 **extra_impl_args) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(head_size,
                                        dtype,
                                        supported_attention_backends=supported_attention_backends,
                                        default_backend=default_backend)
        impl_cls = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(num_heads=num_heads,
                                  head_size=head_size,
                                  softmax_scale=self.softmax_scale,
                                  num_kv_heads=num_kv_heads,
                                  causal=causal,
                                  **extra_impl_args)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = backend_name_to_enum(attn_backend.get_name())
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim] 
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if freqs_cis is not None:
            cos, sin = freqs_cis
            q = _apply_rotary_emb(q, cos, sin, is_neox_style=False)
            k = _apply_rotary_emb(k, cos, sin, is_neox_style=False)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        return output
