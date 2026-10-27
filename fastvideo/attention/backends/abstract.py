# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/backends/abstract.py

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    pass

import torch

_LAYER_IDX_RE = re.compile(r"blocks\.(\d+)")


def layer_idx_from_prefix(prefix: str, default: int | None = None) -> int:
    """Parse the transformer-block index out of a layer prefix.

    Shared by backends that key per-layer behavior off the block number
    (vmoba, VSA-H3). Raises when unparsable unless ``default`` is given.
    """
    match = _LAYER_IDX_RE.search(prefix)
    if match:
        return int(match.group(1))
    if default is None:
        raise ValueError(f"cannot parse a layer index from attention prefix {prefix!r}")
    return default


class AttentionBackend(ABC):
    """Abstract class for attention backends."""
    # For some attention backends, we allocate an output tensor before
    # calling the custom op. When piecewise cudagraph is enabled, this
    # makes sure the output tensor is allocated inside the cudagraph.
    accept_output_buffer: bool = False

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    # @staticmethod
    # @abstractmethod
    # def get_state_cls() -> Type["AttentionState"]:
    #     raise NotImplementedError

    # @classmethod
    # def make_metadata(cls, *args, **kwargs) -> "AttentionMetadata":
    #     return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError


@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Current step of diffusion process
    current_timestep: int
    VSA_sparsity: float = field(default=0.0, kw_only=True)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name not in skip_fields}


@dataclass(frozen=True)
class PackedSequenceMetadata:
    """Document boundaries and optional trailing capacity padding.

    Q/K/V remain one physical ``[1, real_tokens + pad_tokens, heads, dim]``
    sequence so sequence parallelism can shard them normally. Lengths and
    cumulative lengths describe only real documents: trailing capacity rows
    are neither queries nor keys, and backends restore zeros in their output
    rows. The same pad count trails Q and K/V, even for unequal real lengths.
    The host tuples mirror the device tensors so Python fallbacks never
    synchronize a CUDA tensor.

    ``sequence_capacity`` opts FA4 into fixed-shape Q/K/V storage and kernel
    outputs, with that capacity as both max-seqlen bounds. It must equal the
    physical Q and K/V lengths (real totals plus ``pad_tokens``). Construct
    metadata outside compiled regions: validation and eager fallbacks use
    host lengths, but fixed-capacity FA4 reads only the capacity and device
    cumulative lengths. Their final entries define the valid storage prefixes;
    changing their values does not specialize a graph on real token counts.
    """

    query_lengths: tuple[int, ...]
    key_value_lengths: tuple[int, ...]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    max_seqlen_q: int
    max_seqlen_kv: int
    # Trailing capacity rows, not an extra attention document.
    pad_tokens: int = 0
    sequence_capacity: int | None = None

    def __post_init__(self) -> None:
        if type(self.pad_tokens) is not int or self.pad_tokens < 0:
            raise ValueError("pad_tokens must be a non-negative integer")
        if not self.query_lengths:
            raise ValueError("Packed attention requires at least one document")
        if len(self.query_lengths) != len(self.key_value_lengths):
            raise ValueError("Packed query and key/value lengths must contain the same number of documents")
        if any(length <= 0 for length in (*self.query_lengths, *self.key_value_lengths)):
            raise ValueError("Packed attention document lengths must all be positive")
        if self.max_seqlen_q != max(self.query_lengths):
            raise ValueError("max_seqlen_q does not match query_lengths")
        if self.max_seqlen_kv != max(self.key_value_lengths):
            raise ValueError("max_seqlen_kv does not match key_value_lengths")
        if self.sequence_capacity is not None:
            if type(self.sequence_capacity) is not int or self.sequence_capacity <= 0:
                raise ValueError("sequence_capacity must be a positive integer or None")
            if (self.total_query_tokens + self.pad_tokens != self.sequence_capacity
                    or self.total_key_value_tokens + self.pad_tokens != self.sequence_capacity):
                raise ValueError("sequence_capacity must equal both real Q and K/V totals plus pad_tokens")
        expected_shape = (len(self.query_lengths) + 1, )
        for name, tensor in (("cu_seqlens_q", self.cu_seqlens_q), ("cu_seqlens_kv", self.cu_seqlens_kv)):
            if tensor.dtype != torch.int32 or tensor.ndim != 1 or tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must be an int32 tensor with shape {expected_shape}, got "
                                 f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}")
        if self.cu_seqlens_q.device != self.cu_seqlens_kv.device:
            raise ValueError("Packed query and key/value cumulative lengths must be on the same device")

    @classmethod
    def from_lengths(
        cls,
        query_lengths: tuple[int, ...] | list[int],
        *,
        device: torch.device | str,
        key_value_lengths: tuple[int, ...] | list[int] | None = None,
        pad_tokens: int = 0,
        sequence_capacity: int | None = None,
    ) -> "PackedSequenceMetadata":
        query_lengths = tuple(int(length) for length in query_lengths)
        if key_value_lengths is None:
            key_value_lengths = query_lengths
        key_value_lengths = tuple(int(length) for length in key_value_lengths)

        def cumulative(lengths: tuple[int, ...]) -> torch.Tensor:
            values = [0]
            for length in lengths:
                values.append(values[-1] + length)
            return torch.tensor(values, dtype=torch.int32, device=device)

        return cls(
            pad_tokens=pad_tokens,
            sequence_capacity=sequence_capacity,
            query_lengths=query_lengths,
            key_value_lengths=key_value_lengths,
            cu_seqlens_q=cumulative(query_lengths),
            cu_seqlens_kv=cumulative(key_value_lengths),
            max_seqlen_q=max(query_lengths, default=0),
            max_seqlen_kv=max(key_value_lengths, default=0),
        )

    @property
    def num_documents(self) -> int:
        return len(self.query_lengths)

    @property
    def total_query_tokens(self) -> int:
        return sum(self.query_lengths)

    @property
    def total_key_value_tokens(self) -> int:
        return sum(self.key_value_lengths)


@dataclass
class PackedAttentionMetadata(AttentionMetadata):
    """Backend-neutral metadata for dense attention over packed documents."""

    packed_sequence: PackedSequenceMetadata
    attn_mask: torch.Tensor | None = None


T = TypeVar("T", bound=AttentionMetadata)


class AttentionMetadataBuilder(ABC, Generic[T]):
    """Abstract class for attention metadata builders."""

    @abstractmethod
    def __init__(self) -> None:
        """Create the builder, remember some configuration and parameters."""
        raise NotImplementedError

    @abstractmethod
    def prepare(self) -> None:
        """Prepare for one batch."""
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        **kwargs: Any,
    ) -> AttentionMetadata:
        """Build attention metadata with on-device tensors."""
        raise NotImplementedError


class AttentionLayer(Protocol):

    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    _k_scale_float: float
    _v_scale_float: float

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        ...


class AttentionImpl(ABC, Generic[T]):

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        raise NotImplementedError

    def preprocess_qkv(self, qkv: torch.Tensor, attn_metadata: T) -> torch.Tensor:
        """Preprocess QKV tensor before performing attention operation.

        Default implementation returns the tensor unchanged.
        Subclasses can override this to implement custom preprocessing
        like reshaping, tiling, scaling, or other transformations.

        Called AFTER all_to_all for distributed attention
        
        Args:
            qkv: The query-key-value tensor
            attn_metadata: Metadata for the attention operation
            
        Returns:
            Processed QKV tensor
        """
        return qkv

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        """Postprocess the output tensor after the attention operation.

        Default implementation returns the tensor unchanged.
        Subclasses can override this to implement custom postprocessing
        like untiling, scaling, or other transformations.

        Called BEFORE all_to_all for distributed attention

        Args:
            output: The output tensor from the attention operation
            attn_metadata: Metadata for the attention operation

        Returns:
            Postprocessed output tensor
        """

        return output

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError
