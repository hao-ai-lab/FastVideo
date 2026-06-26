# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/backends/abstract.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    pass

import torch


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

    # ------------------------------------------------------------------
    # Capability self-description
    #
    # These let the selector validate a requested backend against a
    # layer's needs and emit a clear reason when falling back, instead of
    # silently discarding the choice (see #1254). They are additive: the
    # defaults describe the least-restrictive behavior, so a backend that
    # does not override them keeps today's behavior. Several backends
    # already declare ``get_supported_head_sizes``; this formalizes that
    # hook on the base class and adds the other capability axes.
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's third-party dependencies are importable.

        Defaults to True. NOTE: backends import heavy deps at module scope
        today, so wiring this into platform selection (to replace the
        scattered ``try/except ImportError`` blocks in
        ``get_attn_backend_cls``) also requires moving those imports behind
        this method -- left as a follow-up.
        """
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int] | None:
        """Attention head sizes this backend supports.

        Return None for no restriction.
        """
        return None

    @classmethod
    def get_supported_dtypes(cls) -> tuple[torch.dtype, ...]:
        """Floating-point dtypes this backend supports."""
        return (torch.float16, torch.bfloat16)

    @classmethod
    def supports_attention_mask(cls) -> bool:
        """Whether this backend can consume an explicit attention mask/bias.

        Dense backends generally can; tiled/sparse video backends generally
        cannot, since they impose their own sparsity pattern.
        """
        return True

    @classmethod
    def supports_varlen(cls) -> bool:
        """Whether this backend supports single-launch variable-length
        sequence packing (``cu_seqlens``), as in ``flash_attn_varlen_func``.
        """
        return False

    @classmethod
    def validate_compatibility(
        cls,
        head_size: int,
        dtype: torch.dtype,
        *,
        needs_attention_mask: bool = False,
        needs_varlen: bool = False,
    ) -> str | None:
        """Return None if this backend is compatible with the given
        requirements, else a human-readable reason for the mismatch.

        The selector logs this reason when falling back, so an unsupported
        request is never silently dropped.
        """
        if not cls.is_available():
            return f"{cls.get_name()} dependencies are not installed"
        supported_sizes = cls.get_supported_head_sizes()
        if supported_sizes is not None and head_size not in supported_sizes:
            return (f"{cls.get_name()} does not support head_size="
                    f"{head_size} (supported: {supported_sizes})")
        if dtype not in cls.get_supported_dtypes():
            return f"{cls.get_name()} does not support dtype={dtype}"
        if needs_attention_mask and not cls.supports_attention_mask():
            return (f"{cls.get_name()} cannot consume the attention mask this "
                    "layer requires")
        if needs_varlen and not cls.supports_varlen():
            return (f"{cls.get_name()} does not support variable-length "
                    "sequence packing (varlen)")
        return None


@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Current step of diffusion process
    current_timestep: int

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name not in skip_fields}


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
        **kwargs: dict[str, Any],
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
