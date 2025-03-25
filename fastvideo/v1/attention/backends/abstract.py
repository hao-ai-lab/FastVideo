# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional,
                    Protocol, Set, Tuple, Type, TypeVar)

from fastvideo.v1.inference_args import InferenceArgs
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

import torch

# from vllm.multimodal import MultiModalPlaceholderMap


# class AttentionType:
#     """
#     Attention type.
#     Use string to be compatible with `torch.compile`.
#     """
#     # Decoder attention between previous layer Q/K/V
#     DECODER = "decoder"
#     # Encoder attention between previous layer Q/K/V for encoder-decoder
#     ENCODER = "encoder"
#     # Encoder attention between previous layer Q/K/V
#     ENCODER_ONLY = "encoder_only"
#     # Attention between dec. Q and enc. K/V for encoder-decoder
#     ENCODER_DECODER = "encoder_decoder"


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
    def get_impl_cls() -> Type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
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
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise NotImplementedError


@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    # Current step of diffusion process 
    current_timestep: int

    @property
    @abstractmethod
    def inference_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run prefill
        attention."""
        pass

    @property
    @abstractmethod
    def training_metadata(self) -> Optional["AttentionMetadata"]:
        """Return the attention metadata that's required to run decode
        attention."""
        pass

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        """Similar to dataclasses.asdict, but avoids deepcopying."""
        if skip_fields is None:
            skip_fields = set()
        # Note that if we add dataclasses as fields, they will need
        # similar handling.
        return {
            field.name: getattr(self, field.name)
            for field in fields(self) if field.name not in skip_fields
        }


T = TypeVar("T", bound=AttentionMetadata)


# class AttentionState(ABC, Generic[T]):
#     """Holds attention backend-specific objects reused during the
#     lifetime of the model runner."""

#     @abstractmethod
#     def __init__(self, runner: "ModelRunnerBase"):
#         ...

#     @abstractmethod
#     @contextmanager
#     def graph_capture(self, max_batch_size: int):
#         """Context manager used when capturing CUDA graphs."""
#         yield

#     @abstractmethod
#     def graph_clone(self, batch_size: int) -> "AttentionState[T]":
#         """Clone attention state to save in CUDA graph metadata."""
#         ...

#     @abstractmethod
#     def graph_capture_get_metadata_for_batch(
#             self,
#             batch_size: int,
#             is_encoder_decoder_model: bool = False) -> T:
#         """Get attention metadata for CUDA graph capture of batch_size."""
#         ...

#     @abstractmethod
#     def get_graph_input_buffers(
#             self,
#             attn_metadata: T,
#             is_encoder_decoder_model: bool = False) -> Dict[str, Any]:
#         """Get attention-specific input buffers for CUDA graph capture."""
#         ...

#     @abstractmethod
#     def prepare_graph_input_buffers(
#             self,
#             input_buffers: Dict[str, Any],
#             attn_metadata: T,
#             is_encoder_decoder_model: bool = False) -> None:
#         """In-place modify input buffers dict for CUDA graph replay."""
#         ...

#     @abstractmethod
#     def begin_forward(self, model_input: "ModelRunnerInputBase") -> None:
#         """Prepare state for forward pass."""
#         ...


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
    def build(self,
              num_step: int,
              encoder_outputs: torch.Tensor,
              inference_args: InferenceArgs,
              forward_batch: ForwardBatch,
    ) -> T:
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
        dropout_rate: float = 0.0,
        causal: bool = False,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        raise NotImplementedError
    
    def preprocess_qkv(
        self, 
        qkv: torch.Tensor, 
        attn_metadata: T
    ) -> torch.Tensor:
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
        replicated_q: Optional[torch.Tensor] = None,
        replicated_k: Optional[torch.Tensor] = None,
        replicated_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
