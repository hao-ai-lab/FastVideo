# Attention Backend Development

This guide is for contributors adding a new attention backend (kernel or
implementation) to FastVideo. If you just want to use existing kernels or build
fastvideo-kernel, see [Attention overview](../attention/index.md).

## When you need this guide

Use this guide when you are:

- Adding a new attention kernel or algorithm.
- Wiring an existing kernel into FastVideo's attention selection.
- Extending attention support to a new platform.

## 0) Choose a backend name and scope

Pick a backend name in `UPPER_SNAKE_CASE` and decide where it should run.
Example: `MY_NEW_ATTN`.

You will use this name in:

- `AttentionBackendEnum` (global list of backends).
- `get_name()` in your backend class (match the enum name).
- Platform selectors (CUDA/ROCm/MPS/NPU) to return your backend class.

## 1) Add enum + platform selection

1) Add your backend to `fastvideo/platforms/interface.py`:

```python
class AttentionBackendEnum(enum.Enum):
    ...
    MY_NEW_ATTN = enum.auto()
```

1) Register it in platform selection (example: CUDA). Update
`fastvideo/platforms/cuda.py` inside `get_attn_backend_cls`:

```python
elif selected_backend == AttentionBackendEnum.MY_NEW_ATTN:
    try:
        from fastvideo.attention.backends.my_new_attn import MyNewAttnBackend
        return "fastvideo.attention.backends.my_new_attn.MyNewAttnBackend"
    except ImportError as e:
        logger.error("Failed to import MY_NEW_ATTN backend: %s", str(e))
        raise
```

If you want support on other platforms, add a similar branch in
`fastvideo/platforms/rocm.py`, `fastvideo/platforms/mps.py`, or `fastvideo/platforms/npu.py`.

## 2) Implement the backend

Create `fastvideo/attention/backends/my_new_attn.py` and implement the required
classes.

Minimal skeleton (no custom metadata):

```python
import torch
from dataclasses import dataclass
from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)

class MyNewAttnBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MY_NEW_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MyNewAttnImpl"]:
        return MyNewAttnImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return MyNewAttnMetadataBuilder


@dataclass
class MyNewAttnMetadata(AttentionMetadata):
    current_timestep: int


class MyNewAttnMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, current_timestep: int, **kwargs):
        return MyNewAttnMetadata(current_timestep=current_timestep)


class MyNewAttnImpl(AttentionImpl):
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
        self.softmax_scale = softmax_scale
        self.causal = causal

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MyNewAttnMetadata,
    ) -> torch.Tensor:
        # Implement attention
        return torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=self.causal,
            scale=self.softmax_scale,
        ).transpose(1, 2)
```

Optional:

- Implement `preprocess_qkv` / `postprocess_output` if your kernel needs tiling
  or reshaping.
- Use `fastvideo.forward_context.get_forward_context()` if you need dynamic
  per-step data (e.g., window sizes).
- Set `accept_output_buffer = True` if your backend writes into a provided
  output buffer.

## 3) Wire into attention layers

Backends are used by `LocalAttention` and `DistributedAttention`. These layers
accept a `supported_attention_backends` tuple. If your backend should be
eligible, update the call sites that construct these layers (search for
`supported_attention_backends=`).

## 4) Add compiled kernels (optional)

If you have a custom CUDA kernel:

1) Add sources in `fastvideo-kernel/csrc/attention/`.
2) Register bindings in `fastvideo-kernel/csrc/common_extension.cpp`.
3) Add to `fastvideo-kernel/CMakeLists.txt` (and any feature flags).
4) Expose in `fastvideo-kernel/python/fastvideo_kernel/ops.py`.
5) Export in `fastvideo-kernel/python/fastvideo_kernel/__init__.py`.

Keep a Python/Triton fallback so the backend runs even when the kernel is not
available.

## 5) Testing and debugging

- Add a small parity test or microbenchmark comparing to SDPA.
- Force your backend with the env var:
  `FASTVIDEO_ATTENTION_BACKEND=MY_NEW_ATTN`.
- Check logs from `fastvideo/attention/selector.py` to confirm selection.

## Checklist

- [ ] Added enum entry in `fastvideo/platforms/interface.py`.
- [ ] Implemented backend in `fastvideo/attention/backends/`.
- [ ] Registered selection in platform(s).
- [ ] Updated layer call sites to include the backend where appropriate.
- [ ] Added tests and documentation.
