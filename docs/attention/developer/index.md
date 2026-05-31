# Adding a New Attention Backend

FastVideo allows integrating new attention mechanisms easily. This guide walks you through adding a new backend (e.g., `MyNewAttn`).

## 1. Implement the Backend (Python)

Create a new file in `fastvideo/attention/backends/` (e.g., `mynew_attn.py`).

Your implementation should inherit from `AttentionBackend` defined in `abstract.py`.

```python
# fastvideo/attention/backends/mynew_attn.py
import torch
from .abstract import AttentionBackend
# Import the context manager to access metadata (optional)
from fastvideo.forward_context import get_forward_context

# Import compiled kernel if applicable (see Section 2)
try:
    # Import from the top-level package
    from fastvideo_kernel import my_compiled_attn_func
except ImportError:
    my_compiled_attn_func = None

class MyNewAttnBackend(AttentionBackend):
    def process_inputs(self, q, k, v, **kwargs):
        # Pre-process inputs if necessary
        return q, k, v

    def forward(self, q, k, v, **kwargs):
        # Optional: Access extra metadata passed via ForwardContext
        # Only needed if your backend requires global state (e.g. window_size)
        try:
            context = get_forward_context()
            metadata = context.attn_metadata
            # Example: window_size = metadata.window_size
        except (AssertionError, AttributeError):
            # Handle case where context is not set (e.g. standard inference)
            pass

        if my_compiled_attn_func is not None:
            return my_compiled_attn_func(q, k, v)
        else:
            # Fallback implementation (e.g., Triton or pure PyTorch)
            return self.fallback_impl(q, k, v)
```

## 2. Passing Extra Information via ForwardContext (Optional)

FastVideo uses a `ForwardContext` to pass global metadata (like current timestep, batch info, or custom attention configurations) to attention backends without changing the `forward` signature of every layer. **This is optional and only required if your backend needs dynamic per-step information.**

To use this:

1. **Set Context**: In your pipeline or generation loop, use the `set_forward_context` context manager.
2. **Access Context**: Inside your attention backend, use `get_forward_context()`.

See [`docs/attention/sta/index.md`](../sta/index.md) for a legacy STA example
of passing complex configuration (window sizes) through `ForwardContext`.

## 3. Adding Compiled Kernels (C++/CUDA)

If your backend requires custom CUDA kernels, you need to add them to the `fastvideo-kernel` package.

### A. Add Source Files

Place your kernel implementation files in `fastvideo-kernel/csrc/attention/`.

* `mynew_attn.cu` (CUDA implementation)
* `mynew_attn.h` (Optional headers)

### B. Register in Extension

Update `fastvideo-kernel/csrc/common_extension.cpp` to expose your function to Python.

```cpp
// 1. Declare external function
#ifdef COMPILE_MYNEW_ATTN
extern torch::Tensor mynew_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
#endif

// 2. Register in module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ... other kernels ...

#ifdef COMPILE_MYNEW_ATTN
    m.def("mynew_attn_fwd", torch::wrap_pybind_function(mynew_attn_forward), "My New Attention Forward");
#endif
}
```

### C. Update CMakeLists.txt

Update `fastvideo-kernel/CMakeLists.txt` to compile your new files.

**Case 1: General CUDA Kernel (Runs on all GPUs)**
Add your source file directly to `EXTENSION_SOURCES` and define the compilation flag.

```cmake
# Add to EXTENSION_SOURCES
list(APPEND EXTENSION_SOURCES csrc/attention/mynew_attn.cu)

# Add compilation definition for common_extension.cpp
list(APPEND COMPILE_DEFS COMPILE_MYNEW_ATTN)
```

**Case 2: ThunderKittens Kernel (Hopper H100 Only)**
If your kernel uses ThunderKittens (TK), it requires specific architecture flags (`sm_90a`). Add it inside the `ENABLE_TK_KERNELS` block.

```cmake
if(ENABLE_TK_KERNELS)
    # Add source only if TK is enabled
    list(APPEND EXTENSION_SOURCES csrc/attention/mynew_attn_tk.cu)
    
    # Add definition to guard registration
    list(APPEND COMPILE_DEFS TK_COMPILE_MYNEW_ATTN)
endif()
```

### D. Expose in Python Ops

Update `fastvideo-kernel/python/fastvideo_kernel/ops.py` to make the function importable and handle fallbacks gracefully.

```python
# fastvideo-kernel/python/fastvideo_kernel/ops.py

# Try to load C++ extension symbols
try:
    from fastvideo_kernel._C import fastvideo_kernel_ops
    mynew_attn_fwd = getattr(fastvideo_kernel_ops, "mynew_attn_fwd", None)
except ImportError:
    mynew_attn_fwd = None

def my_compiled_attn_func(q, k, v):
    # Runtime check: use C++ kernel if available, else fallback
    if mynew_attn_fwd is not None:
        return mynew_attn_fwd(q, k, v)
    else:
        # Call Triton/Python fallback
        return mynew_attn_triton(q, k, v)
```

### E. Expose in Package Init

Update `fastvideo-kernel/python/fastvideo_kernel/__init__.py` to export the function.

```python
from fastvideo_kernel.ops import (
    my_compiled_attn_func,
    # ...
)

__all__ = [
    "my_compiled_attn_func",
    # ...
]
```

## 4. Register the Backend

Update `fastvideo/attention/backends/__init__.py` to export your new class.

```python
from .mynew_attn import MyNewAttnBackend
```

## 5. Platform Integration

If your backend requires specific platform checks (e.g., checking for H100 support), handle that in `fastvideo/platforms/cuda.py` or within your backend's `__init__`.

## 6. Add Documentation

Create a new documentation page for your backend to explain its usage, installation (if custom kernels are needed), and features.

1. **Create Directory**: `docs/attention/mynew_attn/`
2. **Create Index**: `docs/attention/mynew_attn/index.md`
3. **Update Navigation**: Add an entry to `mkdocs.yml` under the "Attention" tab.

## Checklist

* [ ] Created `fastvideo/attention/backends/mynew_attn.py`.
* [ ] (Optional) Added CUDA kernels in `fastvideo-kernel/csrc/attention/`.
* [ ] (Optional) Updated `common_extension.cpp` and `CMakeLists.txt`.
* [ ] (Optional) Exposed kernel in `fastvideo-kernel/python/fastvideo_kernel/ops.py`.
* [ ] (Optional) Exported kernel in `fastvideo-kernel/python/fastvideo_kernel/__init__.py`.
* [ ] Implemented `forward` method respecting the standard signature.
* [ ] Added unit tests in `tests/`.
* [ ] Added documentation in `docs/attention/` and updated `mkdocs.yml`.
