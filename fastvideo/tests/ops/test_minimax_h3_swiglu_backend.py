# SPDX-License-Identifier: Apache-2.0
"""CPU checks for H3 SwiGLU GEMM backend resolution (no GPU, no native kernel)."""
import builtins
from types import SimpleNamespace


def _fake_import(real_import, binding):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "_fastvideo_h3_swiglu_gemm":
            raise ImportError("the Spark prebuild module is absent")
        if name == "fastvideo_kernel" and "h3_swiglu_gemm_binding" in (fromlist or ()):
            if isinstance(binding, BaseException):
                raise binding
            return SimpleNamespace(h3_swiglu_gemm_binding=binding)
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def test_backend_returns_none_when_the_binding_library_is_missing(monkeypatch):
    """A wheel ships the ctypes binding but not its .so; importing it raises OSError."""
    from fastvideo.models.dits.minimax_h3_fusions import swiglu_gemm

    missing_library = OSError("lib_fastvideo_h3_swiglu_gemm.so: cannot open shared object file")
    monkeypatch.setattr(builtins, "__import__", _fake_import(builtins.__import__, missing_library))
    swiglu_gemm._backend.cache_clear()
    try:
        assert swiglu_gemm._backend() is None
    finally:
        swiglu_gemm._backend.cache_clear()


def test_backend_resolves_the_in_tree_binding(monkeypatch):
    """When the prebuild module is absent but the binding imports, _backend returns it."""
    from fastvideo.models.dits.minimax_h3_fusions import swiglu_gemm

    binding = object()
    monkeypatch.setattr(builtins, "__import__", _fake_import(builtins.__import__, binding))
    swiglu_gemm._backend.cache_clear()
    try:
        assert swiglu_gemm._backend() is binding
    finally:
        swiglu_gemm._backend.cache_clear()
