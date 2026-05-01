"""VBench metrics. Bootstraps upstream submodule on sys.path and
installs runtime compat shims for modern torch/transformers/numpy/timm.

The upstream vbench source lives as a git submodule at
``fastvideo/third_party/eval/vbench`` (pinned to a specific
Vchitect/VBench SHA). We do not pip-install it — we only need its
Python modules importable. Its runtime deps (clip, transformers, etc.)
are already in FastVideo's main env.

Compat with modern dependency versions is achieved at import time, in
this file, instead of via on-disk patches to upstream files. Each shim
below corresponds to a specific drift between vbench's pinned-2023 deps
and FastVideo's current pins. Adding a new shim is preferable to editing
the submodule.
"""

from __future__ import annotations

import sys
from pathlib import Path

# fastvideo/eval/metrics/vbench/__init__.py  →  ../../../third_party/eval/vbench
# parents[3] is the ``fastvideo/`` package root.
_UPSTREAM = Path(__file__).resolve().parents[3] / "third_party" / "eval" / "vbench"
if _UPSTREAM.is_dir() and str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))


def _install_compat_shims() -> None:
    """Apply attribute-level shims that make vbench imports resolve.

    Idempotent and side-effect-free if the targeted modules are already
    correct (e.g. on older transformers/numpy).
    """
    # transformers: apply_chunking_to_forward & friends moved from
    # ``transformers.modeling_utils`` to ``transformers.pytorch_utils``
    # (transformers ~= 4.30+). Mirror them back so vbench's legacy
    # ``from transformers.modeling_utils import (...)`` keeps resolving.
    try:
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu
        for _name in ("apply_chunking_to_forward",
                      "find_pruneable_heads_and_indices",
                      "prune_linear_layer"):
            if not hasattr(_mu, _name) and hasattr(_pu, _name):
                setattr(_mu, _name, getattr(_pu, _name))
    except ImportError:
        pass

    # numpy.lib.function_base was removed entirely in numpy>=2; vbench's
    # umt/kinetics still does ``from numpy.lib.function_base import disp``
    # but never calls disp. Install a stub submodule with a no-op ``disp``
    # so the legacy import line resolves.
    try:
        import types
        import numpy.lib as _nl
        if not hasattr(_nl, "function_base"):
            _stub = types.ModuleType("numpy.lib.function_base")
            _stub.disp = lambda *a, **k: None  # type: ignore[attr-defined]
            sys.modules["numpy.lib.function_base"] = _stub
            _nl.function_base = _stub  # type: ignore[attr-defined]
    except ImportError:
        pass


def _install_modeling_finetune_hook() -> None:
    """Wrap vbench's ``vit_large_patch16_224`` to drop the ``cache_dir``
    kwarg that newer timm passes to model factory functions but the
    upstream factory doesn't accept. Installed as a meta-path finder so
    we patch the attribute on the actual module object after it loads,
    without eagerly importing torch+timm at fastvideo.eval import time.
    """
    import importlib.abc

    _target = "vbench.third_party.umt.models.modeling_finetune"

    class _Loader(importlib.abc.Loader):
        def __init__(self, real_loader):
            self._real = real_loader

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            self._real.exec_module(module)
            orig = getattr(module, "vit_large_patch16_224", None)
            if orig is None or getattr(orig, "_fastvideo_patched", False):
                return

            def patched(pretrained=False, **kwargs):
                kwargs.pop("cache_dir", None)
                return orig(pretrained=pretrained, **kwargs)

            patched._fastvideo_patched = True  # type: ignore[attr-defined]
            module.vit_large_patch16_224 = patched

    class _Finder(importlib.abc.MetaPathFinder):
        _reentrant = False

        def find_spec(self, fullname, path, target=None):
            if fullname != _target or self._reentrant:
                return None
            self._reentrant = True
            try:
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        spec.loader = _Loader(spec.loader)
                        return spec
                return None
            finally:
                self._reentrant = False

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


_install_compat_shims()
_install_modeling_finetune_hook()
