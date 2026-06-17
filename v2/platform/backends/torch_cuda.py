"""``cuda`` — the real torch/GPU backend (design_v3 §17; README "Running the real models").

On a CPU box this module is still safe to import: it imports ONLY stdlib at top, gates every cell on
``available=_cuda_available`` (a ``find_spec`` check that never imports torch), and registers thin
**lazy trampolines** whose bodies ``import`` the torch implementations only when actually called. So
``ensure_backends_loaded()`` runs on every box without importing torch, the matrix stays dumpable
(``kernel_matrix()`` / ``component_matrix()`` list these rows ``available=False`` here), and
``resolve_first`` skips them — the CPU mini stays green.

On a real box (torch + CUDA present) ``available`` flips True, ``Platform.detect()`` returns a
``cuda`` platform, and these resolve instead of the numpy/accel rungs:
  * components → the torch adapters in ``torch_adapters.py`` (wrap the real ``fastvideo.models.*``
    module named by the card's ``load_id``; weights from ``ComponentSpec.checkpoint``);
  * the solver ops → plain torch elementwise in ``torch_kernels.py``. (There is NO fused flow-match /
    SDE *solver* kernel in fastvideo-kernel — it ships only primitives — so the honest ``source`` is
    "torch elementwise", at arch ``generic`` with no static scratch.)

WRITTEN-NOT-RUN: the torch code is grounded in the verbatim real APIs but cannot be executed here;
see ``GPU_BRINGUP.md`` for the checklist and the on-box ``# BRINGUP`` confirm points.
"""
from __future__ import annotations

import importlib.util

from ..registry import FLOW_MATCH_STEP, FLOW_SDE_STEP, register_component, register_kernel


def _torch_present() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cuda_available() -> bool:
    if not _torch_present():
        return False
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# --- lazy trampolines: import torch only when actually called (on a GPU box) ------------------- #
def _build_dit(spec, instance, platform):
    from .torch_adapters import build_torch_dit
    return build_torch_dit(spec, instance, platform)


def _build_vae(spec, instance, platform):
    from .torch_adapters import build_torch_vae
    return build_torch_vae(spec, instance, platform)


def _build_text_encoder(spec, instance, platform):
    from .torch_adapters import build_torch_text_encoder
    return build_torch_text_encoder(spec, instance, platform)


def _flow_match_cuda(*args, **kwargs):
    from .torch_kernels import flow_match_step
    return flow_match_step(*args, **kwargs)


def _flow_sde_cuda(*args, **kwargs):
    from .torch_kernels import flow_sde_step
    return flow_sde_step(*args, **kwargs)


# --- components: the real torch adapters (wrap load_id; weights from spec.checkpoint) ----------- #
_ADAPTERS = "v2.platform.backends.torch_adapters"
register_component("dit", _build_dit, device="cuda", available=_cuda_available,
                   source=f"{_ADAPTERS}:TorchWanDiT")
register_component("vae", _build_vae, device="cuda", available=_cuda_available,
                   source=f"{_ADAPTERS}:TorchWanVAE")
register_component("text_encoder", _build_text_encoder, device="cuda", available=_cuda_available,
                   source=f"{_ADAPTERS}:TorchT5Encoder")

# --- solver ops: plain torch elementwise (no fused solver kernel exists in fastvideo-kernel) ---- #
# arch="generic" (elementwise, arch-agnostic), workspace_bytes=0 (no static scratch).
register_kernel(FLOW_MATCH_STEP, _flow_match_cuda, device="cuda", arch="generic",
                available=_cuda_available, source="torch elementwise (no fused solver kernel)")
register_kernel(FLOW_SDE_STEP, _flow_sde_cuda, device="cuda", arch="generic",
                available=_cuda_available, source="torch elementwise (no fused solver kernel)")
