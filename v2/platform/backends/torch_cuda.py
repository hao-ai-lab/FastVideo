"""``cuda`` registration — the real torch/GPU backend (see v2/README.md "Running the real models").

Torch-free at import (stdlib only): gates every cell on ``available=_cuda_available`` (a ``find_spec``
check that never imports torch) and registers ONE lazy component builder for all cuda component kinds +
the two solver ops. So ``ensure_backends_loaded()`` runs on every box without importing torch, the matrix
stays dumpable (these rows ``available=False`` on a CPU box), and ``resolve_first`` skips them.

On a real box ``available`` flips True and these resolve: components -> the single generic builder in
``torch_backend.py`` (which dispatches by ``spec.kind``, wraps the real ``fastvideo.models.*`` module the
card's load_id names — weights from ``ComponentSpec.checkpoint`` via the ``v2.loader`` seam); the solver
ops -> plain torch elementwise in ``torch_kernels.py`` (no fused solver kernel exists in fastvideo-kernel).
"""
from __future__ import annotations

import importlib.util

from v2.platform.registry import FLOW_MATCH_STEP, FLOW_SDE_STEP, register_component, register_kernel


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


# --- ONE lazy builder for every cuda component kind: imports torch only when called (on a GPU box) --- #
def _build_component(spec, instance, platform):
    from v2.platform.backends.torch_backend import build_component
    return build_component(spec, instance, platform)


def _flow_match_cuda(*args, **kwargs):
    from v2.platform.backends.torch_kernels import flow_match_step
    return flow_match_step(*args, **kwargs)


def _flow_sde_cuda(*args, **kwargs):
    from v2.platform.backends.torch_kernels import flow_sde_step
    return flow_sde_step(*args, **kwargs)


# --- components: one generic builder; torch_backend.build_component dispatches by spec.kind ---------- #
_B = "v2.platform.backends.torch_backend"
_KIND_SOURCE = {
    "dit": "WanDiT/LTX2DiT",
    "vae": "WanVAE/LTX2VAE",
    "text_encoder": "T5Encoder/Gemma",
    "image_encoder": "CLIPImageEncoder",
    "upsampler": "LTX2Upsampler",
    "audio_vae": "LTX2AudioVAE",
    "vocoder": "LTX2Vocoder",
}
for _kind, _cls in _KIND_SOURCE.items():
    register_component(_kind, _build_component, device="cuda", available=_cuda_available, source=f"{_B}:{_cls}")
del _kind, _cls

# --- solver ops: plain torch elementwise (no fused solver kernel in fastvideo-kernel) ---------------- #
register_kernel(FLOW_MATCH_STEP,
                _flow_match_cuda,
                device="cuda",
                arch="generic",
                available=_cuda_available,
                source="torch elementwise (no fused solver kernel)")
register_kernel(FLOW_SDE_STEP,
                _flow_sde_cuda,
                device="cuda",
                arch="generic",
                available=_cuda_available,
                source="torch elementwise (no fused solver kernel)")
