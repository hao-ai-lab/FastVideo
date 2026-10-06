# SPDX-License-Identifier: Apache-2.0
"""Loads the k6_video reference (``kandinsky_sr``) for the Kandinsky6 SR parity tests.

Local-only tooling: the reference package is not a FastVideo dependency.  It is found via
``$KANDINSKY_SR_SRC`` (the ``k6_video/src`` directory) or a ``k6_video`` checkout next to this repository.  On a
machine without CUDA / flash-attn / loguru small stubs are installed (per-sequence SDPA instead of flash-attn varlen,
no-op logger) and the base resolutions are shrunk, so the whole reference pipeline runs on CPU with tiny random
models.  On a GPU box with the real packages nothing is stubbed.

IMPORTANT: import FastVideo (``import fastvideo...``) BEFORE calling :func:`load_reference`; diffusers probes
``flash_attn`` at import time and chokes on a stub module.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "fastvideo" / "tests" / "stages" / "kandinsky6_sr"))

import k6_sr_tiny  # noqa: E402  (tiny configs shared with the CPU unit tests)

TINY_RESOLUTIONS = k6_sr_tiny.TINY_RESOLUTIONS


def find_reference_src() -> Path | None:
    """``$KANDINSKY_SR_SRC`` (``none`` disables the lookup, forcing the tests to skip) or ``../k6_video/src``."""
    if os.environ.get("KANDINSKY_SR_SRC", "").lower() == "none":
        return None
    candidates = [os.environ.get("KANDINSKY_SR_SRC"), str(REPO_ROOT.parent / "k6_video" / "src")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "kandinsky_sr").is_dir():
            return Path(candidate)
    return None


def _varlen_sdpa(q, k, v, cu_q, cu_k):
    import torch.nn.functional as F

    outs = []
    for i in range(len(cu_q) - 1):
        qi = q[int(cu_q[i]):int(cu_q[i + 1])].transpose(0, 1).unsqueeze(0)
        ki = k[int(cu_k[i]):int(cu_k[i + 1])].transpose(0, 1).unsqueeze(0)
        vi = v[int(cu_k[i]):int(cu_k[i + 1])].transpose(0, 1).unsqueeze(0)
        outs.append(F.scaled_dot_product_attention(qi, ki, vi).squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


def _install_stubs() -> None:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if "flash_attn" not in sys.modules:
        try:
            importlib.import_module("flash_attn")
        except ImportError:
            mod = types.ModuleType("flash_attn")

            def varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                       softmax_scale=None, causal=False, return_attn_probs=False, **_):
                out = _varlen_sdpa(q, k, v, cu_seqlens_q, cu_seqlens_k)
                return (out, torch.zeros(1), None) if return_attn_probs else out

            def qkvpacked(qkv, cu_seqlens, max_seqlen, dropout_p=0.0, softmax_scale=None, causal=False,
                          return_attn_probs=False, **_):
                q, k, v = qkv.unbind(dim=1)
                out = _varlen_sdpa(q, k, v, cu_seqlens, cu_seqlens)
                return (out, torch.zeros(1), None) if return_attn_probs else out

            mod.flash_attn_varlen_func = varlen
            mod.flash_attn_varlen_qkvpacked_func = qkvpacked
            sys.modules["flash_attn"] = mod
    if "loguru" not in sys.modules:
        try:
            importlib.import_module("loguru")
        except ImportError:
            mod = types.ModuleType("loguru")

            class _Logger:

                def __getattr__(self, name):
                    return lambda *a, **k: self

            mod.logger = _Logger()
            sys.modules["loguru"] = mod


@dataclass
class Reference:
    """Handles on the reference modules used by the parity tests."""

    constants: Any
    dit_mod: Any
    dx_dit_mod: Any
    kvae_mod: Any
    sr_pipeline: Any
    stages: Any
    upscale_utils: Any
    lu_algo: Any
    lu_config: Any
    lu_factory: Any
    piflow_sampler: Any
    algo_utils: Any
    tile_grid: Any
    tiling_utils: Any
    video_io: Any
    tiny: bool


_reference: Reference | None = None


def load_reference(tiny: bool | None = None) -> Reference:
    """Import the reference (skips the calling test when it is unavailable).

    ``tiny`` (default: True unless CUDA is available) shrinks ``constants.RESOLUTIONS`` in place to 64/96 px bases;
    it only takes effect on the first call (tests needing the trained bases patch the dict, see the tiling tests).
    """
    global _reference
    src = find_reference_src()
    if src is None:
        pytest.skip("k6_video reference (kandinsky_sr) not found: set KANDINSKY_SR_SRC to the k6_video/src directory")
    if tiny is None:
        tiny = not torch.cuda.is_available()
    if _reference is None:
        _install_stubs()
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        try:
            imp = importlib.import_module
            _reference = Reference(
                constants=imp("kandinsky_sr.constants"),
                dit_mod=imp("kandinsky_sr.core.components.model.dit"),
                dx_dit_mod=imp("kandinsky_sr.core.components.model.dx_dit"),
                kvae_mod=imp("kandinsky_sr.core.components.video_kvae.cached_model"),
                sr_pipeline=imp("kandinsky_sr.pipeline.sr_pipeline"),
                stages=imp("kandinsky_sr.pipeline.stages"),
                upscale_utils=imp("kandinsky_sr.pipeline.upscale_utils"),
                lu_algo=imp("kandinsky_sr.core.algo.latent_upscaler"),
                lu_config=imp("kandinsky_sr.core.components.latent_upscaler.config"),
                lu_factory=imp("kandinsky_sr.core.components.latent_upscaler.model.factory"),
                piflow_sampler=imp("kandinsky_sr.core.algo.piflow_sampler"),
                algo_utils=imp("kandinsky_sr.core.algo.utils"),
                tile_grid=imp("kandinsky_sr.pipeline.tile_grid"),
                tiling_utils=imp("kandinsky_sr.core.algo.tiling_utils"),
                video_io=imp("kandinsky_sr.pipeline.video_io"),
                tiny=tiny,
            )
        except Exception as exc:  # missing optional dependency of the reference on this machine
            pytest.skip(f"k6_video reference cannot be imported here: {exc!r}")
        if tiny:
            _reference.constants.RESOLUTIONS.clear()
            _reference.constants.RESOLUTIONS.update(TINY_RESOLUTIONS)
    return _reference


def randomize(module: torch.nn.Module, seed: int, std: float = 0.05) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * std)


def build_reference_dit(ref: Reference, piflow: dict | None = k6_sr_tiny.TINY_PIFLOW, dit_cfg: dict | None = None,
                        seed: int = 0):
    cfg = copy.deepcopy(dit_cfg or k6_sr_tiny.TINY_DIT_CFG)
    cfg.pop("sr_params", None)  # bundle metadata, not constructor arguments of the reference model
    cfg.pop("attribute_overrides", None)
    cfg["patch_size"] = tuple(cfg["patch_size"])
    cfg["axes_dims"] = tuple(cfg["axes_dims"])
    cfg["attention_params"] = {512: {"type": "flash"}, "512": {"type": "flash"}}
    if piflow is not None:
        dit = ref.dx_dit_mod.DXDiTWrapper(cfg, out_visual_dim=cfg["out_visual_dim"], n_grid=piflow["n_grid"]).eval()
        dit.piflow_params = dict(piflow)
    else:
        dit = ref.dit_mod.get_dit(cfg).eval()
    randomize(dit, seed + 2)
    return dit


def load_into_port(reference_module: torch.nn.Module, port_module: torch.nn.Module, prefix: str = "") -> None:
    """Load ``reference.state_dict()`` (the unprefixed official Diffusers key layout) into a port module, strictly."""
    from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

    state = {f"{prefix}{key}": value for key, value in reference_module.state_dict().items()}
    mapping = get_param_names_mapping(getattr(port_module, "param_names_mapping", {}))
    custom, _ = hf_to_custom_state_dict(state, mapping)
    port_module.load_state_dict(custom, strict=True)
