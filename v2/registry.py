"""v2 model registry — the SHARED resolution of a model (HF repo id or local checkpoint dir) to its
``(build_card, build_program)`` builders, so every v2 entrypoint (the typed ``VideoGenerator``, a CLI,
the server) uses one source of truth instead of each re-deriving the mapping.

Mirrors fastvideo's ``fastvideo/registry.py`` hybrid resolution:

  1. exact HF repo id in the explicit registry   — PRIMARY: correct per-model card + capabilities, and
                                                    the only way to split same-architecture *capability*
                                                    variants (e.g. a Wan2.1 T2V base vs the i2v "InP"
                                                    1.3B) that pure architecture inference cannot tell apart.
  2. short repo-name match                        — a renamed / forked copy of a known repo.
  3. architecture inference from the checkpoint   — FALLBACK: local paths / unregistered repos, keyed on
                                                    the pipeline / transformer / VAE class names.

Adding a model = one ``ModelEntry`` below (its HF ids -> builders). New architectures (for the fallback)
are added in ``select_by_architecture``. All builders are CPU-clean v2 submodules imported lazily, so
``import v2.registry`` stays cheap and torch-free.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable


@dataclass(frozen=True)
class ModelEntry:
    """A registered model family: the exact HF repo ids it serves + its card/program builders."""
    hf_ids: tuple[str, ...]
    build_card: Callable[[], Any]
    build_program: Callable[[], Any]


def _entries() -> list[ModelEntry]:
    """The explicit registry (PRIMARY). Builders imported lazily to avoid import cycles with the model
    packages and keep importing this module cheap."""
    from v2.recipes.ltx2 import (
        build_ltx2_3_card,
        build_ltx2_3_program,
        build_ltx2_base_card,
        build_ltx2_base_program,
        build_ltx2_card,
        build_ltx2_program,
    )
    from v2.recipes.wan_causal import build_wan_causal_card, build_wan_causal_program
    from v2.recipes.wan21 import (
        build_wan21_card,
        build_wan22_a14b_card,
        build_wan22_ti2v_card,
        build_wan_t2v_14b_card,
        build_wan_t2v_program,
    )
    from v2.recipes.wan21.i2v import build_wan21_i2v_card, build_wan21_i2v_program, build_wan22_i2v_a14b_card
    from v2.recipes.cosmos2 import build_cosmos2_card, build_cosmos2_program
    return [
        ModelEntry(("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ), build_wan21_card, build_wan_t2v_program),
        ModelEntry(("Wan-AI/Wan2.1-T2V-14B-Diffusers", ), build_wan_t2v_14b_card, build_wan_t2v_program),
        # Wan2.1 i2v cluster: CLIP image encoder + first-frame VAE conditioning ([mask|cond] -> 36ch DiT).
        # Fun-1.3B-InP is GPU-verified; the 14B variants reuse the same i2v card/path (weights GPU-pending).
        ModelEntry(("weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers", ), build_wan21_i2v_card, build_wan21_i2v_program),
        ModelEntry(("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", ),
                   lambda: build_wan21_i2v_card("wan2.1-i2v-14b-480p", flow_shift=3.0, height=480, width=832),
                   build_wan21_i2v_program),
        ModelEntry(("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", ),
                   lambda: build_wan21_i2v_card("wan2.1-i2v-14b-720p", flow_shift=5.0, height=720, width=1280),
                   build_wan21_i2v_program),
        # Wan2.2-I2V-A14B: MoE (2 experts + boundary) + i2v conditioning. Reuses Wan adapter + i2v + MoE.
        ModelEntry(("Wan-AI/Wan2.2-I2V-A14B-Diffusers", ), build_wan22_i2v_a14b_card, build_wan21_i2v_program),
        ModelEntry(("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ), build_wan22_ti2v_card, build_wan_t2v_program),
        ModelEntry(("Wan-AI/Wan2.2-T2V-A14B-Diffusers", ), build_wan22_a14b_card, build_wan_t2v_program),
        ModelEntry(("wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers", ), build_wan_causal_card, build_wan_causal_program),
        # LTX-2 two-stage distilled (ships a spatial_upsampler: base -> upsample -> refine):
        ModelEntry(("FastVideo/LTX2-Distilled-Diffusers", ), build_ltx2_card, build_ltx2_program),
        # LTX-2 single-stage, non-distilled base (request-driven many-step, video-only):
        ModelEntry(("Davids048/LTX2-Base-Diffusers", ), build_ltx2_base_card, build_ltx2_base_program),
        # LTX-2.3 distilled — single-stage, JOINT text->video+audio (separate connectors + gated attn):
        ModelEntry(("FastVideo/LTX-2.3-Distilled-Diffusers", ), build_ltx2_3_card, build_ltx2_3_program),
        # Cosmos-Predict2-2B-Video2World — EDM-Karras denoiser (new CosmosDenoiseLoop + CosmosDiT adapter),
        # reusing the Wan VAE adapter + T5. Registered for t2v (video2world conditioning threads later).
        ModelEntry(("nvidia/Cosmos-Predict2-2B-Video2World", ), build_cosmos2_card, build_cosmos2_program),
    ]


def _short(p: str) -> str:
    return p.rstrip("/").split("/")[-1].lower()


def read_arch_signature(root: str) -> dict:
    """Read a diffusers checkpoint's configs into an architecture signature (the FALLBACK dispatch
    keys): pipeline / transformer / VAE class names + the fields that split same-class variants
    (VAE ``z_dim``, a second transformer / ``boundary_ratio`` for MoE, a ``spatial_upsampler``).
    ``root`` is a local diffusers dir or a snapshot of just the ``*.json`` configs."""

    def _load(*parts: str) -> dict:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return {}

    mi = _load("model_index.json")
    tcfg = _load("transformer", "config.json")
    vcfg = _load("vae", "config.json")
    return {
        "pipeline": mi.get("_class_name"),
        "boundary_ratio": mi.get("boundary_ratio"),
        "has_transformer_2": os.path.isdir(os.path.join(root, "transformer_2")),
        "has_spatial_upsampler": os.path.isdir(os.path.join(root, "spatial_upsampler")) or "spatial_upsampler" in mi,
        "transformer_cls": tcfg.get("_class_name"),
        "in_channels": tcfg.get("in_channels"),
        "vae_z_dim": vcfg.get("z_dim", vcfg.get("latent_channels")),
    }


def select_by_architecture(sig: dict):
    """FALLBACK: map an architecture signature -> (build_card, build_program) by class names, so a local
    path / renamed repo / new distilled variant of a known arch still resolves with no registry entry."""
    tr, pipe = sig.get("transformer_cls"), sig.get("pipeline")
    if tr == "CosmosTransformer3DModel":
        from v2.recipes.cosmos2 import build_cosmos2_card, build_cosmos2_program
        return build_cosmos2_card, build_cosmos2_program
    if tr == "LTX2Transformer3DModel":
        from v2.recipes.ltx2 import (
            build_ltx2_base_card,
            build_ltx2_base_program,
            build_ltx2_card,
            build_ltx2_program,
        )
        if sig.get("has_spatial_upsampler"):
            return build_ltx2_card, build_ltx2_program  # two-stage (base -> upsample -> refine)
        return build_ltx2_base_card, build_ltx2_base_program  # single-stage
    if tr == "CausalWanTransformer3DModel":
        from v2.recipes.wan_causal import build_wan_causal_card, build_wan_causal_program
        return build_wan_causal_card, build_wan_causal_program
    if tr == "WanTransformer3DModel":
        from v2.recipes.wan21 import (
            build_wan21_card,
            build_wan22_a14b_card,
            build_wan22_ti2v_card,
            build_wan_t2v_program,
        )
        if pipe == "WanDMDPipeline":  # FastWan: detected by pipeline class -> precise, not a load crash
            raise ValueError("v2 registry: WanDMD/FastWan is not supported via the generic Wan path — its checkpoint's "
                             "to_gate_compress param mapping differs from the generic WanTransformer3DModel load. See "
                             "examples/inference/basic/V2_PORTING_STATUS.md.")
        if sig.get("has_transformer_2") or sig.get("boundary_ratio"):
            return build_wan22_a14b_card, build_wan_t2v_program  # Wan2.2 MoE (two experts)
        if sig.get("vae_z_dim") == 48:
            return build_wan22_ti2v_card, build_wan_t2v_program  # Wan2.2-TI2V-5B (z_dim=48 VAE)
        return build_wan21_card, build_wan_t2v_program  # Wan2.1
    raise ValueError(f"v2 registry: unsupported architecture (transformer={tr!r}, pipeline={pipe!r}). Supported "
                     f"transformers: WanTransformer3DModel / CausalWanTransformer3DModel / LTX2Transformer3DModel. "
                     f"See examples/inference/basic/V2_PORTING_STATUS.md.")


def resolve(model_path: str, root: str | None = None):
    """Resolve ``model_path`` (an HF repo id or a local checkpoint dir) -> ``(build_card, build_program)``.

    PRIMARY: an exact HF-id, then a short repo-name match, in the explicit registry. FALLBACK: when
    ``model_path`` isn't registered, architecture inference from ``root`` (a local dir or a downloaded
    ``*.json`` config snapshot). Raises ``ValueError`` if unregistered with ``root=None`` (so the caller
    knows to fetch the configs), or if the architecture is unsupported. Shared by every v2 entrypoint."""
    entries = _entries()
    for e in entries:  # 1. exact HF id
        if model_path in e.hf_ids:
            return e.build_card, e.build_program
    short = _short(model_path)  # 2. short repo name (renamed / forked copy)
    for e in entries:
        if any(_short(h) == short for h in e.hf_ids):
            return e.build_card, e.build_program
    if root is None:  # 3. architecture inference (needs the configs)
        raise ValueError(f"v2 registry: {model_path!r} is not registered; pass the checkpoint root for architecture "
                         f"inference, or add a ModelEntry in v2/registry.py.")
    return select_by_architecture(read_arch_signature(root))
