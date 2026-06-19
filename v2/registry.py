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


# Bucket-C ports — each is a self-contained recipe package (card-declared torch adapter via
# ``ComponentSpec.adapter`` + a new/forked loop, no shared-backend edit). One table drives both the
# explicit HF-id registry (PRIMARY) and the ``select_by_architecture`` fallback, so adding a port is one
# row. Rows: (hf_ids, package, card_builder, program_builder, transformer_cls). ``transformer_cls`` is the
# diffusers/EntryClass name for the fallback ("" = explicit-id-only; confirm against the real checkpoint's
# transformer/config.json _class_name). Builders are resolved lazily so ``import v2.registry`` stays cheap
# and torch-free (the recipe packages are CPU-clean, but importing them all eagerly on every resolve is waste).
_BUCKET_C: tuple[tuple, ...] = (
    (("KyleShao/Cosmos-Predict2.5-2B-Diffusers", "nvidia/Cosmos-Predict2.5-14B"), "cosmos25", "build_cosmos25_card",
     "build_cosmos25_program", "Cosmos25Transformer3DModel"),
    (("hunyuanvideo-community/HunyuanVideo", ), "hunyuan_video", "build_hunyuan_video_card",
     "build_hunyuan_video_program", "HunyuanVideoTransformer3DModel"),
    (("FastVideo/FastHunyuan-diffusers", ), "hunyuan_video", "build_fast_hunyuan_video_card",
     "build_hunyuan_video_program", ""),
    (("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
      "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"), "hunyuan_video15",
     "build_hunyuan_video15_card", "build_hunyuan_video15_program", "HunyuanVideo15Transformer3DModel"),
    (("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
      "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
      "weizhou03/HunyuanVideo-1.5-Diffusers-1080p", "weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR"),
     "hunyuan_video15", "build_hunyuan_video15_720p_card", "build_hunyuan_video15_program", ""),
    (("FastVideo/LongCat-Video-T2V-Diffusers", "FastVideo/LongCat-Video-I2V-Diffusers",
      "FastVideo/LongCat-Video-VC-Diffusers"), "longcat", "build_longcat_card", "build_longcat_program",
     "LongCatTransformer3DModel"),
    (("stabilityai/stable-diffusion-3.5-medium", ), "sd35", "build_sd35_card", "build_sd35_program",
     "SD3Transformer2DModel"),
    (("FastVideo/GEN3C-Cosmos-7B-Diffusers", ), "gen3c", "build_gen3c_card", "build_gen3c_program",
     "Gen3CTransformer3DModel"),
    (("kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers", ), "kandinsky5", "build_kandinsky5_card",
     "build_kandinsky5_program", "Kandinsky5Transformer3DModel"),
    (("black-forest-labs/FLUX.2-dev", ), "flux2", "build_flux2_card", "build_flux2_program", "Flux2Transformer2DModel"),
    (("black-forest-labs/FLUX.2-klein-4B", "black-forest-labs/FLUX.2-klein-9B"), "flux2", "build_flux2_klein_card",
     "build_flux2_program", ""),
    (("FastVideo/stable-audio-open-1.0-Diffusers", ), "stable_audio", "build_stable_audio_card",
     "build_stable_audio_program", "StableAudioDiT"),
    (("FastVideo/stable-audio-open-small-Diffusers", ), "stable_audio", "build_stable_audio_small_card",
     "build_stable_audio_program", ""),
    (("FastVideo/HunyuanGameCraft-Diffusers", ), "hunyuangamecraft", "build_hunyuangamecraft_card",
     "build_hunyuangamecraft_program", "HunyuanGameCraftTransformer3DModel"),
    (("FastVideo/HY-WorldPlay-Bidirectional-Diffusers", ), "hyworld", "build_hyworld_card", "build_hyworld_program",
     "HYWorldTransformer3DModel"),
    (("FastVideo/LingBot-World-Base-Cam-Diffusers", ), "lingbotworld", "build_lingbotworld_card",
     "build_lingbotworld_program", "LingBotWorldTransformer3DModel"),
    (("FastVideo/Matrix-Game-2.0-Base-Distilled-Diffusers", "FastVideo/Matrix-Game-2.0-GTA-Distilled-Diffusers",
      "FastVideo/Matrix-Game-2.0-TempleRun-Distilled-Diffusers", "FastVideo/Matrix-Game-2.0-Base-Diffusers",
      "FastVideo/Matrix-Game-2.0-GTA-Diffusers", "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers"), "matrixgame2",
     "build_matrixgame2_card", "build_matrixgame2_program", "CausalMatrixGame2WanModel"),
    (("FastVideo/Matrix-Game-3.0-Base-Distilled-Diffusers", ), "matrixgame3", "build_matrixgame3_card",
     "build_matrixgame3_program", "MatrixGame3WanModel"),
    # Residual Wan-family variants — reuse the Wan/Causal ARCH (NO new torch adapter; a new in-package
    # sampler/loop/conditioning). transformer_cls="" -> explicit-HF-id-ONLY: the generic Wan/Causal arch
    # fallback already serves unregistered Wan checkpoints, and only the exact id distinguishes these
    # capability variants (rCM/DMD few-step, v2v, control, causal-MoE) from a base Wan of the same class.
    (("loayrashid/TurboWan2.1-T2V-1.3B-Diffusers", ), "turbowan", "build_turbowan_card", "build_turbowan_program", ""
     ),  # rCM 4-step
    (("loayrashid/TurboWan2.2-I2V-A14B-Diffusers", ), "turbowan", "build_turbowan_i2v_a14b_card",
     "build_turbowan_i2v_program", ""),  # rCM MoE i2v
    (("decart-ai/Lucy-Edit-Dev", "decart-ai/Lucy-Edit-1.1-Dev"), "lucy_edit", "build_lucy_edit_card",
     "build_lucy_edit_program", ""),  # v2v editor
    (("IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers", ), "wan_fun_control", "build_wan_fun_control_card",
     "build_wan_fun_control_program", ""),  # control input
    (("FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers", ), "sfwan22", "build_sfwan22_i2v_a14b_card",
     "build_sfwan22_i2v_program", ""),  # causal MoE i2v
    (("rand0nmr/SFWan2.2-T2V-A14B-Diffusers", ), "sfwan22", "build_sfwan22_t2v_a14b_card", "build_sfwan22_t2v_program",
     ""),  # causal MoE t2v
    (("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "FastVideo/FastWan2.2-TI2V-5B-Diffusers"), "fastwan",
     "build_fastwan_card", "build_fastwan_program", ""),  # DMD 3-step (FullAttn loadable; VSA BRINGUP)
    (("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers"), "fastwan",
     "build_fastwan_t2v_1_3b_card", "build_fastwan_program", ""),  # DMD (VSA + non-strict load BRINGUP)
)


def _lazy(package: str, fn: str, *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """A ``() -> card_or_program`` builder that imports the recipe package only when first called.
    ``args``/``kwargs`` are forwarded to the builder (e.g. a per-id resolution/model_id override)."""

    def _build() -> Any:
        import importlib
        return getattr(importlib.import_module(f"v2.recipes.{package}"), fn)(*args, **kwargs)

    return _build


def _bucket_c_entries() -> list[ModelEntry]:
    return [ModelEntry(hf_ids, _lazy(pkg, cb), _lazy(pkg, pb)) for (hf_ids, pkg, cb, pb, _cls) in _BUCKET_C]


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
    return [  # noqa: RUF005 — explicit list, extended with the bucket-C ports below via _bucket_c_entries()
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
        # LTX-2 / LTX-2.3 repo aliases (naming variants of the already-registered LTX checkpoints; the
        # arch fallback also resolves LTX2Transformer3DModel from a checkpoint root, refining base-vs-
        # two-stage by the spatial_upsampler). LTX-2 -> single-stage base; LTX-2.3 -> the distilled joint
        # A/V card (a non-distilled 2.3 base checkpoint reuses it — override num_steps for many-step).
        ModelEntry(("FastVideo/LTX2-Diffusers", "FastVideo/LTX2-base", "Lightricks/LTX-2"), build_ltx2_base_card,
                   build_ltx2_base_program),
        ModelEntry(("FastVideo/LTX2.3-Diffusers", "FastVideo/LTX2.3-Distilled-Diffusers", "FastVideo/LTX2.3-base",
                    "Lightricks/LTX-2.3", "lightricks/ltx-2.3"), build_ltx2_3_card, build_ltx2_3_program),
        # Cosmos-Predict2-2B-Video2World — EDM-Karras denoiser (new CosmosDenoiseLoop + CosmosDiT adapter),
        # reusing the Wan VAE adapter + T5. Registered for t2v (video2world conditioning threads later).
        ModelEntry(("nvidia/Cosmos-Predict2-2B-Video2World", ), build_cosmos2_card, build_cosmos2_program),
        # TurboWan2.1-T2V-14B: same rCM card as the 1.3B but the 720p resolution default (args override).
        ModelEntry(("loayrashid/TurboWan2.1-T2V-14B-Diffusers", ),
                   _lazy("turbowan", "build_turbowan_card", "turbowan2.1-t2v-14b", height=720, width=1280),
                   _lazy("turbowan", "build_turbowan_program")),
    ] + _bucket_c_entries()  # the bucket-C + Wan-variant self-contained recipe packages (rows in _BUCKET_C)


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
    for (_hf, pkg, cb, pb, cls) in _BUCKET_C:  # the bucket-C ports, keyed by transformer cls
        if cls and tr == cls:
            import importlib
            mod = importlib.import_module(f"v2.recipes.{pkg}")
            return getattr(mod, cb), getattr(mod, pb)
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
