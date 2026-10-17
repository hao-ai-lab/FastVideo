# SPDX-License-Identifier: Apache-2.0
"""Convert community MiniMax-H3 LoRAs to FastVideo's native adapter layout.

Community H3 adapters are trained against the original (ComfyUI) layout of the transformer, not the diffusers-style
module tree FastVideo loads. They differ in four ways that a rename alone cannot bridge:

``prefixes``
    Trainers wrap the module path: ``diffusion_model.`` (ComfyUI, ai-toolkit, musubi-tuner), ``transformer.`` and
    ``base_model.model.`` (PEFT), ``lora_unet_`` (kohya, which also flattens ``.`` to ``_``).

``fused qkv``
    One ``attn.qkv_proj`` of shape ``[3 * 7168, 5376]`` replaces ``to_q``/``to_k``/``to_v``. A LoRA states its delta as
    ``B @ A`` with ``A [r, in]`` and ``B [3 * 7168, r]``. Splitting ``B`` into three row blocks and pairing each with
    the shared ``A`` reproduces each projection's delta exactly, because the factorization runs over the input
    dimension, which a row split leaves alone. The thirds are q, k, v in that order.

``swapped SwiGLU halves``
    ``mlp.fc1`` and ``ff.fc_in`` are both ``[2 * 14336, 5376]``, but the two halves are stored in opposite order in
    the two layouts. Renaming without swapping applies the gate delta to the up projection and vice versa, which
    type-checks and runs and is quietly wrong.

``module names``
    ``out_proj`` -> ``to_out``, ``mlp.fc1/fc2`` -> ``ff.fc_in/fc_out``, ``blocks`` -> ``transformer_blocks``,
    ``token_refiner.blocks`` -> ``token_refiner.refiner_blocks``, and the boundary projections
    (``audio_patch_proj`` -> ``audio_proj_in`` and so on).

:func:`convert_h3_community_lora` returns tensors keyed ``<module>.lora_A.weight`` / ``<module>.lora_B.weight`` /
``<module>.alpha``, which the regular ``LoRAPipeline`` loader consumes unchanged. Anything it cannot place is
returned in ``unrecognized`` so the caller can fail loudly instead of silently adapting nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

# Dense payload keys (``.diff``, ``.set_weight``, ...) are applied by ``DenseLoRAPatch`` while the checkpoint loads.
# They are not low-rank, so the converter leaves them alone and does not count them as unrecognized.
_DENSE_SUFFIXES = (".diff", ".diff_b", ".diff_param", ".set_weight", ".set_param", ".dora_scale")

_STRIP_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "base_model.model.", "transformer.", "lora_unet_",
                   "lora_transformer_")

# ``<module><sep>(lora_A|lora_down|...)[.<adapter name>][.weight]``; kohya writes ``<module>.alpha``.
_KEY = re.compile(r"^(?P<module>.+?)\.(?P<kind>lora_A|lora_B|lora_down|lora_up|lora_alpha|alpha)"
                  r"(?:\.[A-Za-z0-9_-]+)?(?:\.weight)?$")
_KIND = {
    "lora_A": "A",
    "lora_down": "A",
    "lora_B": "B",
    "lora_up": "B",
    "alpha": "alpha",
    "lora_alpha": "alpha",
}

# Kohya flattens ``.`` to ``_``, so every separator below accepts both.
_S = r"[._]"
_BLOCK = (rf"(?:(?P<refiner>token{_S}refiner{_S}(?:refiner{_S})?blocks)|(?:transformer{_S})?blocks)"
          rf"{_S}(?P<idx>\d+){_S}")
_BLOCK_SUBLAYER = re.compile(rf"^{_BLOCK}(?P<layer>"
                             rf"attn{_S}qkv{_S}proj|attn{_S}out{_S}proj|attn{_S}to{_S}q|attn{_S}to{_S}k|attn{_S}to{_S}v|"
                             rf"attn{_S}to{_S}out(?:{_S}0)?|mlp{_S}fc1|mlp{_S}fc2|ff{_S}net{_S}0{_S}proj|ff{_S}net{_S}2|"
                             rf"ff{_S}fc{_S}in|ff{_S}fc{_S}out)$")

# Boundary projections of the joint audio-video model, original name -> FastVideo module.
_TOP_LEVEL = {
    r"video{S}patch{S}proj": "proj_in",
    r"audio{S}patch{S}proj": "audio_proj_in",
    r"condition{S}proj": "context_embedder",
    r"final{S}layer{S}adaln{S}proj{S}linear": "norm_out.linear",
    r"final{S}layer{S}video{S}out": "proj_out",
    r"final{S}layer{S}audio{S}out": "audio_proj_out",
}
_TOP_LEVEL_RE = [(re.compile("^" + pattern.replace("{S}", _S) + "$"), target) for pattern, target in _TOP_LEVEL.items()]

# Module names already in FastVideo's native layout (a PEFT/diffusers-style adapter).
_NATIVE_MODULE = re.compile(r"^(?:transformer_blocks\.\d+|token_refiner\.refiner_blocks\.\d+)\."
                            r"(?:attn\.(?:to_q|to_k|to_v|to_out)|ff\.(?:fc_in|fc_out)|adaln_proj\.linear)$"
                            r"|^(?:norm_out\.linear|proj_in|proj_out|audio_proj_in|audio_proj_out|context_embedder)$")
_NATIVE_TO_OUT_0 = re.compile(r"\.attn\.to_out\.0$")
_NATIVE_FF_NET = (("ff.net.0.proj", "ff.fc_in"), ("ff.net.2", "ff.fc_out"))


@dataclass(frozen=True)
class H3LoRAConversion:
    """Result of converting one adapter."""

    tensors: dict[str, torch.Tensor]
    source_format: str  # "comfy-fused", "native", "mixed", or "unknown"
    source_modules: int = 0
    target_modules: int = 0
    # Source keys the converter could not place; a non-empty list means the adapter is only partly supported.
    unrecognized: tuple[str, ...] = field(default_factory=tuple)
    # Dense payload keys (``.diff``/``.set_weight``) the converter leaves for ``DenseLoRAPatch``.
    dense_keys: tuple[str, ...] = field(default_factory=tuple)


def _strip_prefix(key: str) -> tuple[str, bool]:
    """Return ``key`` without trainer wrappers and whether a kohya (flattened) prefix was removed."""
    flattened = False
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if key.startswith(prefix):
                flattened = flattened or prefix.startswith("lora_")
                key = key[len(prefix):]
                changed = True
    return key, flattened


def _block_scope(match: re.Match[str]) -> str:
    if match.group("refiner"):
        return f"token_refiner.refiner_blocks.{match.group('idx')}"
    return f"transformer_blocks.{match.group('idx')}"


def _split_rows(count: int) -> Callable[[torch.Tensor], list[torch.Tensor]]:

    def split(b: torch.Tensor) -> list[torch.Tensor]:
        if b.shape[0] % count:
            raise ValueError(f"cannot split lora_B with {b.shape[0]} rows into {count} equal parts")
        step = b.shape[0] // count
        return [b[i * step:(i + 1) * step] for i in range(count)]

    return split


def _swap_halves(b: torch.Tensor) -> list[torch.Tensor]:
    if b.shape[0] % 2:
        raise ValueError(f"cannot swap SwiGLU halves of lora_B with {b.shape[0]} rows")
    half = b.shape[0] // 2
    return [torch.cat([b[half:], b[:half]], dim=0)]


def _identity(b: torch.Tensor) -> list[torch.Tensor]:
    return [b]


# layer name in the original layout -> (FastVideo sublayers, lora_B row transform)
_LAYER_PLAN: dict[str, tuple[tuple[str, ...], Callable[[torch.Tensor], list[torch.Tensor]]]] = {
    "attn.qkv_proj": (("attn.to_q", "attn.to_k", "attn.to_v"), _split_rows(3)),
    "attn.out_proj": (("attn.to_out", ), _identity),
    "mlp.fc1": (("ff.fc_in", ), _swap_halves),
    "mlp.fc2": (("ff.fc_out", ), _identity),
    "attn.to_q": (("attn.to_q", ), _identity),
    "attn.to_k": (("attn.to_k", ), _identity),
    "attn.to_v": (("attn.to_v", ), _identity),
    "attn.to_out": (("attn.to_out", ), _identity),
    "attn.to_out.0": (("attn.to_out", ), _identity),
    "ff.net.0.proj": (("ff.fc_in", ), _identity),
    "ff.net.2": (("ff.fc_out", ), _identity),
    "ff.fc_in": (("ff.fc_in", ), _identity),
    "ff.fc_out": (("ff.fc_out", ), _identity),
}
# Separator-free spelling -> canonical layer name, so ``attn.to_q`` and ``attn_to_q`` resolve alike.
_LAYER_BY_COLLAPSED_NAME = {re.sub(_S, "", name): name for name in _LAYER_PLAN}
_FUSED_LAYERS = frozenset({"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"})


def _plan_module(module: str, flattened: bool) -> tuple[tuple[str, ...], Callable[[torch.Tensor], list[torch.Tensor]],
                                                        str] | None:
    """Map a source module to (FastVideo module names, B transform, layout tag), or ``None`` if unknown."""
    block = _BLOCK_SUBLAYER.match(module)
    if block is not None:
        layer = _LAYER_BY_COLLAPSED_NAME[re.sub(_S, "", block.group("layer"))]
        sublayers, transform = _LAYER_PLAN[layer]
        scope = _block_scope(block)
        layout = "comfy-fused" if layer in _FUSED_LAYERS else "native"
        return tuple(f"{scope}.{name}" for name in sublayers), transform, layout
    for pattern, target in _TOP_LEVEL_RE:
        if pattern.match(module):
            return (target, ), _identity, "comfy-fused"
    if flattened:
        return None
    native = _NATIVE_TO_OUT_0.sub(".attn.to_out", module)
    for old, new in _NATIVE_FF_NET:
        native = native.replace(old, new)
    if _NATIVE_MODULE.match(native):
        return (native, ), _identity, "native"
    return None


def convert_h3_community_lora(tensors: Mapping[str, torch.Tensor]) -> H3LoRAConversion:
    """Convert a community H3 adapter state dict to FastVideo-native, per-projection keys.

    Native-layout adapters pass through with only their names normalized. Keys that match no known layout are
    returned in ``unrecognized`` and are not emitted; the caller decides whether that is fatal.
    """
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    source_keys: dict[str, list[str]] = {}
    flattened_modules: set[str] = set()
    unrecognized: list[str] = []
    dense_keys: list[str] = []
    for key, tensor in tensors.items():
        if key.endswith(_DENSE_SUFFIXES):
            dense_keys.append(key)
            continue
        stripped, flattened = _strip_prefix(key)
        match = _KEY.match(stripped)
        if match is None:
            unrecognized.append(key)
            continue
        module = match.group("module")
        if flattened:
            flattened_modules.add(module)
        slot = grouped.setdefault(module, {})
        kind = _KIND[match.group("kind")]
        if kind in slot:
            unrecognized.append(key)
            continue
        slot[kind] = tensor
        source_keys.setdefault(module, []).append(key)

    out: dict[str, torch.Tensor] = {}
    layouts: set[str] = set()
    for module, parts in sorted(grouped.items()):
        plan = _plan_module(module, module in flattened_modules)
        if plan is None or "A" not in parts or "B" not in parts:
            unrecognized.extend(source_keys[module])
            continue
        targets, transform, layout = plan
        a, b = parts["A"], parts["B"]
        if a.dim() != 2 or b.dim() != 2 or b.shape[1] != a.shape[0]:
            raise ValueError(f"{module}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} do not share a rank")
        layouts.add(layout)
        for target, rows in zip(targets, transform(b), strict=True):
            out[f"{target}.lora_A.weight"] = a.contiguous()
            out[f"{target}.lora_B.weight"] = rows.contiguous()
            if "alpha" in parts:
                out[f"{target}.alpha"] = parts["alpha"]
    if not layouts:
        source_format = "unknown"
    elif len(layouts) == 1:
        source_format = next(iter(layouts))
    else:
        source_format = "mixed"
    return H3LoRAConversion(
        tensors=out,
        source_format=source_format,
        source_modules=len(grouped),
        target_modules=sum(1 for key in out if key.endswith(".lora_A.weight")),
        unrecognized=tuple(sorted(unrecognized)),
        dense_keys=tuple(sorted(dense_keys)),
    )
