# SPDX-License-Identifier: Apache-2.0
"""PromptRL bundle exporter.

Produces a self-contained inference bundle::

    <bundle>/
      manifest.json                         # schema, base models, versions
      refiner/                              # PEFT adapter + tokenizer
        adapter_model.safetensors
        adapter_config.json
        tokenizer... (saved via save_pretrained)
      generator/                            # FastVideo Wan LoRA
        promptrl_generator_lora.safetensors # HF-style Wan layer names
        lora_config.json                    # rank/alpha/targets + layout

``manifest.json`` records the prompt template/version, refiner sampling
configuration, base model identifiers, and the compatible FastVideo
version so :class:`~fastvideo.train.methods.rl.promptrl.inference.PromptRefiner`
can reconstruct the exact refiner behavior.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from fastvideo.logger import init_logger

logger = init_logger(__name__)

BUNDLE_SCHEMA_VERSION = 1
GENERATOR_LORA_FILENAME = "promptrl_generator_lora.safetensors"
GENERATOR_KEY_LAYOUT = "fastvideo_hf_v1"

#: FastVideo-training Wan module names -> HF/diffusers Wan layer names.
#: Inverse of the Wan arch ``param_names_mapping`` for LoRA-wrapped paths.
_WAN_CUSTOM_TO_HF = {
    r"^blocks\.(\d+)\.self_attn\.to_q\b(.*)$": r"blocks.\1.attn1.to_q\2",
    r"^blocks\.(\d+)\.self_attn\.to_k\b(.*)$": r"blocks.\1.attn1.to_k\2",
    r"^blocks\.(\d+)\.self_attn\.to_v\b(.*)$": r"blocks.\1.attn1.to_v\2",
    r"^blocks\.(\d+)\.self_attn\.to_out\b(.*)$": r"blocks.\1.attn1.to_out.0\2",
    r"^blocks\.(\d+)\.cross_attn\.to_q\b(.*)$": r"blocks.\1.attn2.to_q\2",
    r"^blocks\.(\d+)\.cross_attn\.to_k\b(.*)$": r"blocks.\1.attn2.to_k\2",
    r"^blocks\.(\d+)\.cross_attn\.to_v\b(.*)$": r"blocks.\1.attn2.to_v\2",
    r"^blocks\.(\d+)\.cross_attn\.to_out\b(.*)$": r"blocks.\1.attn2.to_out.0\2",
    r"^blocks\.(\d+)\.ffn\.fc_in\b(.*)$": r"blocks.\1.ffn.net.0.proj\2",
    r"^blocks\.(\d+)\.ffn\.fc_out\b(.*)$": r"blocks.\1.ffn.net.2\2",
}


@dataclass(slots=True)
class BundleManifest:
    """Contents of ``manifest.json``."""

    base_refiner_model: str
    base_generator_model: str
    fastvideo_version: str
    refiner_lora: dict[str, Any]
    generator_lora: dict[str, Any]
    prompt_template_version: str
    refiner_sampling: dict[str, Any]
    mode: str
    schema_version: int = BUNDLE_SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BundleManifest:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown manifest keys: {unknown}")
        return cls(**raw)


def training_name_to_hf(name: str) -> str:
    """Map a training-layout Wan LoRA key to the HF/diffusers layout."""
    for pattern, replacement in _WAN_CUSTOM_TO_HF.items():
        if re.match(pattern, name):
            return re.sub(pattern, replacement, name)
    logger.warning("No HF mapping for LoRA key %r; exporting as-is", name)
    return name


def extract_generator_lora(transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract LoRA tensors from a training transformer.

    Returns HF-named ``{layer}.lora_A`` / ``{layer}.lora_B`` weights plus
    scalar ``{layer}.lora_alpha`` entries.
    """
    from fastvideo.layers.lora.linear import BaseLayerWithLoRA

    state: dict[str, torch.Tensor] = {}
    for module_name, module in transformer.named_modules():
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        if module.lora_A is None or module.lora_B is None:
            continue
        base_name = training_name_to_hf(module_name)
        lora_a = module.lora_A
        lora_b = module.lora_B
        if hasattr(lora_a, "to_local"):  # DTensor-safe extraction
            lora_a = lora_a.to_local()
            lora_b = lora_b.to_local()
        state[f"{base_name}.lora_A"] = lora_a.detach().float().cpu().contiguous()
        state[f"{base_name}.lora_B"] = lora_b.detach().float().cpu().contiguous()
        alpha = float(getattr(module, "lora_alpha", 0) or 0)
        state[f"{base_name}.lora_alpha"] = torch.tensor(alpha, dtype=torch.float32)
    if not state:
        raise ValueError("Transformer contains no LoRA layers to export")
    return state


def export_promptrl_bundle(
    output_dir: str,
    *,
    manifest: BundleManifest,
    refiner_role: Any = None,
    refiner_adapter_dir: str | None = None,
    refiner_tokenizer: Any = None,
    generator_transformer: torch.nn.Module | None = None,
) -> dict[str, str]:
    """Write a PromptRL inference bundle to *output_dir*.

    Exactly one of ``refiner_role`` / ``refiner_adapter_dir`` provides
    the refiner adapter; ``generator_transformer`` provides the Wan
    LoRA.  Returns the written artifact paths.
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    written: dict[str, str] = {}

    # --- manifest ---
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)
    written["manifest"] = manifest_path

    # --- refiner adapter ---
    refiner_dir = os.path.join(output_dir, "refiner")
    if refiner_role is not None:
        refiner_role.save_adapter(refiner_dir)
    elif refiner_adapter_dir is not None:
        if os.path.isdir(refiner_dir):
            shutil.rmtree(refiner_dir)
        shutil.copytree(refiner_adapter_dir, refiner_dir)
    else:
        raise ValueError("export_promptrl_bundle requires refiner_role or "
                         "refiner_adapter_dir")
    if refiner_tokenizer is not None:
        refiner_tokenizer.save_pretrained(refiner_dir)
    written["refiner"] = refiner_dir

    # --- generator LoRA ---
    generator_dir = os.path.join(output_dir, "generator")
    os.makedirs(generator_dir, exist_ok=True)
    if generator_transformer is not None:
        state = extract_generator_lora(generator_transformer)
        lora_path = os.path.join(generator_dir, GENERATOR_LORA_FILENAME)
        save_file(state, lora_path)
        written["generator"] = lora_path
        config_path = os.path.join(generator_dir, "lora_config.json")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "key_layout": GENERATOR_KEY_LAYOUT,
                    "filename": GENERATOR_LORA_FILENAME,
                    **manifest.generator_lora,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        written["generator_config"] = config_path
    else:
        logger.info("No generator transformer given; bundle ships refiner-only")
    return written


def load_bundle_manifest(bundle_dir: str) -> BundleManifest:
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No manifest.json in bundle {bundle_dir}")
    with open(manifest_path, encoding="utf-8") as handle:
        return BundleManifest.from_dict(json.load(handle))
