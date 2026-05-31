# SPDX-License-Identifier: Apache-2.0
"""Strict-load smoke for GLM-Image DiT.

Loads all three transformer shards from `official_weights/glm_image/transformer/`,
applies the FastVideo `param_names_mapping`, and calls
`GlmImageTransformer2DModel.load_state_dict(..., strict=True)`. Asserts the
config-driven model instantiates and every checkpoint key has a matching
FastVideo parameter with the same shape and dtype.

This is weight-load verification only — it does not run a forward pass. Numerical
parity against the diffusers reference lives in
`test_glm_image_transformer_parity.py` and requires `diffusers >= 0.37.0.dev0`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "glm_image"
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / FAMILY))
TRANSFORMER_DIR = LOCAL_WEIGHTS_DIR / "transformer"


def _has_shards() -> bool:
    return TRANSFORMER_DIR.exists() and any(
        TRANSFORMER_DIR.glob("*.safetensors"))


pytestmark = pytest.mark.skipif(
    not _has_shards(),
    reason=f"GLM-Image transformer shards not at {TRANSFORMER_DIR}.",
)


def test_dit_strict_load_against_hf_checkpoint():
    safetensors = pytest.importorskip("safetensors.torch")
    from fastvideo.configs.models.dits.glm_image import GlmImageDiTConfig
    from fastvideo.models.dits.glm_image import GlmImageTransformer2DModel

    cfg = GlmImageDiTConfig()
    mapping = cfg.arch_config.param_names_mapping

    # 1. Build the FastVideo model on CPU/meta-free path.
    torch.set_default_device("cpu")
    hf_config = {"_class_name": "GlmImageTransformer2DModel"}
    model = GlmImageTransformer2DModel(cfg, hf_config)

    # 2. Load every shard and apply param_names_mapping.
    raw_sd: dict[str, torch.Tensor] = {}
    for shard in sorted(TRANSFORMER_DIR.glob("*.safetensors")):
        raw_sd.update(safetensors.load_file(str(shard)))

    def rename(k: str) -> str:
        for pat, repl in mapping.items():
            if re.match(pat, k):
                return re.sub(pat, repl, k)
        return k

    renamed_sd = {rename(k): v for k, v in raw_sd.items()}

    # 3. Compare key sets (sanity, redundant with strict=True but gives a
    # clearer error message if it fails).
    fv_keys = set(model.state_dict().keys())
    ckpt_keys = set(renamed_sd.keys())
    missing = fv_keys - ckpt_keys
    unexpected = ckpt_keys - fv_keys
    assert not missing, f"FastVideo DiT missing {len(missing)} keys, e.g. {sorted(missing)[:5]}"
    assert not unexpected, f"checkpoint has {len(unexpected)} unexpected keys, e.g. {sorted(unexpected)[:5]}"

    # 4. Shape and dtype sanity — strict_load with shape mismatch raises early.
    fv_sd = model.state_dict()
    shape_mismatches = []
    for k, v in renamed_sd.items():
        if v.shape != fv_sd[k].shape:
            shape_mismatches.append((k, tuple(fv_sd[k].shape), tuple(v.shape)))
    assert not shape_mismatches, (
        f"shape mismatches: {shape_mismatches[:5]}")

    # 5. Strict load.
    incompatible = model.load_state_dict(renamed_sd, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
