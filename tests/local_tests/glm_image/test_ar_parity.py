# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")
os.environ.setdefault("DISABLE_SP", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
FAMILY = "glm_image"
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("GLM_IMAGE_LOCAL_WEIGHTS_DIR",
              REPO_ROOT / "official_weights" / FAMILY))
AR_DIR = LOCAL_WEIGHTS_DIR / "vision_language_encoder"
PROCESSOR_DIR = LOCAL_WEIGHTS_DIR / "processor"


def _has_weights() -> bool:
    return AR_DIR.exists() and any(AR_DIR.glob("*.safetensors"))


def _has_glm_image_transformers() -> bool:
    import importlib.util
    spec = importlib.util.find_spec("transformers")
    if spec is None:
        return False
    import transformers
    return hasattr(transformers, "GlmImageForConditionalGeneration")


pytestmark = [
    pytest.mark.skipif(
        not _has_weights(),
        reason=f"GLM-Image AR encoder weights not found at {AR_DIR}.",
    ),
    pytest.mark.skipif(
        not _has_glm_image_transformers(),
        reason=("transformers in this env lacks "
                "`GlmImageForConditionalGeneration` (main pins 4.57.3; needs "
                "5.0.0rc0+). Bump transformers locally to exercise the AR "
                "lazy-wrapper."),
    ),
]


SAMPLE_PROMPT = (
    "A beautiful landscape photography with rolling hills, a winding river, "
    "and a vibrant sunset in the background. Photorealistic style."
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image AR encoder parity.")
    return torch.device("cuda")


@pytest.fixture(scope="module")
def fastvideo_ar(device):
    pytest.importorskip("fastvideo")
    try:
        from fastvideo.models.encoders.glm_image_ar_loader import (
            GlmImageARLoader)
    except ImportError as e:
        pytest.skip(f"FastVideo lazy-wrapper not yet at target path: {e}")
    loader = GlmImageARLoader(str(AR_DIR), str(PROCESSOR_DIR),
                              torch_dtype=torch.bfloat16)
    loader.to(device).eval()
    return loader


def test_ar_wrapper_exposes_expected_surface(fastvideo_ar):
    assert callable(getattr(fastvideo_ar, "generate", None))
    assert getattr(fastvideo_ar, "processor", None) is not None
    assert getattr(fastvideo_ar, "config", None) is not None
    assert getattr(fastvideo_ar, "generation_config", None) is not None
    from transformers import GlmImageForConditionalGeneration
    assert isinstance(fastvideo_ar._model, GlmImageForConditionalGeneration)


def test_ar_to_and_eval_propagate(fastvideo_ar, device):
    assert next(fastvideo_ar._model.parameters()).device.type == device.type
    assert not fastvideo_ar._model.training


def test_ar_generate_is_deterministic_under_do_sample_false(fastvideo_ar,
                                                            device):
    processor = fastvideo_ar.processor
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": SAMPLE_PROMPT}],
    }]
    inputs = processor.apply_chat_template(messages, tokenize=True,
                                           target_h=1024, target_w=1024,
                                           return_dict=True,
                                           return_tensors="pt").to(device)
    with torch.no_grad():
        out_a = fastvideo_ar.generate(**inputs, max_new_tokens=32,
                                      do_sample=False)
        out_b = fastvideo_ar.generate(**inputs, max_new_tokens=32,
                                      do_sample=False)
    assert out_a.shape == out_b.shape
    assert torch.equal(out_a, out_b)
