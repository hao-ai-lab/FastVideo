# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
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


def _has_weights() -> bool:
    required = ["transformer", "vae", "text_encoder",
                "vision_language_encoder", "processor", "tokenizer",
                "scheduler"]
    return all((LOCAL_WEIGHTS_DIR / r).exists() for r in required)


def _upstream_glm_image_available() -> bool:
    try:
        import transformers
        import diffusers
    except ImportError:
        return False
    return (hasattr(transformers, "GlmImageForConditionalGeneration")
            and hasattr(diffusers, "GlmImagePipeline"))


pytestmark = [
    pytest.mark.skipif(
        not _has_weights(),
        reason=f"GLM-Image full weights not found at {LOCAL_WEIGHTS_DIR}.",
    ),
    pytest.mark.skipif(
        not _upstream_glm_image_available(),
        reason=("Pipeline parity needs transformers>=5.0.0rc0 and "
                "diffusers>=0.37.0.dev0; main pins predate both. Bump locally "
                "to run."),
    ),
]


SAMPLE_PROMPT = (
    "A landscape photo with rolling green hills under a clear blue sky.")
SEED = 0
HEIGHT = 512
WIDTH = 512
STEPS = 8


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image pipeline parity.")
    return torch.device("cuda")


def _to_uint8_hwc(a) -> np.ndarray:
    if torch.is_tensor(a):
        a = a.float().cpu().numpy()
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        scale = 255.0 if float(a.max()) <= 1.5 else 1.0
        a = np.clip(a * scale, 0, 255).astype(np.uint8)
    return a


def _diffusers_image(device) -> np.ndarray:
    diffusers = pytest.importorskip("diffusers")
    pipe = diffusers.GlmImagePipeline.from_pretrained(
        str(LOCAL_WEIGHTS_DIR), torch_dtype=torch.bfloat16).to(device)
    generator = torch.Generator(device=device).manual_seed(SEED)
    out = pipe(prompt=SAMPLE_PROMPT, height=HEIGHT, width=WIDTH,
               num_inference_steps=STEPS, guidance_scale=1.5,
               generator=generator, output_type="np")
    del pipe
    torch.cuda.empty_cache()
    return _to_uint8_hwc(out.images[0])


def _fastvideo_image(device) -> np.ndarray:
    pytest.importorskip("fastvideo")
    try:
        from fastvideo import VideoGenerator
    except ImportError as e:
        pytest.skip(f"FastVideo VideoGenerator unavailable: {e}")
    gen = VideoGenerator.from_pretrained(str(LOCAL_WEIGHTS_DIR), num_gpus=1,
                                         trust_remote_code=True)
    result = gen.generate_video(prompt=SAMPLE_PROMPT,
                                save_video=False,
                                return_frames=True,
                                height=HEIGHT, width=WIDTH,
                                num_inference_steps=STEPS,
                                guidance_scale=1.5,
                                seed=SEED)
    gen.shutdown()
    return _to_uint8_hwc(result["frames"][0])


def test_fastvideo_pipeline_produces_valid_image(device):
    fv = _fastvideo_image(device)
    assert fv.shape == (HEIGHT, WIDTH, 3), f"unexpected shape {fv.shape}"
    assert fv.dtype == np.uint8 and np.isfinite(fv).all()
    assert fv.std() > 10.0, f"image is near-constant (std={fv.std():.2f})"


def test_pipeline_matches_diffusers_distribution(device):
    """Pixel-exact parity is NOT asserted: AR sampling (do_sample=True) and
    independent RNG streams make the two pipelines distinct valid samples, so we
    only gate same-shape, healthy images in a comparable brightness regime."""
    fv = _fastvideo_image(device)
    official = _diffusers_image(device)
    assert official.shape == fv.shape, (
        f"shape mismatch: {official.shape} vs {fv.shape}")
    assert official.std() > 10.0, "diffusers reference image is near-constant"
    mae = np.abs(official.astype(np.float32) - fv.astype(np.float32)).mean()
    bright_gap = abs(float(official.mean()) - float(fv.mean()))
    print(f"[glm-image pipeline] MAE={mae:.2f}/255 "
          f"brightness diffusers={official.mean():.1f} fv={fv.mean():.1f} "
          f"gap={bright_gap:.1f}")
    assert bright_gap < 60.0, (
        f"global brightness regimes diverge ({bright_gap:.1f}/255); "
        "suggests a wiring/normalization bug, not just RNG sampling")
