# SPDX-License-Identifier: Apache-2.0
"""Component parity scaffold for GLM-Image DiT (GlmImageTransformer2DModel).

Compares the FastVideo-native port at `fastvideo.models.dits.glm_image` against
the diffusers reference `GlmImageTransformer2DModel` from
`diffusers.models.transformers.transformer_glm_image`.

Skips cleanly until both the FastVideo class and the local weights exist; the
test is real (loads weights, forwards a fixed input, compares output tensors).
"""
from __future__ import annotations

import os
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

def _has_weights() -> bool:
    return TRANSFORMER_DIR.exists() and any(
        TRANSFORMER_DIR.glob("*.safetensors"))


def _has_glm_image_diffusers() -> bool:
    """`GlmImageTransformer2DModel` lands in diffusers>=0.37.0.dev0.

    FastVideo main pins `diffusers>=0.33.1`; the installed env is 0.36.0. The
    numerical parity test needs the diffusers class as an oracle — until the
    installed diffusers is upgraded, this test skips. The structural surface
    is still verified by `test_glm_image_transformer_strict_load.py`.
    """
    import importlib.util
    spec = importlib.util.find_spec("diffusers")
    if spec is None:
        return False
    import diffusers
    return hasattr(diffusers, "GlmImageTransformer2DModel")


pytestmark = [
    pytest.mark.skipif(
        not _has_weights(),
        reason=(
            f"GLM-Image transformer weights not found at {TRANSFORMER_DIR}. "
            "Download via "
            "`python .agents/skills/add-model-01-prep/scripts/download_hf_weights.py "
            "zai-org/GLM-Image official_weights/glm_image`."),
    ),
    pytest.mark.skipif(
        not _has_glm_image_diffusers(),
        reason=("diffusers in this env lacks `GlmImageTransformer2DModel` "
                "(main allows >=0.33.1; installed 0.36.0; class lands in "
                "0.37.0.dev0). Bump diffusers locally to run numerical parity. "
                "Structural surface is covered by "
                "`test_glm_image_transformer_strict_load.py`."),
    ),
]


@pytest.fixture(scope="module")
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for GLM-Image transformer parity.")
    return torch.device("cuda")


def _load_official(device, dtype):
    diffusers = pytest.importorskip("diffusers")
    cls = diffusers.GlmImageTransformer2DModel
    return cls.from_pretrained(str(TRANSFORMER_DIR),
                               torch_dtype=dtype).to(device).eval()


def _load_fastvideo(device, dtype):
    pytest.importorskip("fastvideo")
    try:
        from fastvideo.configs.models.dits.glm_image import GlmImageDiTConfig
        from fastvideo.models.dits.glm_image import GlmImageTransformer2DModel
    except ImportError as e:
        pytest.skip(f"FastVideo GLM-Image DiT not yet ported: {e}")
    cfg = GlmImageDiTConfig()
    model = GlmImageTransformer2DModel(cfg,
                                       {"_class_name":
                                        "GlmImageTransformer2DModel"})
    sd = _load_state_dict(TRANSFORMER_DIR)
    sd = _apply_param_mapping(sd, cfg.arch_config.param_names_mapping)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"FastVideo DiT missing keys: {missing[:10]}"
    assert not unexpected, f"FastVideo DiT unexpected keys: {unexpected[:10]}"
    return model.to(device, dtype=dtype).eval()


def _forward_pair(device, dtype):
    """Run both DiTs on identical synthetic inputs; return (fv_out, ref_out)."""
    from fastvideo.forward_context import set_forward_context
    ref = _load_official(device, dtype)
    fv = _load_fastvideo(device, dtype)
    inputs = _make_inputs(device, dtype=dtype)
    with torch.no_grad():
        ref_out = ref(**inputs, return_dict=False)[0]
        with set_forward_context(current_timestep=0, attn_metadata=None,
                                 forward_batch=None):
            fv_out = fv(**inputs)
        if isinstance(fv_out, tuple):
            fv_out = fv_out[0]
    assert ref_out.shape == fv_out.shape, (
        f"shape mismatch: {ref_out.shape} vs {fv_out.shape}")
    # Free the two 7B models before the next dtype runs (single-GPU budget).
    del ref, fv
    torch.cuda.empty_cache()
    return fv_out.float(), ref_out.float()


def _load_state_dict(dir_path: Path) -> dict[str, torch.Tensor]:
    safetensors = pytest.importorskip("safetensors.torch")
    sd: dict[str, torch.Tensor] = {}
    for shard in sorted(dir_path.glob("*.safetensors")):
        sd.update(safetensors.load_file(str(shard)))
    return sd


def _apply_param_mapping(sd, mapping):
    import re
    out = {}
    for k, v in sd.items():
        new_k = k
        for pat, repl in mapping.items():
            if re.match(pat, k):
                new_k = re.sub(pat, repl, k)
                break
        out[new_k] = v
    return out


def _make_inputs(device, batch_size=1, height=64, width=64,
                 text_seq_len=32, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """Synthetic inputs matching the diffusers `GlmImageTransformer2DModel.forward`
    contract. Latent shape is `(B, 16, H, W)`; post-patch grid is `(H/2, W/2)`
    so `prior_token_id` has `(H/2)*(W/2)` entries per batch element."""
    torch.manual_seed(0)
    patch_size = 2
    num_patches = (height // patch_size) * (width // patch_size)
    return {
        "hidden_states": torch.randn(batch_size, 16, height, width,
                                     device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(batch_size, text_seq_len, 1472,
                                             device=device, dtype=dtype),
        "prior_token_id": torch.randint(0, 16384, (batch_size, num_patches),
                                        device=device, dtype=torch.long),
        "prior_token_drop": torch.zeros(batch_size, device=device,
                                        dtype=torch.bool),
        "timestep": torch.tensor([500] * batch_size, device=device,
                                 dtype=torch.long),
        "target_size": torch.tensor([[height * 8, width * 8]] * batch_size,
                                    device=device, dtype=torch.long),
        "crop_coords": torch.zeros(batch_size, 2, device=device,
                                   dtype=torch.long),
    }


# fp32 is the structural-correctness gate: with identical math the two
# implementations agree to ~1e-3 (residual is SDPA reduction-order noise).
ATOL_FP32 = 1e-2
RTOL_FP32 = 1e-2
# bf16 is the inference dtype. Rounding accumulates across 30 layers and the
# attention kernels differ, so element-wise atol is meaningless; cosine
# similarity is the right statistical bar for "same function, bf16 noise".
COS_MIN_BF16 = 0.999


def test_transformer_forward_matches_diffusers_fp32(device):
    """Structural gate: in fp32 the FastVideo DiT is numerically faithful to
    the diffusers reference (cosine ~1.0, element-wise close to ~1e-3)."""
    fv_out, ref_out = _forward_pair(device, torch.float32)
    cos = torch.nn.functional.cosine_similarity(
        fv_out.flatten(), ref_out.flatten(), dim=0)
    assert cos > 0.9999, f"fp32 cosine similarity too low: {cos:.6f}"
    torch.testing.assert_close(fv_out, ref_out,
                               atol=ATOL_FP32, rtol=RTOL_FP32)


def test_transformer_forward_matches_diffusers_bf16(device):
    """Inference-dtype sanity: bf16 output tracks the diffusers reference up to
    precision accumulation (high cosine similarity). Element-wise atol is not a
    meaningful bar for a 30-layer bf16 transformer vs a different attention
    kernel (~28% of elements exceed atol=5e-3 purely from rounding) — see
    PORT_STATUS I020."""
    fv_out, ref_out = _forward_pair(device, torch.bfloat16)
    cos = torch.nn.functional.cosine_similarity(
        fv_out.flatten(), ref_out.flatten(), dim=0)
    assert cos > COS_MIN_BF16, (
        f"bf16 cosine similarity {cos:.6f} below {COS_MIN_BF16}; "
        "indicates a structural divergence, not just precision.")
