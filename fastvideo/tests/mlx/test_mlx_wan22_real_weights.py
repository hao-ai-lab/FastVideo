# SPDX-License-Identifier: Apache-2.0
"""Track D Rung 3: real-weight T2V parity for Wan2.2-TI2V-5B MLX vs torch.

Loads the released ``FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`` transformer
into both ``MLXWan22DiT`` and torch ``WanTransformer3DModel``, runs one forward
with a 2-D per-token timestep (frame 0 at t=0 I2V-style, remaining tokens at
t≈900), and asserts allclose. Gated on Metal + local weights so Linux
``mlx[cpu]`` CI stays green.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="MLX is required for the Wan2.2 real-weight parity test")

_ROOT = Path(os.environ.get("FASTVIDEO_WAN22_5B_ROOT", str(Path.home() / "models" / "fastwan22_5b")))
_CHECKPOINT = _ROOT / "transformer" / "diffusion_pytorch_model.safetensors"
_CONFIG = _ROOT / "transformer" / "config.json"

_HAS_METAL = bool(getattr(mx, "metal", None) and mx.metal.is_available())
_HAS_WEIGHTS = _CHECKPOINT.exists() and _CONFIG.exists() and _CHECKPOINT.stat().st_size > 1_000_000_000

pytestmark = [
    pytest.mark.skipif(not _HAS_METAL, reason="Metal required for real 5B fp16 parity"),
    pytest.mark.skipif(
        not _HAS_WEIGHTS,
        reason=f"Wan2.2-5B checkpoint not found/incomplete under {_ROOT} "
        "(set FASTVIDEO_WAN22_5B_ROOT; expected ~10 GB safetensors)",
    ),
]


def _load_torch_wan22_from_diffusers(checkpoint: Path, config_path: Path, *, dtype):
    """Instantiate ``WanTransformer3DModel`` and load Diffusers-format weights."""
    import torch
    from safetensors.torch import load_file

    from fastvideo.configs.models.dits.wanvideo import WanVideoArchConfig, WanVideoConfig
    from fastvideo.models.dits.wanvideo import WanTransformer3DModel
    from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

    hf_config = json.loads(config_path.read_text())
    # Diffusers-only keys that are not arch fields.
    arch_kwargs = {
        k: v
        for k, v in hf_config.items()
        if k not in {"_class_name", "_name_or_path", "_diffusers_version"}
    }
    cfg = WanVideoConfig(arch_config=WanVideoArchConfig(**{
        "num_attention_heads": int(arch_kwargs["num_attention_heads"]),
        "attention_head_dim": int(arch_kwargs["attention_head_dim"]),
        "in_channels": int(arch_kwargs["in_channels"]),
        "out_channels": int(arch_kwargs["out_channels"]),
        "text_dim": int(arch_kwargs["text_dim"]),
        "freq_dim": int(arch_kwargs["freq_dim"]),
        "ffn_dim": int(arch_kwargs["ffn_dim"]),
        "num_layers": int(arch_kwargs["num_layers"]),
        "patch_size": tuple(arch_kwargs["patch_size"]),
        "cross_attn_norm": bool(arch_kwargs.get("cross_attn_norm", True)),
        "qk_norm": arch_kwargs.get("qk_norm", "rms_norm_across_heads"),
        "eps": float(arch_kwargs.get("eps", 1e-6)),
        "rope_max_seq_len": int(arch_kwargs.get("rope_max_seq_len", 1024)),
        "added_kv_proj_dim": arch_kwargs.get("added_kv_proj_dim"),
        "image_dim": arch_kwargs.get("image_dim"),
        "pos_embed_seq_len": arch_kwargs.get("pos_embed_seq_len"),
    }))
    model = WanTransformer3DModel(config=cfg, hf_config=hf_config).eval()
    raw = load_file(str(checkpoint), device="cpu")
    mapping = get_param_names_mapping(model.param_names_mapping)
    custom_sd, _ = hf_to_custom_state_dict(raw.items(), mapping)
    missing, unexpected = model.load_state_dict(custom_sd, strict=False)
    # Diffusers layout should map cleanly; allow empty buffers only.
    assert not unexpected, f"unexpected keys: {unexpected[:8]}"
    # Missing keys are rare (e.g. unused image embedder); warn via print if any.
    if missing:
        print(f"[wan22 real parity] missing torch keys ({len(missing)}): {missing[:6]}...")
    return model.to(dtype=dtype)


@pytest.mark.usefixtures("distributed_setup")
def test_wan22_real_weights_mlx_matches_torch_per_token_timestep() -> None:
    import torch

    from fastvideo.forward_context import set_forward_context
    from fastvideo.mlx_runtime.wan22 import mlx_wan22_dit_from_diffusers_safetensors
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from examples.inference.basic.mlx_wan_prompt_to_video import make_rotary_embeddings

    config = json.loads(_CONFIG.read_text())
    in_ch = int(config["in_channels"])
    text_dim = int(config["text_dim"])
    # Small latent so activations fit alongside the ~10 GB fp16 weights on 36 GB.
    frames, height, width = 2, 16, 16
    p_t, p_h, p_w = tuple(config["patch_size"])
    tokens_per_frame = (height // p_h) * (width // p_w)
    num_tokens = (frames // p_t) * tokens_per_frame

    # Frame 0 clean (t=0), remaining tokens noised (t=900) — I2V-style expand_timesteps.
    per_frame_levels = [0] + [900] * (frames // p_t - 1)
    timestep_1d = [per_frame_levels[i // tokens_per_frame] for i in range(num_tokens)]
    timestep = torch.tensor([timestep_1d], dtype=torch.long)

    rng = np.random.default_rng(2026)
    hidden_np = (rng.standard_normal((1, in_ch, frames, height, width)) * 0.5).astype(np.float32)
    text_np = (rng.standard_normal((1, 32, text_dim)) * 0.1).astype(np.float32)

    # --- torch reference (CPU fp16, same weight dtype as MLX deploy path) ---
    torch_model = _load_torch_wan22_from_diffusers(_CHECKPOINT, _CONFIG, dtype=torch.float16)
    hidden_t = torch.from_numpy(hidden_np).to(torch.float16)
    text_t = torch.from_numpy(text_np).to(torch.float16)
    with torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=ForwardBatch(data_type="dummy")):
        ref = torch_model(hidden_states=hidden_t, encoder_hidden_states=text_t, timestep=timestep)
        ref_np = ref.detach().float().cpu().numpy()

    # Free torch weights before loading MLX so ~10 GB fp16 + activations fit in 36 GB.
    del torch_model
    import gc
    gc.collect()

    # --- MLX (fp16 weights, fp16 compute) ---
    mlx_model = mlx_wan22_dit_from_diffusers_safetensors(_CHECKPOINT, _CONFIG, dtype="fp16")
    freqs_cis = make_rotary_embeddings(
        config, latent_frames=frames, latent_height=height, latent_width=width)
    out = mlx_model(
        mx.array(hidden_np).astype(mx.float16),
        mx.array(text_np).astype(mx.float16),
        mx.array(np.array(timestep_1d, dtype=np.float32)[None, :]),
        freqs_cis,
    )
    mx.eval(out)
    mlx_np = np.array(out.astype(mx.float32))

    assert mlx_np.shape == ref_np.shape
    assert np.isfinite(mlx_np).all()
    max_abs = float(np.abs(mlx_np - ref_np).max())
    mean_abs = float(np.abs(mlx_np - ref_np).mean())
    flat_a, flat_b = mlx_np.ravel(), ref_np.ravel()
    cosine = float(np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12))
    print(f"Wan2.2-5B real-weight MLX-fp16 vs torch-fp16: "
          f"max|Δ|={max_abs:.3e} mean|Δ|={mean_abs:.3e} cosine={cosine:.6f}")
    # Full 30-layer fp16 Metal vs torch SDPA drifts more than the tiny-config
    # fp32 gate (2e-3) / CUDA-tiny dump (5e-3). Measured on M4 Max ~ max 9e-2,
    # mean 6e-3, cosine 0.99994 — assert with recorded headroom.
    assert max_abs <= 0.15, f"max|Δ|={max_abs} over budget"
    assert mean_abs <= 0.02, f"mean|Δ|={mean_abs} over budget"
    assert cosine >= 0.999, f"cosine={cosine} too low"
