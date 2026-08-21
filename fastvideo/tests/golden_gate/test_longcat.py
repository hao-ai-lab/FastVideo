# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: LongCat-Video (native FastVideo impl) transformer block 0.

Spec notes: LongCatVideoConfig() arch defaults match the mirror's
transformer/config.json (hidden 4096, 32 heads, mlp_ratio 4, adaln 512).
The block __init__ needs the FULL LongCatVideoConfig, not the arch config —
self_attn reads config._supported_attention_backends (longcat.py:273) while
cross_attn reads config.arch_config._supported_attention_backends (:548).
Checkpoint is a SINGLE fp32 file (transformer/model.safetensors, no
index.json) with zero renames — the mirror is pre-converted, keys match
module names. 3D RoPE is built inside the block from latent_shape; t is
per-latent-frame temb [B, T, 512]. Defaults (return_kv=False) return one
plain tensor, so no postprocess. Backend FLASH_ATTN per the ssim test
(tests/ssim/test_longcat_similarity.py:200).
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

HIDDEN_SIZE = 4096  # configs/models/dits/longcat.py:77
ADALN_DIM = 512  # configs/models/dits/longcat.py:93


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.longcat import LongCatVideoConfig
    from fastvideo.models.dits.longcat import LongCatTransformerBlock

    config = LongCatVideoConfig()
    arch = config.arch_config
    return LongCatTransformerBlock(
        hidden_size=arch.hidden_size,
        num_heads=arch.num_attention_heads,
        mlp_ratio=arch.mlp_ratio,
        adaln_tembed_dim=arch.adaln_tembed_dim,
        config=config,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    # forward signature (longcat.py:773-783): x [B, N, 4096], context
    # [B, N_text, 4096], t [B, T, 512], latent_shape (T, H, W), N == T*H*W.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t_frames, h, w = 3, 8, 8
    seq = t_frames * h * w
    n_text = 32  # production pads to 512; any length works

    hidden = torch.randn(1, seq, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    context = torch.randn(1, n_text, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, t_frames, ADALN_DIM, generator=generator, dtype=torch.float32)

    return {
        "x": hidden.to(device=device, dtype=torch.bfloat16),
        "context": context.to(device=device, dtype=torch.bfloat16),
        "t": temb.to(device=device, dtype=torch.bfloat16),
        "latent_shape": (t_frames, h, w),
    }


SPEC = GateSpec(
    name="longcat_t2v",
    repo_id="FastVideo/LongCat-Video-T2V-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="blocks.{N}.",
    renames=(),
    attention_backend="FLASH_ATTN",
    model_root_env="LONGCAT_MODEL_ROOT",
    weight_file="model.safetensors",  # single 54.3 GB fp32 file, no index.json
)


def test_longcat_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
