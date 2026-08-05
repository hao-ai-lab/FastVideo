# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: GLM-Image (GlmImageTransformer2DModel) transformer block 0.

Spec notes: inner_dim = 32 heads x 128 = 4096 (glm_image.py:631, arch config
defaults). RoPE is built outside the block by GlmImageRotaryPosEmbed
(glm_image.py:727-729) from only the shape/device of the pre-patchify latent
(B, 16, H, W); cos/sin are fp32 (image_seq, 64). temb is passed post-SiLU in
the hidden dtype (glm_image.py:747-750), so bf16 here. The block forward
returns a TUPLE (hidden_states, encoder_hidden_states) (glm_image.py:502) —
postprocess cats encoder first along seq, matching the attention layout
(glm_image.py:343). attn1.to_out is nn.ModuleList so "attn1.to_out.0.*"
matches natively; only the two ff.net renames apply. The SSIM test pins
TORCH_SDPA and loads local weights via GLM_IMAGE_MODEL_DIR.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

INNER_DIM = 4096  # 32 heads x 128, GlmImageDiTArchConfig defaults
TEXT_LEN = 32
LATENT_H = LATENT_W = 64  # patch_size=2 -> 32x32 = 1024 image tokens


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.glm_image import GlmImageDiTArchConfig
    from fastvideo.models.dits.glm_image import (GlmImageTransformer2DModel,
                                                 GlmImageTransformerBlock)

    arch = GlmImageDiTArchConfig()
    return GlmImageTransformerBlock(
        arch.num_attention_heads * arch.attention_head_dim,  # dim = 4096
        arch.num_attention_heads,                            # 32
        arch.attention_head_dim,                             # 128
        arch.time_embed_dim,                                 # 512
        supported_attention_backends=GlmImageTransformer2DModel._supported_attention_backends,
        prefix=f"transformer_blocks.{layer}",
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.glm_image import GlmImageRotaryPosEmbed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_image = (LATENT_H // 2) * (LATENT_W // 2)  # 1024 tokens

    # rope consumes only shape+device of the raw latent (glm_image.py:518-522)
    rope = GlmImageRotaryPosEmbed(dim=128, patch_size=2, theta=10000.0)
    cos, sin = rope(torch.empty(1, 16, LATENT_H, LATENT_W, device=device))

    hidden = torch.randn(1, n_image, INNER_DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, TEXT_LEN, INNER_DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, 512, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "image_rotary_emb": (cos, sin),  # fp32, already on device
    }


SPEC = GateSpec(
    name="glm_image",
    repo_id="zai-org/GLM-Image",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    renames=(
        (r"\.ff\.net\.0\.proj\.", ".ff.fc_in."),
        (r"\.ff\.net\.2\.", ".ff.fc_out."),
    ),
    attention_backend="TORCH_SDPA",
    # forward returns (hidden_states, encoder_hidden_states); encoder first,
    # like the attention concat layout (glm_image.py:343)
    postprocess=lambda out: torch.cat([out[1], out[0]], dim=1),
    model_root_env="GLM_IMAGE_MODEL_DIR",
)


def test_glm_image_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
