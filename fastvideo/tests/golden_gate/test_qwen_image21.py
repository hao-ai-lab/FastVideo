# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: Qwen-Image-2.1 transformer block 0 with real BF16 weights.

The small packed sequence contains text, one reference image, more text, and
the target image. Causal text segments use masked SDPA; reference and target
segments use FlashAttention, including their unequal query/key lengths. Complex RoPE and
the two modulation rows match the causal-conditioning block interface.

Requires a BF16-capable CUDA GPU and FlashAttention. Select FA4 with
FASTVIDEO_FA4=1 before starting Python. The shared harness downloads
only checkpoint shards containing block 0 and compares a device/runtime-matched
golden exactly; it never loads the text encoder or VAE.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

DIM = 4096  # 32 heads x 128, Qwen-Image-2.1 transformer/config.json
TEXT_BEFORE, TEXT_AFTER = 32, 16
REFERENCE_H, REFERENCE_W = 4, 4
TARGET_H, TARGET_W = 8, 16
CHECKPOINT_REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.qwen_image21 import QwenImage21ArchConfig
    from fastvideo.models.dits.qwen_image21 import QwenImage21TransformerBlock
    from fastvideo.platforms import AttentionBackendEnum

    arch = QwenImage21ArchConfig()
    # The selector reads the construction dtype before the harness casts the
    # loaded block. Keep it BF16 here so FLASH_ATTN cannot fall back to SDPA.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        block = QwenImage21TransformerBlock(
            dim=arch.num_attention_heads * arch.attention_head_dim,
            num_attention_heads=arch.num_attention_heads,
            attention_head_dim=arch.attention_head_dim,
            mlp_ratio=arch.mlp_ratio,
            eps=arch.eps,
            requested_backend=AttentionBackendEnum.FLASH_ATTN,
            supported_attention_backends=arch._supported_attention_backends,
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    assert block.attn.dense_attn is not None, "Qwen golden requires dense FlashAttention"
    assert block.attn.dense_attn.backend is AttentionBackendEnum.FLASH_ATTN
    return block


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.configs.models.dits.qwen_image21 import QwenImage21ArchConfig
    from fastvideo.models.dits.qwen_image21 import (
        QwenImage21Rope,
        QwenImage21Transformer2DModel,
        _qwenimage21_prefix_segments,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    reference_tokens = REFERENCE_H * REFERENCE_W
    target_tokens = TARGET_H * TARGET_W
    image_pad_mask = torch.tensor(
        [False] * TEXT_BEFORE + [True] * reference_tokens + [False] * TEXT_AFTER + [True] * target_tokens,
        dtype=torch.bool,
        device=device,
    )
    img_shapes = [(1, REFERENCE_H, REFERENCE_W), (1, TARGET_H, TARGET_W)]
    image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)
    rope = QwenImage21Rope(theta=10000, axes_dim=QwenImage21ArchConfig().axes_dims_rope)

    def rnd(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)

    return {
        "hidden_states": rnd(1, image_pad_mask.numel(), DIM),
        # Row 0 conditions target tokens; row 1 is the prefix's zero-timestep row.
        "modulation": rnd(2, 4 * DIM),
        "target_token_mask": target_token_mask,
        "rotary_emb": rope(img_shapes, image_pad_mask, device),
        "segments": _qwenimage21_prefix_segments(image_ids, image_pad_mask.numel() - target_tokens),
    }


SPEC = GateSpec(
    name="qwen_image21",
    repo_id="Qwen/Qwen-Image-2.1",
    revision=CHECKPOINT_REVISION,
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    attention_backend="FLASH_ATTN",
)


def test_qwen_image21_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
