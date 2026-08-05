# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: Z-Image-Turbo (Tongyi-MAI) main-stack transformer block 0.

Spec notes: ZImageDiTArchConfig defaults match the checkpoint's
transformer/config.json (dim=3840, n_heads=n_kv_heads=30, adaln_embed_dim=256).
Checkpoint keys match FastVideo module names exactly — no renames. Rotary is
built OUTSIDE the block: freqs_cis is complex64 from RopeEmbedder(theta=256,
axes_dims=(32,48,48)) and must stay complex64 (rotary math runs fp32 inside
attention, zimage.py:104-108). adaln_input is width min(dim, adaln_embed_dim)
= 256, not dim. Attention is plain F.scaled_dot_product_attention with a bool
[B, S] key-padding mask (True=attend); the block ignores fastvideo attention
backends, but TORCH_SDPA matches the ssim test and keys the golden filename.
Block returns a bare tensor [B, S, dim], no postprocess needed.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

DIM = 3840  # 30 heads x 128, Z-Image-Turbo transformer/config.json


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.zimage import ZImageDiTArchConfig
    from fastvideo.models.dits.zimage import ZImageTransformerBlock

    arch = ZImageDiTArchConfig()
    return ZImageTransformerBlock(
        layer,  # main `layers` stack uses the plain index (zimage.py:337-338)
        arch.dim,
        arch.n_heads,
        arch.n_kv_heads,
        arch.norm_eps,
        arch.qk_norm,
        arch.adaln_embed_dim,
        modulation=True,  # main-stack blocks are modulated (zimage.py:337-338)
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    """Fixed padded batch: 2 sequences over an 8x8 image grid, lengths [64, 48]."""
    from fastvideo.models.dits.zimage import RopeEmbedder, ZImageTransformer2DModel

    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, seq = 2, 64
    lengths = [64, 48]

    # freqs_cis exactly as the model builds it (zimage.py:342, 527-531); image
    # position ids start at frame index cap_len+1 -> use start=(1, 0, 0).
    rope = RopeEmbedder(theta=256.0, axes_dims=(32, 48, 48), axes_lens=(1536, 512, 512))
    pos_ids = ZImageTransformer2DModel.create_coordinate_grid(
        (1, 8, 8), start=(1, 0, 0)).flatten(0, 2)  # [64, 3] int32
    freqs_cis = rope(pos_ids).unsqueeze(0).expand(batch, seq, 64).contiguous()  # complex64 [2, 64, 64]

    attn_mask = torch.zeros(batch, seq, dtype=torch.bool)
    for i, length in enumerate(lengths):
        attn_mask[i, :length] = True

    x = torch.randn(batch, seq, DIM, generator=generator, dtype=torch.float32)
    adaln = torch.randn(batch, 256, generator=generator, dtype=torch.float32)

    return {
        "x": x.to(device=device, dtype=torch.bfloat16),
        "attn_mask": attn_mask.to(device),
        "freqs_cis": freqs_cis.to(device),  # stays complex64, never cast to bf16
        "adaln_input": adaln.to(device=device, dtype=torch.bfloat16),
    }


SPEC = GateSpec(
    name="zimage_turbo",
    repo_id="Tongyi-MAI/Z-Image-Turbo",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="layers.{N}.",
    renames=(),
    attention_backend="TORCH_SDPA",
    model_root_env="ZIMAGE_MODEL_DIR",
)


def test_zimage_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
