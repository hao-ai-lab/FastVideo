# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: HunyuanGameCraft double-stream block 0 (Hunyuan-video arch).

Spec notes: the repeated block is MMDoubleStreamBlock from hunyuanvideo.py
(imported by hunyuangamecraft.py:24-28); ctor mirrors
HunyuanGameCraftTransformer3DModel.__init__ (hunyuangamecraft.py:252-261) with
hidden_size 3072 = 24 heads x 128. Checkpoint is a SINGLE FILE (no index.json)
under transformer/ and its keys are already in FastVideo form, so renames is
empty. RoPE is built over the video grid only, rope_dim_list (16,56,56),
rope_theta 256 (hunyuangamecraft.py:340-348) — applied inside
DistributedAttention to the img qkv, so freqs_cis length must equal the img
sequence length. Forward returns an (img, txt) TUPLE (hunyuanvideo.py:280);
postprocess concatenates along the sequence dim into one tensor.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

HIDDEN = 3072  # 24 heads x 128, configs/models/dits/hunyuangamecraft.py:125


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.hunyuangamecraft import HunyuanGameCraftArchConfig
    from fastvideo.models.dits.hunyuanvideo import MMDoubleStreamBlock

    arch = HunyuanGameCraftArchConfig()
    return MMDoubleStreamBlock(
        hidden_size=arch.hidden_size,  # 3072
        num_attention_heads=arch.num_attention_heads,  # 24
        mlp_ratio=arch.mlp_ratio,  # 4.0
        dtype=arch.dtype,  # None
        supported_attention_backends=arch._supported_attention_backends,
        prefix=f"hunyuan_gamecraft.double_blocks.{layer}",
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    """3x4x5 = 60 video tokens + 32 text tokens, hidden 3072.

    Shapes per MMDoubleStreamBlock.forward (hunyuanvideo.py:193-200):
    img [B, S_img, 3072], txt [B, S_txt, 3072], vec [B, 3072],
    freqs_cis = (cos, sin) each [S_img, 128].
    """
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tt, th, tw = 3, 4, 5
    img_seq, txt_seq = tt * th * tw, 32

    freqs_cos, freqs_sin = get_rotary_pos_embed((tt, th, tw), HIDDEN, 24, [16, 56, 56], 256)  # -> [60, 128] fp32 each

    img = torch.randn(1, img_seq, HIDDEN, generator=generator, dtype=torch.float32)
    txt = torch.randn(1, txt_seq, HIDDEN, generator=generator, dtype=torch.float32)
    vec = torch.randn(1, HIDDEN, generator=generator, dtype=torch.float32)

    return {
        "img": img.to(device=device, dtype=torch.bfloat16),
        "txt": txt.to(device=device, dtype=torch.bfloat16),
        "vec": vec.to(device=device, dtype=torch.bfloat16),
        "freqs_cis": (freqs_cos.to(device=device,
                                   dtype=torch.bfloat16), freqs_sin.to(device=device, dtype=torch.bfloat16)),
    }


SPEC = GateSpec(
    name="gamecraft",
    repo_id="FastVideo/HunyuanGameCraft-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="double_blocks.{N}.",
    renames=(),  # checkpoint keys already match the block's state_dict 1:1
    attention_backend="FLASH_ATTN",
    postprocess=lambda out: torch.cat(out, dim=1),  # (img, txt) tuple -> one tensor
    model_root_env="GAMECRAFT_MODEL_PATH",
)


def test_gamecraft_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
