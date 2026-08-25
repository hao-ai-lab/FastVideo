# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: Kandinsky-5.0-T2V-Lite visual decoder block 0.

Spec notes: Kandinsky5ArchConfig defaults (model_dim=2048, ff_dim=5120) do NOT
match the Lite checkpoint — production overrides them from
transformer/config.json via update_model_arch, so the geometry is pinned here
explicitly from that config.json (model_dim=1792, ff_dim=7168, time_dim=512,
axes_dims=(16, 24, 24), attention_type="regular" -> use_nabla=False). RoPE is
built outside the block exactly as Kandinsky5Transformer3DModel.forward does
(kandinsky5.py:762-770): Kandinsky5RoPE3D returns real 2x2 rotation matrices
[B, T, H, W, 1, head_dim/2, 2, 2] fp32, flattened over (T, H, W) for the
dense path. time_embed reaches the block as fp32 (modulation computes in fp32
under autocast); sparse_params is a required positional -> None for the dense
"regular" path. Single-file checkpoint (no index.json) — the harness's
index-less fallback handles it. The ssim test pins FASTVIDEO_FA4=0 alongside
FLASH_ATTN; mirrored here before any fastvideo import.
"""

from __future__ import annotations

import os

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

MODEL_DIM = 1792  # Kandinsky-5.0-T2V-Lite transformer/config.json
TIME_DIM = 512
FF_DIM = 7168
AXES_DIMS = (16, 24, 24)  # head_dim 64, 28 heads


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.kandinsky5 import Kandinsky5ArchConfig
    from fastvideo.models.dits.kandinsky5 import Kandinsky5TransformerDecoderBlock

    arch = Kandinsky5ArchConfig(
        model_dim=MODEL_DIM,
        ff_dim=FF_DIM,
        time_dim=TIME_DIM,
        axes_dims=AXES_DIMS,
        attention_type="regular",
    )
    return Kandinsky5TransformerDecoderBlock(
        arch.model_dim,
        arch.time_dim,
        arch.ff_dim,
        sum(arch.axes_dims),
        arch._supported_attention_backends,
        prefix=f"Kandinsky5.visual_transformer_blocks.{layer}",
        use_nabla=arch.attention_type == "nabla",
        quant_config=None,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.kandinsky5 import Kandinsky5RoPE3D

    generator = torch.Generator(device="cpu").manual_seed(seed)
    b, t, h, w = 1, 2, 4, 4  # post-patchify grid (parity test uses the same)
    seq = t * h * w  # 32
    n_text = 8

    visual = torch.randn(b, seq, MODEL_DIM, generator=generator, dtype=torch.float32)
    text = torch.randn(b, n_text, MODEL_DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(b, TIME_DIM, generator=generator, dtype=torch.float32)

    # Built exactly like model.forward (kandinsky5.py:762-770): rotation
    # matrices [B, T, H, W, 1, head_dim/2, 2, 2] fp32, then the dense
    # fractal_flatten(block_mask=False) is rope.flatten(1, 3) (line 109).
    rope3d = Kandinsky5RoPE3D(AXES_DIMS)
    rope = rope3d((b, t, h, w), [torch.arange(t), torch.arange(h), torch.arange(w)])
    rope = rope.flatten(1, 3)

    return {
        "visual_embed": visual.to(device=device, dtype=torch.bfloat16),
        "text_embed": text.to(device=device, dtype=torch.bfloat16),
        "time_embed": temb.to(device=device),  # stays fp32 (kandinsky5.py:745-746)
        "rope": rope.to(device=device),  # fp32 2x2 rotation matrices
        "sparse_params": None,  # required positional; dense "regular" path
    }


SPEC = GateSpec(
    name="kandinsky5_t2v_lite",
    repo_id="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="visual_transformer_blocks.{N}.",
    renames=(
        (r"\.feed_forward\.in_layer\.", ".feed_forward.mlp.fc_in."),
        (r"\.feed_forward\.out_layer\.", ".feed_forward.mlp.fc_out."),
    ),
    attention_backend="FLASH_ATTN",
)


def test_kandinsky5_golden_gate(distributed_runtime, monkeypatch) -> None:
    # Pin FA4 off for THIS test only (test_kandinsky5_similarity.py:84). A
    # module-level os.environ.setdefault here leaked into every other
    # golden-gate test's env fingerprint at pytest collection time, breaking
    # full-directory runs in environments where FASTVIDEO_FA4 is unset
    # (observed on the self-hosted CI runner lane).
    monkeypatch.setenv("FASTVIDEO_FA4", os.environ.get("FASTVIDEO_FA4", "0"))
    run_gate(SPEC)
