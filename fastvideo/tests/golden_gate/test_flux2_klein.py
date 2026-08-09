# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: FLUX.2-klein-4B double-stream transformer block 0.

Spec notes: Klein-4B block dims equal the Flux2ArchConfig defaults (24 heads x
128 head_dim -> 3072, mlp_ratio 3.0, eps 1e-6, bias=False); the model-level
config diffs (num_layers=5, joint_attention_dim=7680, ...) never enter the
block ctor. Checkpoint is SINGLE-FILE (transformer/diffusion_pytorch_model.
safetensors, no index.json) and its keys match FastVideo names exactly, so no
renames. RoPE lives outside the block and is built exactly as
Flux2Transformer2DModel.forward does (flux_2.py:1043-1054); cos/sin stay fp32
— the block never casts them to bf16. Modulation params arrive as two
(shift, scale, gate) 3-tuples per stream, each tensor [1, 1, 3072] bf16.
Block construction needs the distributed fixture (Flux2FeedForward reads
get_tp_world_size() in __init__). forward returns (encoder_hidden_states,
hidden_states) — postprocess concats them on dim=1. Backend TORCH_SDPA per
the flux2 ssim lane (test_flux2_similarity.py:86).
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

DIM = 3072  # 24 heads x 128, Flux2ArchConfig defaults == Klein-4B block dims
TEXT_LEN = 32
IMG_H, IMG_W = 8, 16  # 128 image tokens


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.flux_2 import Flux2ArchConfig
    from fastvideo.models.dits.flux_2 import Flux2TransformerBlock

    arch = Flux2ArchConfig()
    return Flux2TransformerBlock(
        dim=arch.num_attention_heads * arch.attention_head_dim,
        num_attention_heads=arch.num_attention_heads,
        attention_head_dim=arch.attention_head_dim,
        mlp_ratio=arch.mlp_ratio,
        eps=arch.eps,
        bias=False,
        supported_attention_backends=arch._supported_attention_backends,
        ff_context_swiglu_fp32=arch.ff_context_swiglu_fp32,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.flux_2 import (
        Flux2PosEmbed,
        compute_flux2_freqs_cis_from_ids,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_img = IMG_H * IMG_W

    hidden = torch.randn(1, n_img, DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, TEXT_LEN, DIM, generator=generator, dtype=torch.float32)

    def _mod_sets(n_sets: int):
        # (shift, scale, gate) per set, [1, 1, DIM] each, in the DiT dtype.
        return tuple(
            tuple(
                torch.randn(1, 1, DIM, generator=generator, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
                for _ in range(3)) for _ in range(n_sets))

    # ids built as in Flux2Transformer2DModel.forward (flux_2.py:1044-1051)
    txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(TEXT_LEN))
    img_ids = torch.cartesian_prod(torch.arange(1), torch.arange(IMG_H), torch.arange(IMG_W), torch.arange(1))
    rope = Flux2PosEmbed(theta=2000, axes_dim=[32, 32, 32, 32])
    freqs_cis = compute_flux2_freqs_cis_from_ids(rope, txt_ids, img_ids, device=device)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb_mod_params_img": _mod_sets(2),
        "temb_mod_params_txt": _mod_sets(2),
        "freqs_cis": freqs_cis,  # fp32 (cos, sin) [S_txt + S_img, 128] — keep fp32
    }


SPEC = GateSpec(
    name="flux2_klein_4b",
    repo_id="black-forest-labs/FLUX.2-klein-4B",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    renames=(),
    attention_backend="TORCH_SDPA",
    # (txt, img) tuple -> one tensor; both are [1, S, DIM] so cat on dim=1.
    postprocess=lambda out: torch.cat(out, dim=1),
)


def test_flux2_klein_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
