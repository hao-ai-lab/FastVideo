# SPDX-License-Identifier: Apache-2.0
"""Golden-gate: FLUX.1-dev double-stream transformer block 0.

Spec notes: dim = 24 heads x 128 = 3072 (configs/models/dits/flux.py:16-17).
Checkpoint names load 1:1 (param_names_mapping is empty, flux.py:391-393) so
no renames. RoPE is built outside the block by FluxPosEmbed(theta=10000,
axes_dim=[16, 56, 56]) with text ids FIRST (flux.py:414-415, 536-537); cos/sin
are float64 [S, 128]. temb is [B, 3072] (SD3AdaLayerNormZero 6-way chunk,
sd3.py:394-395). Forward returns the TUPLE (encoder_hidden_states,
hidden_states) (flux.py:329) — postprocess cats both streams along seq into
one golden. The flux SSIM test pins TORCH_SDPA. black-forest-labs/FLUX.1-dev
is a GATED HF repo; FLUX_T2I_MODEL_DIR can point at a local checkout.
"""

from __future__ import annotations

import torch

from fastvideo.tests.golden_gate._harness import GateSpec, distributed_runtime, run_gate

__all__ = ["distributed_runtime"]

DIM = 3072  # 24 heads x 128, FluxTransformer2DArchConfig defaults


def _build_block(layer: int) -> torch.nn.Module:
    from fastvideo.configs.models.dits.flux import FluxTransformer2DArchConfig
    from fastvideo.models.dits.flux import FluxTransformer2DModel, FluxTransformerBlock

    arch = FluxTransformer2DArchConfig()
    return FluxTransformerBlock(
        dim=arch.num_attention_heads * arch.attention_head_dim,  # 3072
        num_attention_heads=arch.num_attention_heads,  # 24
        attention_head_dim=arch.attention_head_dim,  # 128
        supported_attention_backends=FluxTransformer2DModel._supported_attention_backends,
    )


def _make_inputs(device: torch.device, seed: int) -> dict:
    from fastvideo.models.dits.flux import FluxPosEmbed

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_txt, h, w = 32, 16, 16
    n_img = h * w

    # Diffusers-style ids: txt ids all zeros; img ids carry (0, row, col).
    txt_ids = torch.zeros(n_txt, 3, dtype=torch.float64)
    img_ids = torch.zeros(n_img, 3, dtype=torch.float64)
    img_ids[:, 1] = torch.arange(h, dtype=torch.float64).repeat_interleave(w)
    img_ids[:, 2] = torch.arange(w, dtype=torch.float64).repeat(h)

    rope = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])
    cos, sin = rope(torch.cat([txt_ids, img_ids], dim=0))  # text ids first

    hidden = torch.randn(1, n_img, DIM, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, n_txt, DIM, generator=generator, dtype=torch.float32)
    temb = torch.randn(1, DIM, generator=generator, dtype=torch.float32)

    return {
        "hidden_states": hidden.to(device=device, dtype=torch.bfloat16),
        "encoder_hidden_states": encoder.to(device=device, dtype=torch.bfloat16),
        "temb": temb.to(device=device, dtype=torch.bfloat16),
        "image_rotary_emb": (cos.to(device), sin.to(device)),
        # joint_attention_kwargs is accepted but immediately deleted — omit.
    }


SPEC = GateSpec(
    name="flux1_dev",
    repo_id="black-forest-labs/FLUX.1-dev",
    build_block=_build_block,
    make_inputs=_make_inputs,
    prefix_template="transformer_blocks.{N}.",
    renames=(),  # HF weight names already match this module layout (flux.py:391-393)
    attention_backend="TORCH_SDPA",
    # forward returns (encoder_hidden_states, hidden_states) — both [1, S, 3072]
    postprocess=lambda out: torch.cat(out, dim=1),
    model_root_env="FLUX_T2I_MODEL_DIR",
)


def test_flux_golden_gate(distributed_runtime) -> None:
    run_gate(SPEC)
