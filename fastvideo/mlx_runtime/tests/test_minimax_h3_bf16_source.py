# SPDX-License-Identifier: Apache-2.0
"""Reject packed source bytes before they can become invalid affine weights."""
import pytest

mx = pytest.importorskip("mlx.core")
from fastvideo.mlx_runtime.minimax_h3 import mlx_h3_dit_from_diffusers_safetensors


def test_packed_transformer_source_is_rejected(tmp_path):
    mx.save_safetensors(str(tmp_path / "diffusion_pytorch_model.safetensors"), {
        "transformer_blocks.0.attn.to_q.weight": mx.zeros((128, 128), dtype=mx.uint8),
    })
    with pytest.raises(ValueError, match="released BF16 transformer"):
        mlx_h3_dit_from_diffusers_safetensors(
            tmp_path, config={"num_layers": 1, "num_refiner_layers": 0},
            quantization="int6", include_vsa=True,
        )
