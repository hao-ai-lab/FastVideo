# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the packed MiniMax-H3 DiT NVFP4 export overlay."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import fastvideo.layers.quantization.nvfp4_config as nv


def _method(prefix: str) -> nv.NVFP4QuantizeMethod:
    method = object.__new__(nv.NVFP4QuantizeMethod)
    method.weight_fp4 = None
    method.weight_scale = None
    method.x_global_sf = torch.tensor(1.0, dtype=torch.float32)
    method.layer_prefix = prefix
    method._is_refine_only_layer = False
    method._retain_original_weights = None
    return method


class _ExportLinear(nn.Module):

    def __init__(self, prefix: str, out_dim: int = 8, in_dim: int = 16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(out_dim, in_dim, dtype=torch.bfloat16), requires_grad=False)
        self.quant_method = _method(prefix)


def _model() -> nn.Module:
    root = nn.Module()
    root.transformer_blocks = nn.ModuleList([nn.Module()])
    root.transformer_blocks[0].attn = nn.Module()
    root.transformer_blocks[0].attn.to_q = _ExportLinear("transformer_blocks.0.attn.to_q")
    return root


def test_dense_transformer_safetensors_drops_packed_export(tmp_path) -> None:
    packed = tmp_path / nv.H3_NVFP4_DIT_EXPORT_FILENAME
    shard = tmp_path / "diffusion_pytorch_model-00001-of-00008.safetensors"
    packed.write_bytes(b"")
    shard.write_bytes(b"")
    kept = nv.dense_transformer_safetensors([str(packed), str(shard)])
    assert kept == [str(shard)]
    assert nv.find_minimax_h3_nvfp4_dit_export([str(shard)]) == str(packed)


def test_load_minimax_h3_nvfp4_dit_export_overlays_and_purges_weight(tmp_path) -> None:
    model = _model()
    linear = model.transformer_blocks[0].attn.to_q
    prefix = "transformer_blocks.0.attn.to_q"
    tensors = {
        f"{prefix}::_nvfp4_weight": torch.zeros(8, 8, dtype=torch.uint8),
        f"{prefix}::_nvfp4_weight_scale": torch.zeros(8, 1, dtype=torch.uint8),
        f"{prefix}::_nvfp4_alpha": torch.tensor(0.5, dtype=torch.float32),
        f"{prefix}::_weight_global_sf": torch.tensor(2.0, dtype=torch.bfloat16),
    }
    path = tmp_path / nv.H3_NVFP4_DIT_EXPORT_FILENAME
    save_file(tensors, str(path))

    loaded = nv.load_minimax_h3_nvfp4_dit_export(model, str(path), device="cpu")

    assert loaded == 1
    assert linear.weight is None
    assert linear._nvfp4_weight.dtype is torch.uint8
    assert tuple(linear._nvfp4_weight.shape) == (8, 8)
    assert linear._nvfp4_alpha.item() == 0.5
    assert linear._weight_global_sf.item() == 2.0


def test_load_minimax_h3_nvfp4_dit_export_rejects_untagged_linear(tmp_path) -> None:
    model = _model()
    model.transformer_blocks[0].attn.to_q.quant_method = object()
    prefix = "transformer_blocks.0.attn.to_q"
    tensors = {
        f"{prefix}::_nvfp4_weight": torch.zeros(8, 8, dtype=torch.uint8),
        f"{prefix}::_nvfp4_weight_scale": torch.zeros(8, 1, dtype=torch.uint8),
        f"{prefix}::_nvfp4_alpha": torch.tensor(1.0, dtype=torch.float32),
        f"{prefix}::_weight_global_sf": torch.tensor(1.0, dtype=torch.bfloat16),
    }
    path = tmp_path / nv.H3_NVFP4_DIT_EXPORT_FILENAME
    save_file(tensors, str(path))

    with pytest.raises(RuntimeError, match="layer_profile='h3_dit'"):
        nv.load_minimax_h3_nvfp4_dit_export(model, str(path), device="cpu")
