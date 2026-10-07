# SPDX-License-Identifier: Apache-2.0
"""Converter smoke test: a synthetic ModelOpt NVFP4 checkpoint, converted and loaded, matches its reference."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file

import fastvideo.layers.quantization.nvfp4_config as nv

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py"
OUT, IN = 128, 128
ACT_AMAX = 6.0

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
                                reason="NVFP4 mm_fp4 needs a Blackwell GPU")


def _converter():
    spec = importlib.util.spec_from_file_location("convert_minimax_h3_modelopt_nvfp4_dit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _modelopt_linear(prefix: str, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Random E2M1 codes with positive E4M3 block scales: a valid ModelOpt unified-layout linear."""
    return {
        f"{prefix}.weight": torch.randint(0, 256, (OUT, IN // 2), dtype=torch.uint8, generator=gen),
        f"{prefix}.weight_scale": (torch.rand(OUT, IN // 16, generator=gen) + 0.5).to(torch.float8_e4m3fn),
        f"{prefix}.weight_scale_2": torch.tensor(0.01, dtype=torch.float32),
        f"{prefix}.input_scale": torch.tensor(1.0, dtype=torch.float32),
    }


def _write_source(src: Path) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    tensors = {
        **_modelopt_linear("transformer_blocks.0.ff.net.0.proj", gen),
        # Already quantized by ModelOpt: --quantize-attention must keep it on the ModelOpt path.
        **_modelopt_linear("transformer_blocks.0.attn.to_k", gen),
        "transformer_blocks.0.attn.to_q.weight": (torch.randn(OUT, IN, generator=gen) * 0.05).to(torch.bfloat16),
        "transformer_blocks.0.norm.weight": torch.ones(IN, dtype=torch.bfloat16),
    }
    src.mkdir()
    save_file(tensors, str(src / "diffusion_pytorch_model-00001-of-00001.safetensors"))
    weight_map = {k: "diffusion_pytorch_model-00001-of-00001.safetensors" for k in tensors}
    (src / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (src / "config.json").write_text(json.dumps({"quantization_config": {"quant_algo": "NVFP4"}}))
    return tensors


class _Linear(nn.Module):

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(OUT, IN, dtype=torch.bfloat16), requires_grad=False)
        self.quant_method = nv.NVFP4QuantizeMethod(layer_prefix=prefix)


def _model() -> nn.Module:
    root = nn.Module()
    block = nn.Module()
    block.ff, block.attn = nn.Module(), nn.Module()
    block.ff.fc_in = _Linear("transformer_blocks.0.ff.fc_in")
    block.attn.to_q = _Linear("transformer_blocks.0.attn.to_q")
    block.attn.to_k = _Linear("transformer_blocks.0.attn.to_k")
    root.transformer_blocks = nn.ModuleList([block])
    return root


def test_converted_checkpoint_loads_and_matches_reference(tmp_path) -> None:
    pytest.importorskip("flashinfer")
    conv = _converter()
    src, dst = tmp_path / "src", tmp_path / "dst"
    source = _write_source(src)
    amax = tmp_path / "amax.json"
    amax.write_text(json.dumps({f"b0.{name}": {"all": ACT_AMAX} for name in ("ff.fc_in", "attn.to_q", "attn.to_k")}))
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--dst", str(dst), "--quantize-attention",
                    "--act-amax", str(amax)], check=True)

    export = dst / nv.H3_NVFP4_DIT_EXPORT_FILENAME
    with safe_open(str(export), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        # ModelOpt bytes are carried over bit for bit, including the attention projection it already quantized.
        for prefix, module in (("transformer_blocks.0.ff.net.0.proj", "transformer_blocks.0.ff.fc_in"),
                               ("transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_k")):
            assert torch.equal(reader.get_tensor(f"{module}::_nvfp4_weight"), source[f"{prefix}.weight"])
    assert "transformer_blocks.0.attn.to_q::_nvfp4_weight" in keys
    dense_index = json.loads((dst / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    assert set(dense_index) == {"transformer_blocks.0.norm.weight"}
    assert "quantization_config" not in json.loads((dst / "config.json").read_text())

    model = _model()
    assert nv.load_minimax_h3_nvfp4_dit_export(model, str(export), device="cuda") == 3
    x = torch.randn(256, IN, device="cuda").clamp(-ACT_AMAX, ACT_AMAX).to(torch.bfloat16)
    references = {
        "ff.fc_in": conv.dequantize_modelopt(*(source[f"transformer_blocks.0.ff.net.0.proj.{s}"].cuda()
                                               for s in ("weight", "weight_scale", "weight_scale_2"))),
        "attn.to_k": conv.dequantize_modelopt(*(source[f"transformer_blocks.0.attn.to_k.{s}"].cuda()
                                                for s in ("weight", "weight_scale", "weight_scale_2"))),
        "attn.to_q": source["transformer_blocks.0.attn.to_q.weight"].cuda(),
    }
    block = model.transformer_blocks[0]
    for name, reference in references.items():
        layer = block.get_submodule(name)
        assert layer._nvfp4_input_global_sf.item() == pytest.approx(448.0 * 6.0 / ACT_AMAX)
        out = layer.quant_method.apply(layer, x).float()
        ref = x.float() @ reference.float().T
        error = ((out - ref).norm() / ref.norm()).item()
        assert error < 0.2, f"{name}: relative error {error:.3f}"
