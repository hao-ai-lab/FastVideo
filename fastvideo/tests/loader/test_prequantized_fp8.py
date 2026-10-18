# SPDX-License-Identifier: Apache-2.0
"""Packed H3 FP8 exports preserve dense linears through post-load conversion.

Uses a tiny synthetic checkpoint on CPU, plus CUDA when available. No model
downloads or distributed process groups are required.
"""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from fastvideo.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from fastvideo.layers.quantization.fp8_config import FP8Config, FP8QuantizeMethod
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model


class MiniMaxH3Tiny(torch.nn.Module):
    # Different checkpoint names exercise source-to-runtime name mapping.
    param_names_mapping = {r"^export_main\.": "main.", r"^export_refiner\.": "refiner."}

    def __init__(self, config):
        super().__init__()
        self.main = ReplicatedLinear(128, 64, bias=False, params_dtype=torch.bfloat16,
                                     quant_config=config.quant_config, prefix="transformer_blocks.0.attn.to_q")
        self.refiner = ReplicatedLinear(128, 64, bias=True, params_dtype=torch.bfloat16,
                                        quant_config=config.quant_config, prefix="token_refiner.0.attn.to_q")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA companion check requires a GPU")
    if request.param == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(request.param)


def _load(paths, device, quant_config):
    return maybe_load_fsdp_model(
        model_cls=MiniMaxH3Tiny,
        init_params={"config": SimpleNamespace(quant_config=quant_config)},
        weight_dir_list=[str(p) for p in paths],
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=torch.bfloat16,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cpu_offload=True,  # Stage source tensors on CPU on both test devices.
        training_mode=False,
    )


def _dense_weights():
    generator = torch.Generator().manual_seed(42)
    return {
        "export_main.weight": torch.randn(64, 128, generator=generator).to(torch.bfloat16),
        "export_refiner.weight": torch.randn(64, 128, generator=generator).to(torch.bfloat16),
        "export_refiner.bias": torch.randn(64, generator=generator).to(torch.bfloat16),
    }


def test_packed_fp8_preserves_dense_refiner(tmp_path, device):
    weights = _dense_weights()
    packed_weight = weights.pop("export_main.weight").to(torch.float8_e4m3fn)
    scale = torch.linspace(0.01, 0.02, 64, dtype=torch.float32).reshape(64, 1)
    # Weight and scale may live in separate shards.
    first, second = tmp_path / "first.safetensors", tmp_path / "second.safetensors"
    save_file({"export_main.weight": packed_weight}, str(first))
    save_file({**weights, "export_main.weight_scale": scale}, str(second))

    model = _load([first, second], device, FP8Config())

    assert isinstance(model.main.quant_method, FP8QuantizeMethod)
    assert model.main.quant_method.granularity == "channel"
    assert model.main._fp8_weight.dtype == torch.float8_e4m3fn
    assert model.main._fp8_weight.device == device
    assert torch.equal(model.main._fp8_weight.cpu().float(), packed_weight.float())
    assert torch.equal(model.main._fp8_weight_scale.cpu(), scale.flatten())
    assert getattr(model.main, "weight", None) is None

    assert isinstance(model.refiner.quant_method, UnquantizedLinearMethod)
    assert not hasattr(model.refiner, "_fp8_weight")
    assert model.refiner.weight.dtype == torch.bfloat16
    assert torch.equal(model.refiner.weight.cpu(), weights["export_refiner.weight"])
    assert torch.equal(model.refiner.bias.cpu(), weights["export_refiner.bias"])
    x = torch.ones(16, 128, dtype=torch.bfloat16, device=device)
    expected = torch.nn.functional.linear(x, weights["export_refiner.weight"].to(device),
                                          weights["export_refiner.bias"].to(device))
    assert torch.equal(model.refiner(x)[0], expected)
    if device.type == "cuda":
        main_output = model.main(x)[0]
        assert main_output.shape == (16, 64)
        assert torch.isfinite(main_output).all()


def test_dense_checkpoint_still_uses_runtime_fp8(tmp_path, device):
    path = tmp_path / "dense.safetensors"
    save_file(_dense_weights(), str(path))

    model = _load([path], device, FP8Config())

    for layer in (model.main, model.refiner):
        assert isinstance(layer.quant_method, FP8QuantizeMethod)
        assert layer._fp8_weight.dtype == torch.float8_e4m3fn
        assert getattr(layer, "weight", None) is None


def test_packed_fp8_requires_matching_quant_config(tmp_path):
    path = tmp_path / "packed.safetensors"
    weights = _dense_weights()
    weights["export_main.weight"] = weights["export_main.weight"].to(torch.float8_e4m3fn)
    weights["export_main.weight_scale"] = torch.ones(64, dtype=torch.float32)
    save_file(weights, str(path))

    with pytest.raises(RuntimeError, match="not an FP8 linear"):
        _load([path], torch.device("cpu"), None)
