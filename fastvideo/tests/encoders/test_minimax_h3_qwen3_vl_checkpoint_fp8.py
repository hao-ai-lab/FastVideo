# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os

import pytest
import torch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29514")

import fastvideo.models.encoders.minimax_h3_checkpoint_fp8 as h3_fp8
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.layers.linear import ColumnParallelLinear, UnquantizedLinearMethod
from fastvideo.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod, VocabParallelEmbedding
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.encoders.minimax_h3_checkpoint_fp8 import (
    MiniMaxH3SerializedFP8Config,
    MiniMaxH3SerializedFP8LinearMethod,
)
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.loader.text_encoder_quantization import (
    _configure_text_encoder_quantization,
    _process_quantized_text_encoder_weights,
    _read_text_encoder_checkpoint_quantization_config,
)


def _checkpoint_quantization_config(**overrides) -> dict:
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["model.visual", "lm_head"],
    }
    config.update(overrides)
    return config


def test_h3_accepts_only_the_serialized_blockwise_checkpoint_contract() -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    assert config.weight_block_size == (128, 128)
    assert config.get_supported_act_dtypes() == [torch.bfloat16]

    with pytest.raises(ValueError, match=r"weight_block_size=\[128, 128\]"):
        MiniMaxH3SerializedFP8Config.from_config(
            _checkpoint_quantization_config(weight_block_size=[1, 128]))
    with pytest.raises(ValueError, match="dynamic activation"):
        MiniMaxH3SerializedFP8Config.from_config(
            _checkpoint_quantization_config(activation_scheme="static"))
    with pytest.raises(ValueError, match="vision stack"):
        MiniMaxH3SerializedFP8Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["lm_head"]))
    with pytest.raises(ValueError, match="partially quantized language"):
        MiniMaxH3SerializedFP8Config.from_config(
            _checkpoint_quantization_config(modules_to_not_convert=["model.visual", "language_model.layers.3"]))


def test_serialized_fp8_allocates_checkpoint_weight_and_scale_without_requantization(distributed_setup) -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    layer = ColumnParallelLinear(
        input_size=128,
        output_size=256,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.layers.0.self_attn.q_proj",
    )

    assert isinstance(layer.quant_method, MiniMaxH3SerializedFP8LinearMethod)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight.shape == (256, 128)
    assert layer.weight_scale_inv.dtype == torch.float32
    assert layer.weight_scale_inv.shape == (2, 1)

    layer.weight.data.zero_()
    layer.weight_scale_inv.data.fill_(0.25)
    weight_pointer = layer.weight.data_ptr()
    scale_pointer = layer.weight_scale_inv.data_ptr()
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.weight.data_ptr() == weight_pointer
    assert layer.weight_scale_inv.data_ptr() == scale_pointer
    assert not hasattr(layer, "_fp8_weight")


def test_serialized_fp8_quantizes_only_language_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    visual_linear = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.visual.blocks.0.attn.proj",
    )
    embedding = VocabParallelEmbedding(
        num_embeddings=128,
        embedding_dim=128,
        org_num_embeddings=128,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.embed_tokens",
    )

    assert isinstance(visual_linear.quant_method, UnquantizedLinearMethod)
    assert visual_linear.weight.dtype == torch.get_default_dtype()
    assert isinstance(embedding.quant_method, UnquantizedEmbeddingMethod)
    assert embedding.weight.dtype == torch.get_default_dtype()


def test_serialized_fp8_cpu_execution_fails_closed(distributed_setup) -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    layer = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.layers.0.mlp.up_proj",
    )
    layer.weight.data.zero_()
    layer.weight_scale_inv.data.fill_(1.0)
    assert isinstance(layer.quant_method, MiniMaxH3SerializedFP8LinearMethod)
    layer.quant_method.process_weights_after_loading(layer)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.zeros(2, 128, dtype=torch.bfloat16))


def test_runtime_preflight_reports_capability_and_missing_dependencies(monkeypatch) -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    with pytest.raises(RuntimeError, match="sm100 or newer"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))

    def missing_quantizer() -> None:
        raise RuntimeError("SGLang-compatible Triton quantizer is missing")

    monkeypatch.setattr(h3_fp8, "_require_sglang_per_token_group_fp8_quantization", missing_quantizer)
    with pytest.raises(RuntimeError, match="Triton quantizer is missing"):
        config.validate_runtime(torch.device("cuda"))

    monkeypatch.setattr(h3_fp8, "_require_sglang_per_token_group_fp8_quantization", lambda: None)

    def missing_flashinfer():
        raise RuntimeError("FlashInfer groupwise GEMM is missing")

    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_fp8_gemm", missing_flashinfer)
    with pytest.raises(RuntimeError, match="FlashInfer groupwise GEMM is missing"):
        config.validate_runtime(torch.device("cuda"))


def test_loader_detects_and_capability_gates_checkpoint_metadata(tmp_path) -> None:
    checkpoint_config = _checkpoint_quantization_config()
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": checkpoint_config}),
        encoding="utf-8",
    )

    assert _read_text_encoder_checkpoint_quantization_config(str(tmp_path)) == checkpoint_config
    model_config = MiniMaxH3Qwen3VLConfig()
    quant_config = _configure_text_encoder_quantization(
        model_config,
        MiniMaxH3Qwen3VLConditioner,
        str(tmp_path),
    )
    assert isinstance(quant_config, MiniMaxH3SerializedFP8Config)
    assert model_config.quant_config is quant_config

    unsupported_config = MiniMaxH3Qwen3VLConfig()
    with pytest.raises(ValueError, match="does not support serialized 'fp8'"):
        _configure_text_encoder_quantization(
            unsupported_config,
            TextEncoder,
            str(tmp_path),
        )


def test_loader_leaves_bf16_checkpoint_path_unchanged(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3VLModel"]}), encoding="utf-8")
    model_config = MiniMaxH3Qwen3VLConfig()

    quant_config = _configure_text_encoder_quantization(
        model_config,
        MiniMaxH3Qwen3VLConditioner,
        str(tmp_path),
    )

    assert quant_config is None
    assert model_config.quant_config is None


def test_post_load_processing_visits_only_serialized_fp8_linears(distributed_setup) -> None:
    config = MiniMaxH3SerializedFP8Config.from_config(_checkpoint_quantization_config())
    quantized = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        quant_config=config,
        prefix="minimax_h3_qwen3_vl.language_model.layers.0.self_attn.q_proj",
    )
    plain = ColumnParallelLinear(
        input_size=128,
        output_size=128,
        bias=False,
        prefix="plain",
    )
    quantized.weight.data.zero_()
    quantized.weight_scale_inv.data.fill_(1.0)
    model = torch.nn.ModuleList([quantized, plain])

    assert _process_quantized_text_encoder_weights(model, torch.device("cpu")) == 1
    assert quantized.weight.device.type == "cpu"
    assert plain.weight.device.type == "cpu"


def test_flashinfer_groupwise_path_pins_output_dtype_and_trtllm_scale_layout(monkeypatch) -> None:
    input_tensor = torch.zeros(2, 256, dtype=torch.bfloat16)
    weight = torch.zeros(128, 256, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(1, 2, dtype=torch.float32)
    quantized_input = torch.zeros_like(input_tensor, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty(2, 2, dtype=torch.float32).t()
    input_scale.fill_(1.0)
    receipt: dict[str, object] = {}

    def fake_quantize(
        value: torch.Tensor,
        group_size: int,
        *,
        column_major_scales: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert value.data_ptr() == input_tensor.data_ptr()
        assert value.shape == input_tensor.shape
        assert group_size == 128
        assert column_major_scales is True
        return quantized_input, input_scale

    def fake_gemm(
        activation: torch.Tensor,
        checkpoint_weight: torch.Tensor,
        activation_scale: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype,
        backend: str,
    ) -> torch.Tensor:
        receipt.update(
            activation=activation,
            checkpoint_weight=checkpoint_weight,
            activation_scale=activation_scale,
            checkpoint_scale=checkpoint_scale,
            out_dtype=out_dtype,
            backend=backend,
        )
        return torch.zeros(activation.shape[0], checkpoint_weight.shape[0], dtype=out_dtype)

    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_backend", lambda device: "trtllm")
    monkeypatch.setattr(h3_fp8, "_sglang_per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_fp8_gemm", lambda: fake_gemm)

    previous_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        output = h3_fp8._flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
            input_tensor,
            weight,
            (128, 128),
            weight_scale,
        )
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert output.dtype == torch.bfloat16
    assert receipt["out_dtype"] == torch.bfloat16
    assert receipt["backend"] == "trtllm"
    assert receipt["activation"] is quantized_input
    assert receipt["checkpoint_weight"] is weight
    assert receipt["checkpoint_scale"] is weight_scale
    assert receipt["activation_scale"] is input_scale


def test_flashinfer_groupwise_cutlass_path_pads_m_to_a_multiple_of_four(monkeypatch) -> None:
    """FlashInfer documents that ``m`` must be padded to a multiple of 4, and
    the CUTLASS groupwise module (the sm12x route) rejects unaligned ``m`` at
    dispatch with ``cutlass gemm.can_implement failed``. m=559 is this PR's
    own benchmark prompt length. The route must pad the quantized activation
    rows and per-token scales, pin the MN scale layout, and slice the padded
    rows off the output."""
    m, k, n = 559, 256, 128
    padded_m = 560
    input_tensor = torch.zeros(m, k, dtype=torch.bfloat16)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(n // 128, k // 128, dtype=torch.float32)
    quantized_input = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
    input_scale = torch.ones(m, k // 128, dtype=torch.float32)
    receipt: dict[str, object] = {}

    def fake_quantize(
        value: torch.Tensor,
        group_size: int,
        *,
        column_major_scales: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert value.data_ptr() == input_tensor.data_ptr()
        assert group_size == 128
        assert column_major_scales is False
        return quantized_input, input_scale

    def fake_gemm(
        activation: torch.Tensor,
        checkpoint_weight: torch.Tensor,
        activation_scale: torch.Tensor,
        checkpoint_scale: torch.Tensor,
        *,
        out_dtype: torch.dtype,
        backend: str,
        scale_major_mode: str,
    ) -> torch.Tensor:
        assert activation.shape[0] % 4 == 0, "cutlass GEMM requires m padded to a multiple of 4"
        receipt.update(
            activation=activation,
            activation_scale=activation_scale,
            checkpoint_scale=checkpoint_scale,
            backend=backend,
            scale_major_mode=scale_major_mode,
        )
        # Tag each row with a bf16-exact value (integers above 256 are not
        # exactly representable in bf16) so the caller-side slice of the
        # first m rows is observable.
        rows = torch.arange(activation.shape[0], dtype=torch.float32) % 256
        return rows.unsqueeze(1).expand(activation.shape[0], checkpoint_weight.shape[0]).contiguous().to(out_dtype)

    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_backend", lambda device: "cutlass")
    monkeypatch.setattr(h3_fp8, "_sglang_per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_fp8_gemm", lambda: fake_gemm)

    output = h3_fp8._flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
        input_tensor,
        weight,
        (128, 128),
        weight_scale,
    )

    activation = receipt["activation"]
    assert activation.shape == (padded_m, k)
    assert torch.equal(activation[:m].view(torch.uint8), quantized_input.view(torch.uint8))
    assert not activation[m:].view(torch.uint8).any(), "padded activation rows must be zero"

    activation_scale = receipt["activation_scale"]
    assert activation_scale.shape == (k // 128, padded_m)
    assert torch.equal(activation_scale[:, :m], input_scale.transpose(-1, -2))
    assert not activation_scale[:, m:].any(), "padded scale columns must be zero"

    assert receipt["backend"] == "cutlass"
    assert receipt["scale_major_mode"] == "MN"
    assert receipt["checkpoint_scale"].shape == (k // 128, n // 128)

    # The padded rows never reach the caller.
    assert output.shape == (m, n)
    assert output.dtype == torch.bfloat16
    assert torch.equal(output.float()[:, 0], torch.arange(m, dtype=torch.float32) % 256)


def test_flashinfer_groupwise_cutlass_path_leaves_aligned_m_unpadded(monkeypatch) -> None:
    """Aligned token counts must not pay a padding copy on the cutlass route."""
    m, k, n = 560, 256, 128
    input_tensor = torch.zeros(m, k, dtype=torch.bfloat16)
    weight = torch.zeros(n, k, dtype=torch.float8_e4m3fn)
    weight_scale = torch.ones(k // 128, n // 128, dtype=torch.float32)
    quantized_input = torch.zeros(m, k, dtype=torch.float8_e4m3fn)
    input_scale = torch.ones(k // 128, m, dtype=torch.float32)
    receipt: dict[str, object] = {}

    def fake_quantize(value, group_size, *, column_major_scales):
        return quantized_input, input_scale

    def fake_gemm(activation, checkpoint_weight, activation_scale, checkpoint_scale, *,
                  out_dtype, backend, scale_major_mode):
        receipt.update(activation=activation, activation_scale=activation_scale)
        return torch.zeros(activation.shape[0], checkpoint_weight.shape[0], dtype=out_dtype)

    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_backend", lambda device: "cutlass")
    monkeypatch.setattr(h3_fp8, "_sglang_per_token_group_quant_fp8", fake_quantize)
    monkeypatch.setattr(h3_fp8, "_get_flashinfer_groupwise_fp8_gemm", lambda: fake_gemm)

    output = h3_fp8._flashinfer_gemm_w8a8_block_fp8_linear_with_fallback(
        input_tensor,
        weight,
        (128, 128),
        weight_scale,
    )

    assert receipt["activation"] is quantized_input
    assert receipt["activation_scale"] is input_scale
    assert output.shape == (m, n)
