# SPDX-License-Identifier: Apache-2.0
"""CPU checks for Qwen-Image-2.1's conditioner contract and shared vision graph.

The source harness retains the native graph, normalization, and configuration
but substitutes ordinary linear/embedding layers for tensor-parallel execution.
These checks can run before installing FastVideo's CUDA dependencies. Released
weight parity through the production loader lives in the separate parity test.
"""

import ast
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.testing import assert_close

PARITY_SCOPE = "implementation_subcomponent"
REPO_ROOT = Path(__file__).resolve().parents[3]


class _RemoveFastVideoImports(ast.NodeTransformer):

    def visit_ImportFrom(self, node):
        return ast.copy_location(ast.Pass(), node) if node.module and node.module.startswith("fastvideo") else node


def _load_source(monkeypatch, name, relative_path, **namespace):
    module = ModuleType(name)
    namespace.setdefault("init_logger", lambda _name: SimpleNamespace(warning=lambda *_args: None))
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, name, module)
    path = REPO_ROOT / relative_path
    tree = _RemoveFastVideoImports().visit(ast.parse(path.read_text()))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def native(monkeypatch):
    class Backend(Enum):
        FLASH_ATTN = 1
        TORCH_SDPA = 2

    class Linear(nn.Linear):

        def __init__(self, input_size, output_size, bias=False, **_kwargs):
            super().__init__(input_size, output_size, bias=bias)

        def forward(self, inputs):
            return super().forward(inputs), None

    class Embedding(nn.Embedding):

        def __init__(self, num_embeddings, embedding_dim, **_kwargs):
            super().__init__(num_embeddings, embedding_dim)

    class Quantization:

        @classmethod
        def get_name(cls):
            return cls.__name__

    class FP8(Quantization):
        pass

    class NVFP4(Quantization):
        pass

    def weight_loader(parameter, tensor):
        if parameter.shape != tensor.shape:
            raise ValueError(f"Shape mismatch: {parameter.shape} != {tensor.shape}")
        with torch.no_grad():
            parameter.copy_(tensor)

    base = _load_source(monkeypatch, "qwen21_test_model_config", "fastvideo/configs/models/base.py")
    configs = _load_source(
        monkeypatch,
        "qwen21_test_encoder_config",
        "fastvideo/configs/models/encoders/base.py",
        ArchConfig=base.ArchConfig,
        ModelConfig=base.ModelConfig,
        QuantizationConfig=Quantization,
        AttentionBackendEnum=Backend,
    )
    h3_config = _load_source(
        monkeypatch,
        "qwen21_test_h3_config",
        "fastvideo/configs/models/encoders/minimax_h3_qwen3_vl.py",
        TextEncoderArchConfig=configs.TextEncoderArchConfig,
        TextEncoderConfig=configs.TextEncoderConfig,
    )
    config = _load_source(
        monkeypatch,
        "qwen21_test_config",
        "fastvideo/configs/models/encoders/qwen_image21.py",
        ModelConfig=base.ModelConfig,
        MiniMaxH3Qwen3VLArchConfig=h3_config.MiniMaxH3Qwen3VLArchConfig,
        MiniMaxH3Qwen3VLConfig=h3_config.MiniMaxH3Qwen3VLConfig,
        _OFFICIAL_ARCHITECTURES=h3_config._OFFICIAL_ARCHITECTURES,
        _VISION_CONFIG_MAPPING=h3_config._VISION_CONFIG_MAPPING,
    )
    base_encoder = _load_source(
        monkeypatch,
        "qwen21_test_base_encoder",
        "fastvideo/models/encoders/base.py",
        TextEncoderConfig=configs.TextEncoderConfig,
        ImageEncoderConfig=configs.ImageEncoderConfig,
        BaseEncoderOutput=configs.BaseEncoderOutput,
        AttentionBackendEnum=Backend,
    )
    logger = SimpleNamespace(debug=lambda *_args: None)
    custom_op = _load_source(
        monkeypatch,
        "qwen21_test_custom_op",
        "fastvideo/layers/custom_op.py",
        init_logger=lambda _name: logger,
    )
    norms = _load_source(
        monkeypatch,
        "qwen21_test_layernorm",
        "fastvideo/layers/layernorm.py",
        CustomOp=custom_op.CustomOp,
        current_platform=SimpleNamespace(is_cuda_alike=lambda: False),
    )
    h3 = _load_source(
        monkeypatch,
        "qwen21_test_h3",
        "fastvideo/models/encoders/minimax_h3_qwen3_vl.py",
        MiniMaxH3Qwen3VLConfig=h3_config.MiniMaxH3Qwen3VLConfig,
        TextEncoder=base_encoder.TextEncoder,
        get_tp_world_size=lambda: 1,
        ColumnParallelLinear=Linear,
        RowParallelLinear=Linear,
        VocabParallelEmbedding=Embedding,
        RMSNorm=norms.RMSNorm,
        QuantizationConfig=Quantization,
        MiniMaxH3SerializedFP8Config=FP8,
        MiniMaxH3SerializedNVFP4Config=NVFP4,
        default_weight_loader=weight_loader,
    )
    # The project's supported Torch implements GQA directly. Torch 2.3 on a
    # development Mac needs the mathematically equivalent explicit expansion.
    sdpa = F.scaled_dot_product_attention

    def compatible_sdpa(query, key, value, *, enable_gqa=False, **kwargs):
        if enable_gqa:
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        return sdpa(query, key, value, **kwargs)

    h3.F = SimpleNamespace(**{name: getattr(F, name) for name in dir(F) if not name.startswith("_")})
    h3.F.scaled_dot_product_attention = compatible_sdpa
    model = _load_source(
        monkeypatch,
        "qwen21_test_conditioner",
        "fastvideo/models/encoders/qwen_image21.py",
        QwenImage21Qwen3VLConfig=config.QwenImage21Qwen3VLConfig,
        MiniMaxH3Qwen3VLConditioner=h3.MiniMaxH3Qwen3VLConditioner,
        default_weight_loader=weight_loader,
    )
    return SimpleNamespace(config=config, model=model, h3=h3, h3_config=h3_config)


def _tiny_config(native):
    return native.config.QwenImage21Qwen3VLConfig(
        arch_config=native.config.QwenImage21Qwen3VLArchConfig(
            vocab_size=48,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            output_hidden_state_index=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            rope_scaling={"rope_type": "default", "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
            vision_hidden_size=16,
            vision_intermediate_size=24,
            vision_num_heads=2,
            vision_depth=3,
            vision_patch_size=2,
            vision_temporal_patch_size=1,
            vision_out_hidden_size=16,
            vision_num_position_embeddings=16,
            vision_deepstack_visual_indexes=(0, 1, 2),
            vision_start_token_id=40,
            vision_end_token_id=41,
            image_token_id=42,
            video_token_id=43,
        ))


def test_checkpoint_config_selects_full_decoder_and_processor(native):
    config = native.config.QwenImage21Qwen3VLConfig()
    config.update_model_arch({
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {
            "hidden_size": 4096,
            "num_hidden_layers": 36,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
        },
        "vision_config": {"out_hidden_size": 4096, "deepstack_visual_indexes": [8, 16, 24]},
    })
    assert config.architectures == ["QwenImage21Qwen3VLConditioner"]
    assert config.output_hidden_state_index == config.num_hidden_layers == 36
    assert config.num_hidden_layers_override is None
    assert config.require_processor and not config.tokenizer_kwargs["truncation"]
    assert config.tokenizer_kwargs["padding_side"] == "left"
    h3 = native.h3_config.MiniMaxH3Qwen3VLConfig()
    h3_arch = (h3.hidden_size, h3.num_hidden_layers_override, h3.output_hidden_state_index, h3.text_len)
    assert h3_arch == (5120, 50, 50, 1024)


def test_hidden_output_is_last_layer_before_loaded_final_norm(native):
    torch.manual_seed(13)
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    ids = torch.tensor([[1, 2, 3, 4]])
    captured = []
    hook = model.language_model.layers[-1].register_forward_hook(lambda _module, _args, output: captured.append(output))
    with torch.no_grad():
        model.language_model.norm.weight.fill_(23)
    result = model(ids)
    hook.remove()
    assert_close(result, captured[0], rtol=0, atol=0)
    assert not torch.allclose(result, model.language_model.norm(result))
    assert "language_model.norm.weight" in model.state_dict()


def test_padding_keeps_active_token_features(native):
    torch.manual_seed(14)
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    expected = model.encode_ids(torch.tensor([1, 2, 3, 4]))
    actual = model(torch.tensor([[0, 0, 1, 2, 3, 4]]), attention_mask=torch.tensor([[0, 0, 1, 1, 1, 1]]))
    assert_close(actual[0, 2:], expected, rtol=1e-5, atol=1e-6)


def test_separate_reference_vision_matches_packed_reference_graph(native):
    torch.manual_seed(15)
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    grid = torch.tensor([[1, 4, 4], [1, 4, 6]])
    pixels = torch.randn(40, 3 * 2 * 2)
    expected, expected_deepstack = model.visual(pixels, grid)
    actual, actual_deepstack = model._visual_features(pixels, grid)
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for actual_features, expected_features in zip(actual_deepstack, expected_deepstack, strict=True):
        assert_close(actual_features, expected_features, rtol=1e-5, atol=1e-6)
    ids = torch.tensor([[1, 40, *([42] * 4), 41, 2, 40, *([42] * 6), 41, 3]])
    positions = model._get_rope_index(ids, grid, None, None)
    assert positions.shape == (3, 1, ids.shape[1])
    # The second reference's spatial offsets follow the first image block.
    assert positions[:, 0, 9].tolist() == [7, 7, 7]
    result = model(ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(ids == 42).long())
    assert torch.isfinite(result).all()


def test_padded_image_batch_keeps_deepstack_features_with_each_prompt(native):
    torch.manual_seed(18)
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    first_ids = torch.tensor([1, 40, 42, 41, 2])
    second_ids = torch.tensor([3, 40, 42, 42, 41, 4, 5])
    pixels = torch.randn(12, 12)
    grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
    expected_first = model.encode_ids(first_ids, pixel_values=pixels[:4], image_grid_thw=grid[:1])
    expected_second = model.encode_ids(second_ids, pixel_values=pixels[4:], image_grid_thw=grid[1:])
    batch_ids = torch.stack([torch.cat([torch.zeros(2, dtype=torch.long), first_ids]), second_ids])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]])
    actual = model(batch_ids, attention_mask=mask, pixel_values=pixels, image_grid_thw=grid)
    assert_close(actual[0, 2:], expected_first, rtol=1e-5, atol=1e-6)
    assert_close(actual[1], expected_second, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[0, :2]) == 0


def test_weight_loading_is_complete_with_only_lm_head_excluded(native):
    torch.manual_seed(16)
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    state = {name: torch.randn_like(value) for name, value in model.state_dict().items()}
    source = [("model." + name, value) for name, value in state.items()]
    source.append(("lm_head.weight", torch.empty(48, 16)))
    loaded = model.load_weights(source)
    assert loaded == set(dict(model.named_parameters()))
    for name, value in state.items():
        assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Unexpected"):
        model.load_weights([("model.language_model.layers.99.mlp.up_proj.weight", torch.empty(1))])


@pytest.mark.parametrize("failure", ["patch_count", "missing_images", "wrong_tokens", "empty_mask"])
def test_bad_conditioning_inputs_fail_early(native, failure):
    model = native.model.QwenImage21Qwen3VLConditioner(_tiny_config(native))
    ids = torch.tensor([[1, 40, 42, 41, 2]])
    inputs = {"input_ids": ids, "pixel_values": torch.randn(4, 12), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    if failure == "patch_count":
        inputs["pixel_values"] = torch.randn(3, 12)
    elif failure == "missing_images":
        inputs.pop("pixel_values")
        inputs.pop("image_grid_thw")
    elif failure == "wrong_tokens":
        inputs["mm_token_type_ids"] = torch.zeros_like(ids)
    else:
        inputs["attention_mask"] = torch.zeros_like(ids)
    with pytest.raises(ValueError):
        model(**inputs)
