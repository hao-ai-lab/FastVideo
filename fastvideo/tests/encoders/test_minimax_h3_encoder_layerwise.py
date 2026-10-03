# SPDX-License-Identifier: Apache-2.0
"""Text-only H3 encoder streaming parity and placement contracts."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fastvideo.hooks.hooks import ModuleHookManager
from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLArchConfig, MiniMaxH3Qwen3VLConfig
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import MiniMaxH3SerializedNVFP4Config
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.loader.text_encoder_quantization import _process_quantized_text_encoder_weights
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for encoder streaming")
@pytest.mark.parametrize("quantized", [False, True])
def test_streamed_encoder_matches_resident_and_releases_layers(distributed_setup, monkeypatch, quantized):
    # DiT residency must not accidentally keep encoder layers resident too.
    monkeypatch.setenv("FASTVIDEO_LAYERWISE_RESIDENT_BLOCKS", "6")
    config = MiniMaxH3Qwen3VLConfig()
    config.arch_config = MiniMaxH3Qwen3VLArchConfig(
        vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=3, num_hidden_layers_override=2, output_hidden_state_index=2,
        num_attention_heads=1, num_key_value_heads=1, head_dim=128,
        rope_scaling={"mrope_interleaved": True, "mrope_section": [32, 16, 16], "rope_type": "default"},
        vision_depth=1, vision_hidden_size=64, vision_intermediate_size=128,
        vision_num_heads=1, vision_deepstack_visual_indexes=(), vision_out_hidden_size=128,
    )
    config.quant_config = MiniMaxH3SerializedNVFP4Config() if quantized else None
    torch.manual_seed(81)
    model = MiniMaxH3Qwen3VLConditioner(config).to(dtype=torch.bfloat16).eval()
    for name, parameter in model.named_parameters():
        if name.endswith("weight_packed"):
            parameter.data.random_(0, 256)
        elif name.endswith("weight_scale"):
            parameter.data.fill_(0x38)
        elif name.endswith("weight_global_scale"):
            parameter.data.fill_(2.0)
        else:
            parameter.data.normal_(std=0.02)
    if quantized:
        _process_quantized_text_encoder_weights(model, torch.device("cuda"))
        linear = model.language_model.layers[0].self_attn.q_proj.to("cuda")
        x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
        expected_linear = linear(x)[0]
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("Unexpected device scalar read")):
            actual_linear = linear(x)[0]
        torch.testing.assert_close(actual_linear, expected_linear, rtol=0, atol=0)
    ids = torch.tensor([1, 7, 4, 21, 5, 31, 18], device="cuda")
    model.to("cuda")
    expected = model.encode_ids(ids)
    assert torch.isfinite(expected).all()
    model.to("cpu")
    model.prepare_layerwise_offload(torch.device("cuda"))
    model.prepare_layerwise_offload(torch.device("cuda"))  # repeated setup is harmless
    assert model.language_model.embed_tokens.weight.device.type == "cpu"
    assert next(model.visual.parameters()).device.type == "cpu"
    for _ in range(2):
        actual = model.encode_ids(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for layer in model.language_model.layers:
            assert all(parameter.numel() == 0 for parameter in layer.parameters())
            manager = ModuleHookManager.get_from(layer)
            assert manager is not None
            assert not manager.forward_hooks["LayerwiseOffloadHook"].state.gpu_named_parameters
    with pytest.raises(ValueError, match="text-only"):
        model.encode_ids(ids, pixel_values=torch.zeros(1, device="cuda"),
                         image_grid_thw=torch.ones(1, 3, device="cuda", dtype=torch.int64))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pipeline_does_not_move_streamed_encoder_whole(device):
    module = SimpleNamespace(_h3_encoder_layerwise_device=torch.device("cuda"))
    module.to = lambda *_: pytest.fail("Whole encoder move defeats streaming")
    assert MiniMaxH3Pipeline._move_module(None, module, device)


def test_conditioning_stage_keeps_streamed_encoder_placement(monkeypatch):
    import fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning as conditioning

    module = SimpleNamespace(_h3_encoder_layerwise_device=torch.device("cuda"))
    module.parameters = lambda: iter([torch.empty(1)])
    module.to = lambda *_: pytest.fail("Conditioning must retain layerwise placement")
    stage = MiniMaxH3ConditioningStage.__new__(MiniMaxH3ConditioningStage)
    stage.conditioner, stage.ref2va = module, False
    stage._encode_fl2va = lambda *_: (torch.zeros(1, 2, 128), torch.zeros(2, dtype=torch.int32))
    monkeypatch.setattr(conditioning, "get_local_torch_device", lambda: torch.device("cpu"))
    batch = ForwardBatch(data_type="video", prompt="streaming parity")
    output = stage.forward(batch, SimpleNamespace(text_encoder_cpu_offload=True))
    assert output.prompt_embeds[0].shape == (1, 2, 128)
