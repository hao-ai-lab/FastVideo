# SPDX-License-Identifier: Apache-2.0
"""Numerical parity for Dense LingBot-Video T2V Qwen3-VL text conditioning."""

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch.testing import assert_close
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from fastvideo.configs.models.encoders.lingbot_video import LingBotVideoQwen3VLTextConfig
from fastvideo.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from fastvideo.models.encoders.lingbot_video import (
    LingBotVideoQwen3VLAttention,
    LingBotVideoQwen3VLDecoderLayer,
    LingBotVideoQwen3VLTextModel,
)
from fastvideo.models.loader.component_loader import TextEncoderLoader
from fastvideo.models.loader.utils import set_default_torch_dtype
from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    OFFICIAL_DENSE,
    download_components,
)


PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _require_gpu_test() -> torch.device:
    """Keep heavyweight parity opt-in and off non-Slurm development nodes."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated GPU")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the released Qwen3-VL encoder")
    return torch.device("cuda")


def _load_official(device: torch.device, checkpoint: Path) -> Qwen3VLForConditionalGeneration:
    """Load the released full Qwen3-VL wrapper used by the official pipeline."""
    return Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint / "text_encoder",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()


def _load_native(device: torch.device, checkpoint: Path) -> LingBotVideoQwen3VLTextModel:
    """Load converted fused tensors through FastVideo's production text loader."""
    args = SimpleNamespace(
        text_encoder_cpu_offload=True,
        override_text_encoder_quant=None,
        override_text_encoder_safetensors=None,
        pin_cpu_memory=False,
    )
    model = TextEncoderLoader().load_model(
        str(checkpoint / "text_encoder"),
        LingBotVideoQwen3VLTextConfig(),
        device,
        args,
        dtype="bf16",
        use_text_encoder_override=True,
    )
    return model.eval()


def _capture_intermediates(model: torch.nn.Module) -> tuple[dict[str, torch.Tensor], list[Any]]:
    """Capture shared Qwen3 checkpoints and materialize native deferred residuals."""
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    suffixes = (
        "embed_tokens",
        "input_layernorm",
        "q_norm",
        "k_norm",
        "o_proj",
        "post_attention_layernorm",
        "down_proj",
        "norm",
    )

    def save(name: str):
        def hook(_module, _inputs, output):
            if name.count(".") == 1 and name.startswith("layers.") and isinstance(output, tuple):
                tensor = output[0] if output[1] is None else output[0] + output[1]
            else:
                tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                captured[name] = tensor.detach().float().cpu()

        return hook

    for name, module in model.named_modules():
        is_layer = name.startswith("layers.") and name.count(".") == 1
        if is_layer or name == "norm" or name.endswith(suffixes):
            handles.append(module.register_forward_hook(save(name)))
    return captured, handles


def _report_first_intermediate_drift(official: dict[str, torch.Tensor], native: dict[str, torch.Tensor]) -> None:
    """Print the earliest shared module output with material numerical drift."""
    for name, expected in official.items():
        actual = native.get(name)
        if actual is None or actual.shape != expected.shape:
            continue
        drift = (actual - expected).abs()
        if drift.max().item() > 1e-3:
            print(
                f"first_intermediate_drift={name} max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f}"
            )
            return
    print("first_intermediate_drift=not_found")


def test_lingbot_video_text_encoder_constructs_final_layers_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a tiny meta encoder without invoking either replace-after-build constructor."""
    import fastvideo.layers.linear as linear
    import fastvideo.layers.vocab_parallel_embedding as embedding
    import fastvideo.models.encoders.qwen3 as qwen3

    monkeypatch.setattr(linear, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(embedding, "get_tp_rank", lambda: 0)
    monkeypatch.setattr(embedding, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(qwen3, "get_tp_world_size", lambda: 1)

    def reject_base_constructor(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("replace-after-build base constructor was invoked")

    attention_types: list[type[torch.nn.Module]] = []
    original_attention_init = qwen3.Qwen3Attention.__init__

    def record_attention_type(module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        attention_types.append(type(module))
        original_attention_init(module, *args, **kwargs)

    monkeypatch.setattr(qwen3.Qwen3ForCausalLM, "__init__", reject_base_constructor)
    monkeypatch.setattr(qwen3.Qwen3DecoderLayer, "__init__", reject_base_constructor)
    monkeypatch.setattr(qwen3.Qwen3Attention, "__init__", record_attention_type)
    config = LingBotVideoQwen3VLTextConfig()
    for name, value in {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 32,
    }.items():
        setattr(config.arch_config, name, value)
    with set_default_torch_dtype(torch.bfloat16), torch.device("meta"):
        model = LingBotVideoQwen3VLTextModel(config)

    layer_parameter_names = {
        "self_attn.qkv_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    }
    expected_names = {"embed_tokens.weight", "norm.weight"} | {
        f"layers.{index}.{name}" for index in range(2) for name in layer_parameter_names
    }
    assert set(model.state_dict()) == expected_names
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(isinstance(layer, LingBotVideoQwen3VLDecoderLayer) for layer in model.layers)
    assert all(isinstance(layer.self_attn, LingBotVideoQwen3VLAttention) for layer in model.layers)
    assert attention_types == [LingBotVideoQwen3VLAttention, LingBotVideoQwen3VLAttention]


def test_lingbot_video_dense_text_encoder_parity() -> None:
    """Compare official and native final hidden states with identical T2V tokens."""
    device = _require_gpu_test()
    official_checkpoint = download_components(OFFICIAL_DENSE, "text_encoder", "processor")
    fastvideo_checkpoint = download_components(FASTVIDEO_DENSE, "text_encoder")
    init_distributed_environment(world_size=1, rank=0, local_rank=0)
    initialize_model_parallel(tensor_model_parallel_size=1, sequence_model_parallel_size=1)
    processor = AutoProcessor.from_pretrained(official_checkpoint / "processor")
    inputs = processor(
        text=[PROMPT_TEMPLATE.format("A red fox runs through fresh snow at sunrise.")],
        images=None,
        videos=None,
        do_resize=False,
        truncation=True,
        max_length=37698,
        padding="longest",
        return_tensors="pt",
    ).to(device)
    official = _load_official(device, official_checkpoint)
    native = _load_native(device, fastvideo_checkpoint)
    official_text_model = official.model.language_model
    official_intermediates, official_handles = _capture_intermediates(official_text_model)
    native_intermediates, native_handles = _capture_intermediates(native)
    with torch.no_grad():
        expected = official(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1]
        native_outputs = native(
            input_ids=inputs["input_ids"].cpu(),
            attention_mask=inputs["attention_mask"].cpu(),
            output_hidden_states=True,
        )
        actual = native_outputs.hidden_states[-1]
    for handle in official_handles + native_handles:
        handle.remove()
    drift = (actual.float() - expected.float()).abs()
    print(f"max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f}")
    _report_first_intermediate_drift(official_intermediates, native_intermediates)
    assert drift.mean().item() <= 5e-3
    assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)
