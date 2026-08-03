# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for the MiniMax H3 Qwen3-VL conditioner."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from fastvideo.configs.models.encoders import BaseEncoderOutput, MiniMaxH3Qwen3VLArchConfig, MiniMaxH3Qwen3VLConfig
from fastvideo.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConditioner
from fastvideo.models.registry import ModelRegistry
from fastvideo.pipelines.basic.minimax_h3.stages import _module_lifecycle
from fastvideo.pipelines.basic.minimax_h3.stages.conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_TEXT_ENCODER_LAYER
from fastvideo.pipelines.basic.minimax_h3.types import get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


@pytest.fixture(autouse=True)
def _keep_conditioner_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_module_lifecycle, "get_local_torch_device", lambda: torch.device("cpu"))


class _FakeQwen3VLModel(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 51,
        hidden_state_count: int = 52,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros((), dtype=dtype))
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=num_hidden_layers))
        self.hidden_state_count = hidden_state_count
        self.calls: list[dict[str, Any]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.marker.dtype

    def forward(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        sequence_length = kwargs["input_ids"].shape[1]
        hidden_states = tuple(
            torch.full((1, sequence_length, 4), float(index), dtype=torch.float32)
            for index in range(self.hidden_state_count)
        )
        return SimpleNamespace(
            last_hidden_state=hidden_states[-1],
            hidden_states=hidden_states,
            attentions=None,
        )


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.labels = {"<Picture 1>: ": [101], "<Picture 2>: ": [102, 103]}
        self.special_tokens = {
            "<|vision_start|>": 201,
            "<|image_pad|>": 202,
            "<|vision_end|>": 203,
        }

    def __call__(self, text: str, add_special_tokens: bool) -> dict[str, list[int]]:
        self.calls.append((text, add_special_tokens))
        return {"input_ids": self.labels.get(text, [301, 302])}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_tokens[token]


class _FakeImageProcessor:
    merge_size = 2

    def __init__(self) -> None:
        self.calls: list[tuple[list[Any], str]] = []

    def __call__(self, images: list[Any], return_tensors: str) -> dict[str, torch.Tensor]:
        self.calls.append((images, return_tensors))
        return {
            "pixel_values": torch.arange(12, dtype=torch.float64).reshape(2, 2, 3),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long),
        }


class _FakeProcessor:
    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()
        self.mm_calls: list[list[list[int]]] = []

    def create_mm_token_type_ids(self, token_ids: list[list[int]]) -> list[list[int]]:
        self.mm_calls.append(token_ids)
        return [[int(token == 202) for token in token_ids[0]]]


class _LegacyFakeProcessor:
    """Models Transformers releases before ProcessorMixin gained modality IDs."""

    def __init__(self) -> None:
        self.image_processor = _FakeImageProcessor()
        self.image_token_id = 202
        self.video_token_id = 204


def _conditioner(
    num_hidden_layers: int = 51,
    hidden_state_count: int = 52,
    dtype: torch.dtype = torch.float32,
) -> tuple[MiniMaxH3Qwen3VLConditioner, _FakeQwen3VLModel]:
    hf_model = _FakeQwen3VLModel(
        num_hidden_layers=num_hidden_layers,
        hidden_state_count=hidden_state_count,
        dtype=dtype,
    )
    conditioner = MiniMaxH3Qwen3VLConditioner(MiniMaxH3Qwen3VLConfig(), hf_model=hf_model)
    return conditioner, hf_model


def _run_stage(
    conditioner: MiniMaxH3Qwen3VLConditioner,
    tokenizer: _FakeTokenizer,
    processor: Any,
    prompt: Any = "prompt",
    images: list[Any] | None = None,
) -> tuple[ForwardBatch, Any]:
    batch = ForwardBatch(data_type="video", prompt=prompt)
    state = get_minimax_h3_state(batch)
    state.keyframes = [] if images is None else images
    stage = MiniMaxH3ConditioningStage(conditioner, tokenizer, processor)
    stage(batch, SimpleNamespace(text_encoder_cpu_offload=False))
    return batch, state


def test_config_maps_upstream_architecture_to_fastvideo_adapter() -> None:
    arch_config = MiniMaxH3Qwen3VLArchConfig()
    config = MiniMaxH3Qwen3VLConfig()

    assert arch_config.architectures == ["MiniMaxH3Qwen3VLConditioner"]
    assert MINIMAX_H3_TEXT_ENCODER_LAYER == 50
    assert arch_config.tokenizer_kwargs == {"add_special_tokens": False}
    assert MiniMaxH3Qwen3VLConditioner.supports_hf_from_pretrained

    config.update_model_arch({"architectures": ["Qwen3VLForConditionalGeneration"]})
    assert config.architectures == ["MiniMaxH3Qwen3VLConditioner"]
    model_cls, architecture = ModelRegistry.resolve_model_cls(config.architectures)
    assert model_cls is MiniMaxH3Qwen3VLConditioner
    assert architecture == "MiniMaxH3Qwen3VLConditioner"


def test_hf_loader_strictly_loads_full_checkpoint_then_keeps_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    hf_model = _FakeQwen3VLModel()
    calls: list[tuple[str, dict[str, Any]]] = []

    class _FakeQwen3VLFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: Any) -> tuple[Any, dict[str, list[str]]]:
            calls.append((model_path, kwargs))
            return SimpleNamespace(model=hf_model), {}

    monkeypatch.setattr(transformers, "Qwen3VLForConditionalGeneration", _FakeQwen3VLFactory, raising=False)
    config = MiniMaxH3Qwen3VLConfig()

    conditioner = MiniMaxH3Qwen3VLConditioner.from_pretrained_local(
        "/tmp/fake-minimax-h3",
        config,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert isinstance(conditioner, MiniMaxH3Qwen3VLConditioner)
    assert conditioner.hf_model is hf_model
    assert conditioner.training is False
    assert hf_model.training is False
    assert calls == [(
        "/tmp/fake-minimax-h3",
        {
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "output_loading_info": True,
        },
    )]


def test_hf_loader_rejects_partial_checkpoint_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    class _FakeQwen3VLFactory:
        @classmethod
        def from_pretrained(cls, model_path: str, **kwargs: Any) -> tuple[Any, dict[str, list[str]]]:
            return SimpleNamespace(model=_FakeQwen3VLModel()), {"missing_keys": ["model.language_model.norm.weight"]}

    monkeypatch.setattr(transformers, "Qwen3VLForConditionalGeneration", _FakeQwen3VLFactory, raising=False)

    with pytest.raises(RuntimeError, match="did not load strictly"):
        MiniMaxH3Qwen3VLConditioner.from_pretrained_local(
            "/tmp/fake-minimax-h3",
            MiniMaxH3Qwen3VLConfig(),
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )


def test_component_forward_returns_fastvideo_encoder_output() -> None:
    conditioner, hf_model = _conditioner()
    input_ids = torch.tensor([[1, 2]])
    attention_mask = torch.ones_like(input_ids)

    output = conditioner(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    assert isinstance(output, BaseEncoderOutput)
    assert output.hidden_states is not None
    assert len(output.hidden_states) == 52
    assert output.attention_mask is attention_mask
    call = hf_model.calls[0]
    assert call["input_ids"] is input_ids
    assert call["position_ids"] is None
    assert call["inputs_embeds"] is None
    assert call["output_hidden_states"] is True
    assert call["return_dict"] is True
    assert call["use_cache"] is False


def test_text_only_presentation_is_owned_by_conditioning_stage() -> None:
    conditioner, hf_model = _conditioner()
    tokenizer = _FakeTokenizer()
    processor = _FakeProcessor()

    batch, state = _run_stage(conditioner, tokenizer, processor, prompt="  verbatim prompt  ")

    assert tokenizer.calls == [("  verbatim prompt  ", False)]
    assert processor.image_processor.calls == []
    assert processor.mm_calls == [[[301, 302]]]
    assert torch.equal(state.text_token_tags, torch.ones(2, dtype=torch.long))
    assert state.text_token_tags.device.type == "cpu"
    assert state.prompt_embeds.dtype == torch.float32
    assert torch.equal(state.prompt_embeds, torch.full((1, 2, 4), 50.0))
    assert len(batch.prompt_embeds) == 1
    assert batch.prompt_embeds[0] is state.prompt_embeds

    call = hf_model.calls[0]
    assert torch.equal(call["input_ids"], torch.tensor([[301, 302]]))
    assert torch.equal(call["attention_mask"], torch.ones(1, 2, dtype=torch.long))
    assert torch.equal(call["mm_token_type_ids"], torch.zeros(1, 2, dtype=torch.long))
    assert call["pixel_values"] is None
    assert call["image_grid_thw"] is None
    assert call["use_cache"] is False
    assert call["output_hidden_states"] is True


def test_keyframes_build_exact_picture_and_vision_blocks() -> None:
    conditioner, hf_model = _conditioner()
    tokenizer = _FakeTokenizer()
    processor = _FakeProcessor()
    images = [object(), object()]

    _, state = _run_stage(conditioner, tokenizer, processor, images=images)

    expected_ids = [
        101,
        201,
        202,
        202,
        202,
        202,
        203,
        102,
        103,
        201,
        202,
        202,
        203,
        301,
        302,
    ]
    expected_tags = [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]

    assert tokenizer.calls == [("<Picture 1>: ", False), ("<Picture 2>: ", False), ("prompt", False)]
    assert processor.image_processor.calls == [(images, "pt")]
    assert processor.mm_calls == [[expected_ids]]
    assert torch.equal(state.text_token_tags, torch.tensor(expected_tags, dtype=torch.long))
    assert state.prompt_embeds.shape == (1, len(expected_ids), 4)

    call = hf_model.calls[0]
    assert torch.equal(call["input_ids"], torch.tensor([expected_ids]))
    assert call["pixel_values"].dtype == torch.float32
    assert torch.equal(call["pixel_values"], torch.arange(12, dtype=torch.float32).reshape(2, 2, 3))
    assert torch.equal(call["image_grid_thw"], torch.tensor([[1, 4, 4], [1, 2, 4]]))


def test_legacy_processor_builds_modality_ids_without_new_transformers_helper() -> None:
    conditioner, hf_model = _conditioner()
    processor = _LegacyFakeProcessor()

    _run_stage(conditioner, _FakeTokenizer(), processor, images=[object(), object()])

    expected_ids = [101, 201, 202, 202, 202, 202, 203, 102, 103, 201, 202, 202, 203, 301, 302]
    assert torch.equal(hf_model.calls[0]["mm_token_type_ids"], torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0,
                                                                             0, 1, 1, 0, 0, 0]]))
    assert torch.equal(hf_model.calls[0]["input_ids"], torch.tensor([expected_ids]))


@pytest.mark.parametrize("num_hidden_layers", [0, 50])
def test_rejects_conditioners_without_layer_50_input(num_hidden_layers: int) -> None:
    conditioner, hf_model = _conditioner(num_hidden_layers=num_hidden_layers)

    with pytest.raises(ValueError, match="requires more than 50"):
        _run_stage(conditioner, _FakeTokenizer(), _FakeProcessor())

    assert hf_model.calls == []


def test_rejects_missing_hidden_state_50() -> None:
    conditioner, _ = _conditioner(num_hidden_layers=51, hidden_state_count=50)

    with pytest.raises(ValueError, match=r"hidden_states\[50\]"):
        _run_stage(conditioner, _FakeTokenizer(), _FakeProcessor())


def test_rejects_prompt_batches() -> None:
    conditioner, _ = _conditioner()

    with pytest.raises(ValueError, match="single string"):
        _run_stage(conditioner, _FakeTokenizer(), _FakeProcessor(), prompt=["prompt"])


def test_conditioning_keeps_text_encoder_dtype() -> None:
    conditioner, hf_model = _conditioner(dtype=torch.bfloat16)
    processor = _FakeProcessor()

    _, state = _run_stage(conditioner, _FakeTokenizer(), processor, images=[object(), object()])

    assert state.prompt_embeds.dtype == torch.bfloat16
    assert hf_model.calls[0]["pixel_values"].dtype == torch.bfloat16
