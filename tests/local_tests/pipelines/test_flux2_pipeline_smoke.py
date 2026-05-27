# SPDX-License-Identifier: Apache-2.0
"""Smoke / preflight tests for the Flux2 Klein T2I pipeline.

The preflight validates import, registry, preset, and config wiring in an
environment with the expected optional packages. The load/generate smoke is
activated with CUDA plus
``FLUX2_MODEL_DIR=/path/to/black-forest-labs__FLUX.2-klein-4B``.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn


MODEL_DIR = Path(os.getenv("FLUX2_MODEL_DIR", ""))
FULL_MODEL_DIR = Path(os.getenv("FLUX2_FULL_MODEL_DIR", ""))
FULL_HEIGHT = int(os.getenv("FLUX2_FULL_HEIGHT", "128"))
FULL_WIDTH = int(os.getenv("FLUX2_FULL_WIDTH", "128"))
FULL_NUM_INFERENCE_STEPS = int(os.getenv("FLUX2_FULL_STEPS", "1"))
FULL_GUIDANCE_SCALE = float(os.getenv("FLUX2_FULL_GUIDANCE_SCALE", "4.0"))
FULL_MAX_SEQUENCE_LENGTH = int(os.getenv("FLUX2_FULL_MAX_SEQUENCE_LENGTH", "64"))
FULL_NUM_GPUS = int(os.getenv("FLUX2_FULL_NUM_GPUS", "2"))
FULL_TP_SIZE = int(os.getenv("FLUX2_FULL_TP_SIZE", str(FULL_NUM_GPUS)))
FULL_SP_SIZE = int(
    os.getenv(
        "FLUX2_FULL_SP_SIZE",
        "1" if FULL_NUM_GPUS > 1 else str(FULL_NUM_GPUS),
    )
)
requires_flux2_runtime = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 pipeline imports require the CUDA/kernel runtime",
)


@requires_flux2_runtime
def test_flux2_full_typed_surface_preflight() -> None:
    """Import + registry + preset wiring preflight for full Flux2."""
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.models.encoders.mistral3 import Mistral3TextConfig
    from fastvideo.configs.pipelines.flux_2 import Flux2PipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.models.registry import ModelRegistry
    from fastvideo.pipelines.basic.flux_2.flux_2_pipeline import (
        EntryClass,
        Flux2Pipeline,
    )

    assert Flux2Pipeline.__name__ == "Flux2Pipeline"
    assert EntryClass is Flux2Pipeline

    default_preset, model_family = registry.get_preset_selection(
        "black-forest-labs/FLUX.2-dev"
    )
    assert model_family == "flux2"
    assert default_preset == "flux2_dev"

    info = registry.get_model_info(
        "black-forest-labs/FLUX.2-dev",
        workload_type=WorkloadType.T2I,
        override_pipeline_cls_name="Flux2Pipeline",
    )
    assert info.pipeline_cls is Flux2Pipeline
    assert info.pipeline_config_cls is Flux2PipelineConfig

    names = {p.name for p in get_presets_for_family("flux2")}
    assert "flux2_dev" in names
    preset = get_preset("flux2_dev", "flux2")
    assert preset.defaults["num_inference_steps"] == 50
    assert preset.defaults["height"] == 1024
    assert preset.defaults["width"] == 1024
    assert preset.defaults["guidance_scale"] == 4.0
    assert preset.defaults["num_frames"] == 1

    cfg = Flux2PipelineConfig()
    assert cfg.embedded_cfg_scale == 4.0
    assert cfg.flux2_text_encoder_type == "mistral3"
    assert cfg.text_encoder_out_layers == (10, 20, 30)
    assert isinstance(cfg.text_encoder_configs[0], Mistral3TextConfig)
    model_cls, arch = ModelRegistry.resolve_model_cls(
        "Mistral3ForConditionalGeneration"
    )
    assert arch == "Mistral3ForConditionalGeneration"
    assert model_cls.__name__ == "Mistral3ForConditionalGeneration"


class _FakeFlux2Processor:

    def __init__(self) -> None:
        self.calls: list[tuple[list[list[dict[str, Any]]], dict[str, Any]]] = []

    def apply_chat_template(
        self,
        messages: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        self.calls.append((messages, kwargs))
        assert kwargs["add_generation_prompt"] is False
        assert kwargs["tokenize"] is True
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True
        assert messages[0][0]["role"] == "system"
        assert messages[0][1]["role"] == "user"
        batch_size = len(messages)
        max_length = int(kwargs["max_length"])
        input_ids = torch.arange(max_length, dtype=torch.long).repeat(
            batch_size,
            1,
        )
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeMistral3Encoder(nn.Module):

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self._hidden_size = hidden_size

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        **_kwargs: Any,
    ) -> Any:
        assert output_hidden_states is True
        assert use_cache is False
        assert attention_mask.shape == input_ids.shape
        base = input_ids.to(self.weight.dtype).unsqueeze(-1).expand(
            *input_ids.shape,
            self._hidden_size,
        )
        hidden_states = tuple(base + float(i) for i in range(31))
        return SimpleNamespace(hidden_states=hidden_states)


@requires_flux2_runtime
def test_flux2_full_text_stage_uses_mistral3_format_and_embedded_guidance() -> None:
    """Full Flux2 text encoding uses Mistral3 formatting and disables generic CFG."""
    from fastvideo.configs.pipelines.flux_2 import Flux2PipelineConfig
    from fastvideo.pipelines.basic.flux_2.flux_2_text_encoding import (
        Flux2TextEncodingStage,
    )
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    processor = _FakeFlux2Processor()
    encoder = _FakeMistral3Encoder()
    stage = Flux2TextEncodingStage(text_encoders=[encoder], tokenizers=[processor])
    cfg = Flux2PipelineConfig()
    cfg.text_encoder_out_layers = (10, 20, 30)
    args = SimpleNamespace(pipeline_config=cfg)

    batch = ForwardBatch(
        data_type="image",
        prompt="a cat [IMG] on a chair",
        guidance_scale=4.0,
        negative_prompt="should not be encoded",
    )
    assert batch.do_classifier_free_guidance is True

    out = stage.forward(batch, cast(Any, args))

    assert out.do_classifier_free_guidance is False
    assert out.negative_prompt_embeds == []
    assert len(out.prompt_embeds) == 1
    assert out.prompt_embeds[0].shape == (1, 512, 12)
    assert out.extra["flux2_txt_ids"].shape == (1, 512, 4)
    assert out.extra["flux2_txt_ids"][0, -1].tolist() == [0, 0, 0, 511]
    assert len(processor.calls) == 1
    messages, _kwargs = processor.calls[0]
    assert "[IMG]" not in messages[0][1]["content"][0]["text"]


@requires_flux2_runtime
def test_flux2_klein_typed_surface_preflight() -> None:
    """Import + registry + preset wiring preflight."""
    import fastvideo.registry as registry
    from fastvideo.api.presets import get_preset, get_presets_for_family
    from fastvideo.configs.pipelines.flux_2 import Flux2KleinPipelineConfig
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.basic.flux_2.flux_2_klein_pipeline import (
        EntryClass,
        Flux2KleinPipeline,
    )

    assert Flux2KleinPipeline.__name__ == "Flux2KleinPipeline"
    assert EntryClass is Flux2KleinPipeline

    default_preset, model_family = registry.get_preset_selection(
        "black-forest-labs/FLUX.2-klein-4B"
    )
    assert model_family == "flux2"
    assert default_preset == "flux2_klein_4b"

    info = registry.get_model_info(
        "black-forest-labs/FLUX.2-klein-4B",
        workload_type=WorkloadType.T2I,
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    assert info.pipeline_cls is Flux2KleinPipeline
    assert info.pipeline_config_cls is Flux2KleinPipelineConfig

    names = {p.name for p in get_presets_for_family("flux2")}
    assert "flux2_klein_4b" in names
    preset = get_preset("flux2_klein_4b", "flux2")
    assert preset.defaults["num_inference_steps"] == 4
    assert preset.defaults["height"] == 1024
    assert preset.defaults["width"] == 1024
    assert preset.defaults["guidance_scale"] == 1.0
    assert preset.defaults["num_frames"] == 1

    assert Flux2KleinPipeline._required_config_modules == [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 Klein pipeline load/generate smoke requires CUDA",
)
def test_flux2_klein_pipeline_load_generate_smoke() -> None:
    """Optional real load + four-step latent generate smoke for local weights."""
    if not MODEL_DIR.exists():
        pytest.skip("Set FLUX2_MODEL_DIR to activate Flux2 Klein load/generate smoke")

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(MODEL_DIR),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2KleinPipeline",
    )
    try:
        result = generator.generate_video(
            prompt="a photo of a banana on a wooden table, studio lighting",
            output_path="outputs_video/flux2_klein_smoke",
            save_video=False,
            return_frames=True,
            height=1024,
            width=1024,
            num_frames=1,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=0,
        )
    finally:
        generator.shutdown()

    assert isinstance(result, dict)
    result_dict = cast(dict[str, Any], result)
    samples = result_dict["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim in (3, 5)
    assert torch.isfinite(samples).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Flux2 full pipeline load/generate smoke requires CUDA",
)
def test_flux2_full_pipeline_load_generate_smoke() -> None:
    """Optional real load + short latent generate smoke for full Flux2 weights."""
    if not FULL_MODEL_DIR.exists():
        pytest.skip("Set FLUX2_FULL_MODEL_DIR to activate Flux2 full load/generate smoke")
    if torch.cuda.device_count() < FULL_NUM_GPUS:
        pytest.skip(
            f"Flux2 full load/generate smoke requires {FULL_NUM_GPUS} CUDA devices; "
            f"found {torch.cuda.device_count()}"
        )

    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(FULL_MODEL_DIR),
        num_gpus=FULL_NUM_GPUS,
        tp_size=FULL_TP_SIZE,
        sp_size=FULL_SP_SIZE,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="Flux2Pipeline",
    )
    try:
        result = generator.generate_video(
            prompt="a photo of a banana on a wooden table, studio lighting",
            output_path="outputs_video/flux2_full_smoke",
            save_video=False,
            return_frames=True,
            height=FULL_HEIGHT,
            width=FULL_WIDTH,
            num_frames=1,
            num_inference_steps=FULL_NUM_INFERENCE_STEPS,
            guidance_scale=FULL_GUIDANCE_SCALE,
            max_sequence_length=FULL_MAX_SEQUENCE_LENGTH,
            seed=0,
        )
    finally:
        generator.shutdown()

    assert isinstance(result, dict)
    result_dict = cast(dict[str, Any], result)
    samples = result_dict["samples"]
    assert torch.is_tensor(samples)
    assert samples.ndim in (3, 5)
    assert torch.isfinite(samples).all()
