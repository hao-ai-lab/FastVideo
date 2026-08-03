# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the MiniMax-H3 T2VA/FL2VA stage composition."""

from __future__ import annotations

import json
from types import SimpleNamespace

import av
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.testing import assert_close

import fastvideo.pipelines.composed_pipeline_base as composed_pipeline_base
from fastvideo.pipelines.stages import utils as stage_utils
from fastvideo.api.results import GenerationResult
from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.models.vaes.minimax_h3_audio import MiniMaxH3AudioDecoderOutput
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLOutput, DecoderOutput, DiagonalGaussianDistribution
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Pipeline
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    build_packed_sequence,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3AudioDecodingStage,
    MiniMaxH3ConditioningStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3InputPreparationStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3VideoDecodingStage,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.pipelines.pipeline_registry import PipelineType, import_pipeline_classes


_HEIGHT = 32
_WIDTH = 32
_NUM_FRAMES = 124
_NUM_LATENT_FRAMES = 37
_NUM_AUDIO_LATENTS = 207
_VIDEO_CHANNELS = 4
_AUDIO_CHANNELS = 6
_PATCH_SIZE = (1, 2, 2)
_AUDIO_SAMPLE_RATE = 32_000
_AUDIO_HOP_LENGTH = 800
_REQUEST_SEED = 73


@pytest.fixture(autouse=True)
def _keep_synthetic_pipeline_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(stage_utils, "get_local_torch_device", lambda: torch.device("cpu"))


class _TinyConditioner(nn.Module):
    """Deterministic stand-in for the separately-tested Qwen3-VL adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(arch_config=SimpleNamespace(num_hidden_layers=64))
        self.calls: list[dict[str, object]] = []

    @property
    def dtype(self) -> torch.dtype:
        return self.marker.dtype

    @property
    def num_hidden_layers(self) -> int:
        return 64

    def forward(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> BaseEncoderOutput:
        if input_ids is None:
            raise ValueError("input_ids are required")
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        values = torch.arange(input_ids.shape[1] * 8, device=input_ids.device, dtype=torch.float32)
        values = values.reshape(1, input_ids.shape[1], 8).to(self.dtype).div_(100)
        hidden_states = (torch.zeros_like(values), ) * 50 + (values, )
        return BaseEncoderOutput(last_hidden_state=values, hidden_states=hidden_states, attention_mask=attention_mask)


class _TinyTokenizer:

    _SPECIAL_TOKEN_IDS = {
        "<|vision_start|>": 11,
        "<|image_pad|>": 12,
        "<|vision_end|>": 13,
    }

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        input_ids = [21] if text.startswith("<Picture ") else [31, 32]
        return SimpleNamespace(input_ids=input_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS[token]


class _TinyImageProcessor:

    merge_size = 1

    def __init__(self) -> None:
        self.calls: list[list[Image.Image]] = []

    def __call__(self, images: list[Image.Image], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.calls.append(list(images))
        return {
            "pixel_values": torch.zeros(len(images), 1, 3),
            "image_grid_thw": torch.ones(len(images), 3, dtype=torch.long),
        }


class _TinyProcessor:

    def __init__(self) -> None:
        self.image_processor = _TinyImageProcessor()

    @staticmethod
    def create_mm_token_type_ids(token_ids: list[list[int]]) -> list[list[int]]:
        return [[0] * len(ids) for ids in token_ids]


class _TinyTransformer(nn.Module):
    """Records the packed call and emits non-zero velocity for every row."""

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(arch_config=SimpleNamespace(patch_size=_PATCH_SIZE))
        self.patch_size = _PATCH_SIZE
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append({name: value.detach().clone() for name, value in kwargs.items()})
        return (
            torch.full_like(kwargs["hidden_states"], 0.25),
            torch.full_like(kwargs["audio_hidden_states"], -0.125),
        )


class _TinyVideoVAE(nn.Module):
    """Tiny spatial encoder and deterministic H3-shaped video decoder."""

    spatial_compression_ratio = 16
    latent_channels = _VIDEO_CHANNELS

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            latent_channels=_VIDEO_CHANNELS,
            latents_mean=[0.0] * _VIDEO_CHANNELS,
            latents_std=[1.0] * _VIDEO_CHANNELS,
        )
        self.register_buffer("latents_mean", torch.zeros(1, _VIDEO_CHANNELS, 1, 1, 1))
        self.register_buffer("latents_std", torch.ones(1, _VIDEO_CHANNELS, 1, 1, 1))

    @staticmethod
    def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
        return pixels

    def encode_keyframe(
        self,
        pixels: torch.Tensor,
    ) -> AutoencoderKLOutput:
        pooled = F.adaptive_avg_pool3d(pixels.mean(dim=1, keepdim=True), (1, 2, 2))
        mean = pooled.repeat(1, _VIDEO_CHANNELS, 1, 1, 1)
        posterior = DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))
        return AutoencoderKLOutput(latent_dist=posterior)

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean

    @staticmethod
    def denormalize_pixels(video: torch.Tensor) -> torch.Tensor:
        return video

    def decode(self, latents: torch.Tensor) -> DecoderOutput:
        level = latents.mean(dim=(1, 2, 3, 4), keepdim=True).sigmoid()
        video = level.expand(latents.shape[0], 3, _NUM_FRAMES, _HEIGHT, _WIDTH).contiguous()
        return DecoderOutput(sample=video)


class _TinyAudioVAE(nn.Module):
    """Decodes the two channel-major latent batches into a stereo waveform."""

    latent_channels = _AUDIO_CHANNELS
    sampling_rate = _AUDIO_SAMPLE_RATE

    def __init__(self) -> None:
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            latent_channels=_AUDIO_CHANNELS,
            latents_mean=[0.0] * _AUDIO_CHANNELS,
            latents_std=[1.0] * _AUDIO_CHANNELS,
            sampling_rate=_AUDIO_SAMPLE_RATE,
        )
        self.register_buffer("latents_mean", torch.zeros(2, _AUDIO_CHANNELS, 1))
        self.register_buffer("latents_std", torch.ones(2, _AUDIO_CHANNELS, 1))

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latents_std + self.latents_mean

    def decode(self, latents: torch.Tensor) -> MiniMaxH3AudioDecoderOutput:
        mono = latents.mean(dim=1, keepdim=True).tanh()
        return MiniMaxH3AudioDecoderOutput(sample=mono.repeat_interleave(_AUDIO_HOP_LENGTH, dim=-1))


def _fastvideo_args() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_precision="fp32"),
        disable_autocast=False,
        enable_stage_verification=True,
        output_type="pt",
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
    )


def _keyframes(mode: str) -> tuple[Image.Image | None, Image.Image | None]:
    first = Image.new("RGB", (48, 80), color=(200, 20, 10))
    last = Image.new("RGB", (80, 48), color=(10, 30, 220))
    return (
        first if mode in ("first", "both") else None,
        last if mode in ("last", "both") else None,
    )


def _injected_latents() -> tuple[torch.Tensor, torch.Tensor]:
    video = torch.linspace(
        -1,
        1,
        _VIDEO_CHANNELS * _NUM_LATENT_FRAMES * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, _VIDEO_CHANNELS, _NUM_LATENT_FRAMES, 2, 2)
    audio = torch.linspace(
        -0.5,
        0.5,
        MINIMAX_H3_AUDIO_CHANNELS * _AUDIO_CHANNELS * _NUM_AUDIO_LATENTS,
        dtype=torch.float32,
    ).reshape(MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_CHANNELS, _NUM_AUDIO_LATENTS)
    return video, audio


def _make_batch(mode: str, *, inject_latents: bool, num_grid_points: int = 4) -> ForwardBatch:
    first, last = _keyframes(mode)
    video, audio = _injected_latents()
    return ForwardBatch(
        data_type="video",
        prompt="a tiny robot dances",
        negative_prompt="",
        pil_image=first,
        last_image=last,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        fps=24,
        num_inference_steps=num_grid_points,
        num_videos_per_prompt=1,
        seed=_REQUEST_SEED,
        generator=torch.Generator("cpu").manual_seed(_REQUEST_SEED),
        guidance_scale=1.0,
        batch_cfg=False,
        latents=video.clone() if inject_latents else None,
        audio_latents=audio.clone() if inject_latents else None,
        save_video=False,
        return_frames=False,
    )


def _components() -> SimpleNamespace:
    return SimpleNamespace(
        conditioner=_TinyConditioner(),
        tokenizer=_TinyTokenizer(),
        processor=_TinyProcessor(),
        transformer=_TinyTransformer(),
        vae=_TinyVideoVAE(),
        audio_vae=_TinyAudioVAE(),
        scheduler=MiniMaxH3Scheduler(shift=12.0),
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )


def _component_modules(components: SimpleNamespace) -> dict[str, object]:
    return {
        "text_encoder": components.conditioner,
        "tokenizer": components.tokenizer,
        "processor": components.processor,
        "transformer": components.transformer,
        "vae": components.vae,
        "audio_vae": components.audio_vae,
        "scheduler": components.scheduler,
        "audio_scheduler": components.audio_scheduler,
    }


def _composed_pipeline(components: SimpleNamespace, args: SimpleNamespace) -> MiniMaxH3Pipeline:
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.fastvideo_args = args
    pipeline.modules = _component_modules(components)
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.post_init_called = True
    pipeline.create_pipeline_stages(args)
    return pipeline


def _write_modular_manifest(path) -> dict[str, tuple[str, str]]:
    component_types = {
        "text_encoder": ("transformers", "Qwen3VLForConditionalGeneration"),
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
        "processor": ("transformers", "Qwen3VLProcessor"),
        "vae": ("diffusers", "AutoencoderKLMiniMaxH3"),
        "audio_vae": ("diffusers", "AutoencoderKLMiniMaxH3Audio"),
        "transformer": ("diffusers", "MiniMaxH3Transformer3DModel"),
        "scheduler": ("diffusers", "MiniMaxH3Scheduler"),
        "audio_scheduler": ("diffusers", "MiniMaxH3Scheduler"),
    }
    manifest = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "_blocks_class_name": "MiniMaxH3Blocks",
    }
    for name, (library, class_name) in component_types.items():
        (path / name).mkdir()
        manifest[name] = [
            library,
            class_name,
            {
                "type_hint": [library, class_name],
                "subfolder": name,
            },
        ]
    (path / "modular_model_index.json").write_text(json.dumps(manifest), encoding="utf-8")
    return component_types


def _prepare_latents(batch: ForwardBatch, components: SimpleNamespace, args: SimpleNamespace) -> None:
    stages = (
        MiniMaxH3InputPreparationStage(components.vae),
        MiniMaxH3ConditioningStage(
            components.conditioner,
            components.tokenizer,
            components.processor,
        ),
        MiniMaxH3LatentPreparationStage(
            components.transformer,
            components.vae,
            components.audio_vae,
            components.scheduler,
        ),
    )
    for stage in stages:
        stage(batch, args)


def test_composed_pipeline_runs_stage2_end_to_end() -> None:
    """The composed pipeline owns the tested stages in their execution order."""
    args = _fastvideo_args()
    batch = _make_batch("both", inject_latents=True)
    pipeline = _composed_pipeline(_components(), args)

    output = pipeline.forward(batch, args)

    assert list(pipeline._stage_name_mapping) == [
        "input_preparation_stage",
        "conditioning_stage",
        "latent_preparation_stage",
        "denoising_stage",
        "video_decoding_stage",
        "audio_decoding_stage",
    ]
    assert output.output is not None
    assert output.output.shape == (1, 3, _NUM_FRAMES, _HEIGHT, _WIDTH)
    assert output.extra["audio"].shape == (_NUM_AUDIO_LATENTS * _AUDIO_HOP_LENGTH, 2)
    assert set(output.extra) == {"audio", "audio_sample_rate"}


def test_pipeline_reads_diffusers_modular_manifest(tmp_path) -> None:
    """FastVideo resolves the published three-item component specifications."""
    component_types = _write_modular_manifest(tmp_path)

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.fastvideo_args = SimpleNamespace(revision=None)
    config = pipeline._load_config(str(tmp_path))

    assert config["_class_name"] == "MiniMaxH3ModularPipeline"
    assert set(config) == {"_class_name", "_diffusers_version", "_blocks_class_name", *component_types}
    for name, component_type in component_types.items():
        assert config[name][:2] == list(component_type)
        assert config[name][2] == {
            "type_hint": list(component_type),
            "subfolder": name,
        }


def test_pipeline_accepts_standard_and_hintless_modular_specs(tmp_path) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "scheduler").mkdir()
    manifest = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
        "scheduler": [
            "diffusers",
            "MiniMaxH3Scheduler",
            {"subfolder": "scheduler"},
        ],
    }
    (tmp_path / "modular_model_index.json").write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.fastvideo_args = SimpleNamespace(revision=None)

    config = pipeline._load_config(str(tmp_path))

    assert config["transformer"] == ["diffusers", "MiniMaxH3Transformer3DModel"]
    assert config["scheduler"] == [
        "diffusers",
        "MiniMaxH3Scheduler",
        {"subfolder": "scheduler"},
    ]


@pytest.mark.parametrize(
    "spec,error",
    [
        (["diffusers", "MiniMaxH3Scheduler", []], "Invalid Diffusers loading method"),
        (
            ["diffusers", "MiniMaxH3Scheduler", {"type_hint": ["diffusers"]}],
            "Invalid Diffusers type_hint",
        ),
        (
            ["diffusers", "MiniMaxH3Scheduler", {"subfolder": "video_scheduler"}],
            "requires component names and subfolders to match",
        ),
    ],
)
def test_pipeline_rejects_unsupported_modular_specs(tmp_path, spec: list[object], error: str) -> None:
    (tmp_path / "transformer").mkdir()
    manifest = {
        "_class_name": "MiniMaxH3ModularPipeline",
        "_diffusers_version": "0.36.0.dev0",
        "scheduler": spec,
    }
    (tmp_path / "modular_model_index.json").write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.fastvideo_args = SimpleNamespace(revision=None)

    with pytest.raises(ValueError, match=error):
        pipeline._load_config(str(tmp_path))


@pytest.mark.parametrize("config_source", ["default", "json", "instance"])
def test_pipeline_factory_loads_modular_manifest(tmp_path, monkeypatch, config_source: str) -> None:
    """Direct class loading reaches the modular manifest with the selected H3 config."""
    _write_modular_manifest(tmp_path)
    components = _components()
    monkeypatch.setattr(
        composed_pipeline_base,
        "maybe_init_distributed_environment_and_model_parallel",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(composed_pipeline_base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(composed_pipeline_base, "warmup_sequence_parallel_communication", lambda: None)
    monkeypatch.setattr(composed_pipeline_base, "attach_activation_trace", lambda module: None)

    pipeline_config = None
    explicit_config = None
    if config_source == "json":
        config_path = tmp_path / "pipeline_config.json"
        config_path.write_text(json.dumps({"dit_precision": "fp32"}), encoding="utf-8")
        pipeline_config = str(config_path)
    elif config_source == "instance":
        explicit_config = MiniMaxH3PipelineConfig(dit_precision="fp32")
        pipeline_config = explicit_config
    pipeline = MiniMaxH3Pipeline.from_pretrained(
        str(tmp_path),
        pipeline_config=pipeline_config,
        loaded_modules=_component_modules(components),
    )

    assert isinstance(pipeline.fastvideo_args.pipeline_config, MiniMaxH3PipelineConfig)
    if config_source == "json":
        assert pipeline.fastvideo_args.pipeline_config.dit_precision == "fp32"
        assert pipeline.fastvideo_args.pipeline_config.pipeline_config_path == pipeline_config
    elif config_source == "instance":
        assert pipeline.fastvideo_args.pipeline_config is explicit_config
    assert pipeline.model_path == str(tmp_path)
    assert list(pipeline._stage_name_mapping) == [
        "input_preparation_stage",
        "conditioning_stage",
        "latent_preparation_stage",
        "denoising_stage",
        "video_decoding_stage",
        "audio_decoding_stage",
    ]


def test_internal_pipeline_base_is_not_publicly_registered() -> None:
    """Only the manifest-compatible public subclasses are registry entrypoints."""
    basic_pipelines = import_pipeline_classes(PipelineType.BASIC)[PipelineType.BASIC.value]
    assert "MiniMaxH3Pipeline" not in basic_pipelines


def test_pipeline_requires_exact_scheduler_shifts() -> None:
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.modules = {
        "scheduler": MiniMaxH3Scheduler(shift=12.0),
        "audio_scheduler": MiniMaxH3Scheduler(shift=3.0),
    }
    pipeline.initialize_pipeline(_fastvideo_args())

    pipeline.modules["audio_scheduler"] = SimpleNamespace()
    with pytest.raises(ValueError, match="audio scheduler must expose shift=3"):
        pipeline.initialize_pipeline(_fastvideo_args())


def test_component_forwards_follow_cpu_offload_lifecycle(monkeypatch) -> None:
    """CPU-parked H3 components move for forward, then return to CPU."""
    monkeypatch.setattr(stage_utils, "get_local_torch_device", lambda: torch.device("cuda"))
    args = _fastvideo_args()
    args.text_encoder_cpu_offload = True
    args.vae_cpu_offload = True
    args.dit_cpu_offload = True
    args.dit_layerwise_offload = False
    args.use_fsdp_inference = False
    components = _components()

    def record_moves(module: nn.Module) -> list[torch.device]:
        calls: list[torch.device] = []

        def record(device, *args, **kwargs):
            del args, kwargs
            calls.append(torch.device(device))
            return module

        module.to = record
        return calls

    conditioner_moves = record_moves(components.conditioner)
    video_vae_moves = record_moves(components.vae)
    audio_vae_moves = record_moves(components.audio_vae)
    transformer_moves = record_moves(components.transformer)

    output = _composed_pipeline(components, args).forward(_make_batch("both", inject_latents=True), args)

    assert output.output is not None
    assert conditioner_moves == [torch.device("cuda"), torch.device("cpu")]
    assert video_vae_moves == [
        torch.device("cuda"),
        torch.device("cpu"),
        torch.device("cuda"),
        torch.device("cpu"),
    ]
    assert audio_vae_moves == [torch.device("cuda"), torch.device("cpu")]
    assert transformer_moves == [torch.device("cuda"), torch.device("cpu")]


def test_conditioning_keeps_text_encoder_dtype() -> None:
    """Prompt embeddings keep Qwen's BF16 output instead of an FP32 DiT island."""
    args = _fastvideo_args()
    batch = _make_batch("text", inject_latents=True)
    components = _components()
    components.conditioner.to(torch.bfloat16)

    MiniMaxH3InputPreparationStage(components.vae)(batch, args)
    MiniMaxH3ConditioningStage(
        components.conditioner,
        components.tokenizer,
        components.processor,
    )(batch, args)

    assert batch.prompt_embeds
    assert batch.prompt_embeds[0].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "mode,expected_anchors,expected_condition_rows",
    [
        ("text", (), 0),
        ("first", ("first",), 1),
        ("last", ("last",), 1),
        ("both", ("first", "last"), 2),
    ],
)
def test_t2va_fl2va_pipeline_contracts(
    mode: str,
    expected_anchors: tuple[str, ...],
    expected_condition_rows: int,
) -> None:
    """All four paths preserve layout/RNG/fixed rows and jointly decode AV."""
    args = _fastvideo_args()
    batch = _make_batch(mode, inject_latents=True)
    components = _components()
    _prepare_latents(batch, components, args)

    assert len(components.conditioner.calls) == 1
    pixel_values = components.conditioner.calls[0]["pixel_values"]
    if expected_condition_rows:
        assert isinstance(pixel_values, torch.Tensor)
        assert pixel_values.shape[0] == expected_condition_rows
    else:
        assert pixel_values is None
    layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
    assert batch.latents is not None
    assert batch.audio_latents is not None
    assert layout.num_condition_video_rows == expected_condition_rows
    text_token_tags = layout.token_tags[layout.text_indices]

    expected_layout = build_packed_sequence(
        text_token_tags,
        _NUM_LATENT_FRAMES,
        2,
        2,
        _NUM_AUDIO_LATENTS,
        _PATCH_SIZE,
        expected_anchors,
    )
    assert layout.sequence_length == expected_layout.sequence_length
    for name in ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"):
        assert_close(getattr(layout, name), getattr(expected_layout, name), rtol=0, atol=0)
    assert layout.position_ids.dtype == torch.float64
    assert set(layout.token_tags[layout.audio_indices].tolist()) == {MINIMAX_H3_AUDIO_TAG}
    assert set(layout.token_tags[layout.video_indices].tolist()) == {MINIMAX_H3_VIDEO_TAG}

    initial_video_rows = batch.latents.clone()
    initial_audio_rows = batch.audio_latents.clone()
    condition_prefix = initial_video_rows[:expected_condition_rows].clone()

    MiniMaxH3DenoisingStage(components.transformer, components.scheduler, components.audio_scheduler)(batch, args)

    assert batch.timesteps is not None
    assert len(batch.timesteps) == batch.num_inference_steps - 1
    assert len(components.transformer.calls) == batch.num_inference_steps - 1
    for call in components.transformer.calls:
        assert call["timestep"].ndim == 1
        assert call["timestep_indices"].shape == layout.token_tags.shape
        assert_close(call["position_ids"], layout.position_ids, rtol=0, atol=0)
        assert_close(call["token_tags"], layout.token_tags, rtol=0, atol=0)
        assert_close(call["hidden_states"][0, :expected_condition_rows], condition_prefix, rtol=0, atol=0)
    assert_close(batch.latents[:expected_condition_rows], condition_prefix, rtol=0, atol=0)
    assert not torch.equal(batch.latents[expected_condition_rows:], initial_video_rows[expected_condition_rows:])
    assert not torch.equal(batch.audio_latents, initial_audio_rows)

    # Injecting both target streams skips their draws. Only one request-generator
    # draw per keyframe condition is allowed; fixed-seed posterior samples do not
    # consume this stream.
    expected_generator = torch.Generator("cpu").manual_seed(_REQUEST_SEED)
    for _ in expected_anchors:
        torch.randn((1, _VIDEO_CHANNELS, 1, 2, 2), generator=expected_generator)
    assert_close(
        torch.randn(8, generator=batch.generator),
        torch.randn(8, generator=expected_generator),
        rtol=0,
        atol=0,
    )

    MiniMaxH3VideoDecodingStage(components.vae, components.transformer)(batch, args)
    MiniMaxH3AudioDecodingStage(components.audio_vae)(batch, args)
    assert batch.output is not None
    assert batch.output.shape == (1, 3, _NUM_FRAMES, _HEIGHT, _WIDTH)
    assert batch.extra["audio"].shape == (_NUM_AUDIO_LATENTS * _AUDIO_HOP_LENGTH, 2)
    assert batch.extra["audio_sample_rate"] == _AUDIO_SAMPLE_RATE
    assert batch.prompt_embeds == []
    assert set(batch.extra) == {"audio", "audio_sample_rate"}
    video_duration = _NUM_FRAMES / 24
    audio_duration = batch.extra["audio"].shape[0] / batch.extra["audio_sample_rate"]
    assert abs(video_duration - audio_duration) <= 1 / 24


def test_target_noise_draws_video_then_audio() -> None:
    """Without overrides, target video noise precedes channel-major audio noise."""
    args = _fastvideo_args()
    batch = _make_batch("text", inject_latents=False)
    components = _components()
    _prepare_latents(batch, components, args)
    assert batch.latents is not None
    assert batch.audio_latents is not None

    expected_generator = torch.Generator("cpu").manual_seed(_REQUEST_SEED)
    expected_video = torch.randn(
        (1, _VIDEO_CHANNELS, _NUM_LATENT_FRAMES, 2, 2),
        generator=expected_generator,
        dtype=torch.float32,
    )
    expected_audio = torch.randn(
        (_NUM_AUDIO_LATENTS * MINIMAX_H3_AUDIO_CHANNELS, _AUDIO_CHANNELS),
        generator=expected_generator,
        dtype=torch.float32,
    )
    assert_close(batch.latents, patchify_video_latents(expected_video, _PATCH_SIZE), rtol=0, atol=0)
    assert_close(batch.audio_latents, expected_audio, rtol=0, atol=0)
    assert_close(
        torch.randn(8, generator=batch.generator),
        torch.randn(8, generator=expected_generator),
        rtol=0,
        atol=0,
    )


def test_latent_output_drops_internal_pipeline_state() -> None:
    """Only CPU output latents cross the FastVideo executor boundary."""
    args = _fastvideo_args()
    args.output_type = "latent"
    batch = _make_batch("text", inject_latents=True)
    components = _components()
    _prepare_latents(batch, components, args)

    MiniMaxH3VideoDecodingStage(components.vae, components.transformer)(batch, args)
    MiniMaxH3AudioDecodingStage(components.audio_vae)(batch, args)

    assert batch.output is not None
    assert batch.output.shape == (1, _VIDEO_CHANNELS, _NUM_LATENT_FRAMES, 2, 2)
    assert batch.extra["audio"].shape == (
        MINIMAX_H3_AUDIO_CHANNELS,
        _AUDIO_CHANNELS,
        _NUM_AUDIO_LATENTS,
    )
    assert batch.extra["audio_sample_rate"] == _AUDIO_SAMPLE_RATE
    assert batch.prompt_embeds == []
    assert set(batch.extra) == {"audio", "audio_sample_rate"}


def test_joint_output_muxes_one_video_and_stereo_audio_stream(tmp_path) -> None:
    """FastVideo's existing single-pass writer accepts the H3 output contract."""
    output_path = tmp_path / "minimax_h3_tiny.mp4"
    args = _fastvideo_args()
    batch = _make_batch("both", inject_latents=True)
    components = _components()
    _prepare_latents(batch, components, args)
    MiniMaxH3DenoisingStage(components.transformer, components.scheduler, components.audio_scheduler)(batch, args)
    MiniMaxH3VideoDecodingStage(components.vae, components.transformer)(batch, args)
    MiniMaxH3AudioDecodingStage(components.audio_vae)(batch, args)

    assert batch.output is not None
    frames = [
        (frame * 255).to(torch.uint8).contiguous().numpy()
        for frame in batch.output[0].permute(1, 2, 3, 0)
    ]
    result = GenerationResult.from_legacy_result({
        "samples": batch.output,
        "frames": frames,
        "audio": batch.extra["audio"],
        "audio_sample_rate": batch.extra["audio_sample_rate"],
    })
    assert result.samples is batch.output
    assert result.frames is frames
    assert result.audio is batch.extra["audio"]
    assert result.audio_sample_rate == _AUDIO_SAMPLE_RATE
    assert result.extra == {}

    assert VideoGenerator._save_video_with_audio_single_pass(
        output_path=str(output_path),
        frames=result.frames,
        fps=24,
        audio=result.audio,
        sample_rate=result.audio_sample_rate,
    )
    assert output_path.is_file()
    with av.open(str(output_path)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].width == _WIDTH
        assert container.streams.video[0].height == _HEIGHT
        assert container.streams.audio[0].codec_context.layout.name == "stereo"
        assert container.streams.audio[0].codec_context.sample_rate == _AUDIO_SAMPLE_RATE
