# SPDX-License-Identifier: Apache-2.0
"""CPU acceptance contracts for the MiniMax-H3 Ref2VA pipeline."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch.testing import assert_close

import fastvideo.pipelines.basic.minimax_h3.stages._module_lifecycle as module_lifecycle
import fastvideo.pipelines.basic.minimax_h3.stages.reference_preparation as reference_preparation
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioDiagonalGaussianDistribution,
    MiniMaxH3AudioEncoderOutput,
)
from fastvideo.models.vaes.minimax_h3_video import AutoencoderKLOutput, DiagonalGaussianDistribution
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3RefPipeline
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    patchify_video_latents,
)
from fastvideo.pipelines.basic.minimax_h3.stages import (
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3Ref2VAConditioningStage,
    MiniMaxH3Ref2VALayoutPreparationStage,
    MiniMaxH3ReferenceEncodingStage,
    MiniMaxH3ReferencePreparationStage,
    MiniMaxH3TimestepPreparationStage,
)
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3Reference, get_minimax_h3_state
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from tests.local_tests.minimax_h3.test_minimax_h3_pipeline import (
    _AUDIO_CHANNELS,
    _AUDIO_HOP_LENGTH,
    _AUDIO_SAMPLE_RATE,
    _HEIGHT,
    _NUM_AUDIO_LATENTS,
    _NUM_FRAMES,
    _NUM_LATENT_FRAMES,
    _PATCH_SIZE,
    _REQUEST_SEED,
    _VIDEO_CHANNELS,
    _WIDTH,
    _TinyAudioVAE,
    _TinyConditioner,
    _TinyTransformer,
    _TinyVideoVAE,
    _fastvideo_args,
    _injected_latents,
)

_REFERENCE_AUDIO_HOP_LENGTH = 4


@pytest.fixture(autouse=True)
def _keep_reference_acceptance_tiny(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep production reference semantics while replacing only released canvas sizes."""
    monkeypatch.setattr(module_lifecycle, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        reference_preparation,
        "prepare_reference_image",
        lambda image, height, width: image.resize((_WIDTH, _HEIGHT)),
    )
    monkeypatch.setattr(
        reference_preparation,
        "prepare_reference_frames",
        lambda frames, num_frames: frames[:num_frames],
    )


class _RefTokenizer:

    _SPECIAL_TOKEN_IDS = {
        "<|vision_start|>": 11,
        "<|image_pad|>": 12,
        "<|video_pad|>": 13,
        "<|vision_end|>": 14,
    }

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._text_ids: dict[str, int] = {}

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        self.calls.append(text)
        token_id = self._text_ids.setdefault(text, 100 + len(self._text_ids))
        return SimpleNamespace(input_ids=[token_id])

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._SPECIAL_TOKEN_IDS[token]


class _RefImageProcessor:

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


class _RefVideoProcessor:

    def __init__(self) -> None:
        self.calls: list[list[np.ndarray]] = []

    def __call__(
        self,
        videos: list[np.ndarray],
        do_sample_frames: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert not do_sample_frames
        assert return_tensors == "pt"
        self.calls.append(list(videos))
        grids = [[math.ceil(video.shape[0] / 2), 1, 1] for video in videos]
        return {
            "pixel_values_videos": torch.zeros(len(videos), 1, 3),
            "video_grid_thw": torch.tensor(grids, dtype=torch.long),
        }


class _RefProcessor:

    def __init__(self) -> None:
        self.image_processor = _RefImageProcessor()
        self.video_processor = _RefVideoProcessor()

    @staticmethod
    def create_mm_token_type_ids(token_ids: list[list[int]]) -> list[list[int]]:
        return [[0] * len(ids) for ids in token_ids]


class _RefVideoVAE(_TinyVideoVAE):

    def encode(self, pixels: torch.Tensor) -> AutoencoderKLOutput:
        num_latent_frames = 1 if pixels.shape[2] == 1 else 2
        pooled = F.adaptive_avg_pool3d(
            pixels.mean(dim=1, keepdim=True),
            (num_latent_frames, 2, 2),
        )
        mean = pooled.repeat(1, _VIDEO_CHANNELS, 1, 1, 1)
        posterior = DiagonalGaussianDistribution(torch.cat((mean, torch.zeros_like(mean)), dim=1))
        return AutoencoderKLOutput(latent_dist=posterior)


class _RefAudioVAE(_TinyAudioVAE):

    hop_length = _REFERENCE_AUDIO_HOP_LENGTH

    def encode(self, sample: torch.Tensor) -> MiniMaxH3AudioEncoderOutput:
        if sample.ndim != 3 or sample.shape[1] != 1:
            raise ValueError(f"sample must be [batch, 1, samples], got {tuple(sample.shape)}")
        num_latents = math.ceil(sample.shape[-1] / self.hop_length)
        padded = F.pad(sample, (0, num_latents * self.hop_length - sample.shape[-1]))
        mean = padded.reshape(sample.shape[0], 1, num_latents, self.hop_length).mean(-1)
        mean = mean.repeat(1, _AUDIO_CHANNELS, 1)
        posterior = MiniMaxH3AudioDiagonalGaussianDistribution(mean, torch.zeros_like(mean))
        return MiniMaxH3AudioEncoderOutput(latent_dist=posterior)


def _components() -> SimpleNamespace:
    return SimpleNamespace(
        conditioner=_TinyConditioner(),
        tokenizer=_RefTokenizer(),
        processor=_RefProcessor(),
        transformer=_TinyTransformer(),
        vae=_RefVideoVAE(),
        audio_vae=_RefAudioVAE(),
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


def _composed_pipeline(components: SimpleNamespace, args: SimpleNamespace) -> MiniMaxH3RefPipeline:
    pipeline = MiniMaxH3RefPipeline.__new__(MiniMaxH3RefPipeline)
    pipeline.fastvideo_args = args
    pipeline.modules = _component_modules(components)
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline.post_init_called = True
    pipeline.create_pipeline_stages(args)
    return pipeline


def _image(color: tuple[int, int, int]) -> MiniMaxH3Reference:
    return MiniMaxH3Reference(
        source=Image.new("RGB", (48, 80), color=color),
        media_type="image",
    )


def _video(*, soundtrack: bool, fps: float | None = 24.0) -> MiniMaxH3Reference:
    values = np.arange(25, dtype=np.uint8)[:, None, None, None]
    frames = np.broadcast_to(values, (25, _HEIGHT, _WIDTH, 3)).copy()
    return MiniMaxH3Reference(
        source=frames,
        media_type="video",
        soundtrack=torch.linspace(-1, 1, 8).reshape(1, 8) if soundtrack else None,
        fps=fps,
        sample_rate=_AUDIO_SAMPLE_RATE if soundtrack else None,
    )


def _audio(*, samples: int = 8, sample_rate: int | None = _AUDIO_SAMPLE_RATE) -> MiniMaxH3Reference:
    return MiniMaxH3Reference(
        source=torch.linspace(-0.5, 0.5, samples).reshape(1, samples),
        media_type="audio",
        sample_rate=sample_rate,
    )


def _scenario(name: str) -> tuple[list[MiniMaxH3Reference], int, int]:
    if name == "multiple_images":
        return [_image((220, 10, 20)), _image((10, 20, 220))], 2, 0
    if name == "video_soundtrack":
        return [_video(soundtrack=True)], 2, 4
    if name == "image_audio":
        return [_image((220, 10, 20)), _audio()], 1, 4
    if name == "mixed":
        return [
            _image((220, 10, 20)),
            _audio(),
            _video(soundtrack=True),
            _image((10, 20, 220)),
        ], 4, 8
    raise AssertionError(f"unknown scenario: {name}")


def _make_batch(
    references: list[MiniMaxH3Reference],
    *,
    inject_latents: bool,
    seed: int = _REQUEST_SEED,
    num_grid_points: int = 3,
) -> ForwardBatch:
    video, audio = _injected_latents()
    return ForwardBatch(
        data_type="video",
        prompt="a tiny robot dances",
        negative_prompt="",
        references=references,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_NUM_FRAMES,
        fps=24,
        num_inference_steps=num_grid_points,
        num_videos_per_prompt=1,
        seed=seed,
        generator=torch.Generator("cpu").manual_seed(seed),
        guidance_scale=1.0,
        batch_cfg=False,
        latents=video.clone() if inject_latents else None,
        audio_latents=audio.clone() if inject_latents else None,
        save_video=False,
        return_frames=False,
    )


def _prepare_reference_latents(
    batch: ForwardBatch,
    components: SimpleNamespace,
    args: SimpleNamespace,
) -> None:
    stages = (
        MiniMaxH3ReferencePreparationStage(components.vae, components.audio_vae),
        MiniMaxH3Ref2VAConditioningStage(
            components.conditioner,
            components.tokenizer,
            components.processor,
        ),
        MiniMaxH3ReferenceEncodingStage(
            components.vae,
            components.audio_vae,
            components.transformer,
            components.scheduler,
        ),
        MiniMaxH3Ref2VALayoutPreparationStage(components.transformer),
        MiniMaxH3LatentPreparationStage(components.transformer, components.vae, components.audio_vae),
    )
    for stage in stages:
        stage(batch, args)


def _expected_condition_indices(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    assert state.text_token_tags is not None
    cursor = int(state.text_token_tags.shape[0])
    video_indices: list[int] = []
    audio_indices: list[int] = []
    for reference in state.prepared_references:
        if reference.media_type == "image":
            count = reference.num_latent_frames * (reference.latent_height // 2) * (reference.latent_width // 2)
            video_indices.extend(range(cursor, cursor + count))
            cursor += count
        elif reference.media_type == "audio":
            count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS
            audio_indices.extend(range(cursor, cursor + count))
            cursor += count
        else:
            audio_count = reference.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS if reference.has_audio else 0
            audio_indices.extend(range(cursor, cursor + audio_count))
            cursor += audio_count
            video_count = reference.num_latent_frames * (reference.latent_height // 2) * (
                reference.latent_width // 2
            )
            video_indices.extend(range(cursor, cursor + video_count))
            cursor += video_count
    return torch.tensor(video_indices, dtype=torch.long), torch.tensor(audio_indices, dtype=torch.long)


def test_ref_pipeline_stage_chain_runs_multiple_images_end_to_end() -> None:
    args = _fastvideo_args()
    components = _components()
    batch = _make_batch(_scenario("multiple_images")[0], inject_latents=True)
    pipeline = _composed_pipeline(components, args)

    output = pipeline.forward(batch, args)

    assert list(pipeline._stage_name_mapping) == [
        "reference_preparation_stage",
        "conditioning_stage",
        "reference_encoding_stage",
        "layout_preparation_stage",
        "latent_preparation_stage",
        "timestep_preparation_stage",
        "denoising_stage",
        "video_decoding_stage",
        "audio_decoding_stage",
    ]
    assert MiniMaxH3RefPipeline._extra_config_module_map == {"transformer": "transformer_ref"}
    assert "transformer_ref" not in MiniMaxH3RefPipeline._required_config_modules
    assert output.output is not None
    assert output.output.shape == (1, 3, _NUM_FRAMES, _HEIGHT, _WIDTH)
    assert output.extra["audio"].shape == (_NUM_AUDIO_LATENTS * _AUDIO_HOP_LENGTH, 2)
    assert output.extra["audio_sample_rate"] == _AUDIO_SAMPLE_RATE
    assert set(output.extra) == {"audio", "audio_sample_rate"}
    assert output.references is None
    assert len(components.transformer.calls) == batch.num_inference_steps - 1

    images = components.processor.image_processor.calls[0]
    assert [image.getpixel((0, 0)) for image in images] == [(220, 10, 20), (10, 20, 220)]


@pytest.mark.parametrize(
    "scenario",
    ["multiple_images", "video_soundtrack", "image_audio", "mixed"],
)
def test_reference_combinations_build_fixed_condition_rows_and_layout(scenario: str) -> None:
    references, expected_video_rows, expected_audio_rows = _scenario(scenario)
    args = _fastvideo_args()
    components = _components()
    batch = _make_batch(references, inject_latents=True)

    _prepare_reference_latents(batch, components, args)
    state = get_minimax_h3_state(batch)

    assert [reference.media_type for reference in state.prepared_references] == [
        reference.media_type for reference in references
    ]
    assert state.layout is not None
    assert state.video_latents is not None
    assert state.audio_latents is not None
    assert state.condition_video_latents is not None
    assert state.layout.num_condition_video_rows == expected_video_rows
    assert state.layout.num_condition_audio_rows == expected_audio_rows
    assert state.video_latents.shape[0] == expected_video_rows + _NUM_LATENT_FRAMES
    assert state.audio_latents.shape[0] == expected_audio_rows + 2 * _NUM_AUDIO_LATENTS
    assert_close(
        state.video_latents[:expected_video_rows],
        state.condition_video_latents,
        rtol=0,
        atol=0,
    )
    if expected_audio_rows:
        assert state.condition_audio_latents is not None
        assert_close(
            state.audio_latents[:expected_audio_rows],
            state.condition_audio_latents,
            rtol=0,
            atol=0,
        )
    else:
        assert state.condition_audio_latents is None

    expected_video_indices, expected_audio_indices = _expected_condition_indices(state)
    assert_close(
        state.layout.video_indices[:expected_video_rows],
        expected_video_indices,
        rtol=0,
        atol=0,
    )
    assert_close(
        state.layout.audio_indices[:expected_audio_rows],
        expected_audio_indices,
        rtol=0,
        atol=0,
    )

    fixed_video = state.video_latents[:expected_video_rows].clone()
    fixed_audio = state.audio_latents[:expected_audio_rows].clone()
    target_video = state.video_latents[expected_video_rows:].clone()
    target_audio = state.audio_latents[expected_audio_rows:].clone()
    MiniMaxH3TimestepPreparationStage(components.scheduler, components.audio_scheduler)(batch, args)
    MiniMaxH3DenoisingStage(components.transformer, components.scheduler, components.audio_scheduler)(batch, args)

    assert_close(state.video_latents[:expected_video_rows], fixed_video, rtol=0, atol=0)
    assert_close(state.audio_latents[:expected_audio_rows], fixed_audio, rtol=0, atol=0)
    assert not torch.equal(state.video_latents[expected_video_rows:], target_video)
    assert not torch.equal(state.audio_latents[expected_audio_rows:], target_audio)
    for call in components.transformer.calls:
        assert_close(call["hidden_states"][0, :expected_video_rows], fixed_video, rtol=0, atol=0)
        assert_close(call["audio_hidden_states"][0, :expected_audio_rows], fixed_audio, rtol=0, atol=0)


def test_video_soundtrack_is_one_audio_then_video_reference_block() -> None:
    args = _fastvideo_args()
    components = _components()
    batch = _make_batch([_video(soundtrack=True)], inject_latents=True)

    _prepare_reference_latents(batch, components, args)
    state = get_minimax_h3_state(batch)

    assert state.layout is not None
    assert state.text_token_tags is not None
    text_rows = state.text_token_tags.shape[0]
    assert state.layout.audio_indices[:4].tolist() == list(range(text_rows, text_rows + 4))
    assert state.layout.video_indices[:2].tolist() == [text_rows + 4, text_rows + 5]
    assert components.tokenizer.calls[:4] == [
        "<Audio 1>: ",
        "<Video 1>: ",
        "<0.2 seconds>",
        "<1.0 seconds>",
    ]
    assert state.prepared_references[0].block_timestamps == [0.25, 1.0]


def test_mixed_reference_order_changes_presentation_and_layout() -> None:
    args = _fastvideo_args()

    first_components = _components()
    first_batch = _make_batch([_image((200, 0, 0)), _audio()], inject_latents=True)
    _prepare_reference_latents(first_batch, first_components, args)
    first_state = get_minimax_h3_state(first_batch)

    second_components = _components()
    second_batch = _make_batch([_audio(), _image((200, 0, 0))], inject_latents=True)
    _prepare_reference_latents(second_batch, second_components, args)
    second_state = get_minimax_h3_state(second_batch)

    assert first_components.tokenizer.calls[:2] == ["<Picture 1>: ", "<Audio 1>: "]
    assert second_components.tokenizer.calls[:2] == ["<Audio 1>: ", "<Picture 1>: "]
    first_input_ids = first_components.conditioner.calls[0]["input_ids"]
    second_input_ids = second_components.conditioner.calls[0]["input_ids"]
    assert isinstance(first_input_ids, torch.Tensor)
    assert isinstance(second_input_ids, torch.Tensor)
    assert not torch.equal(first_input_ids, second_input_ids)
    assert first_state.layout is not None and second_state.layout is not None
    assert not torch.equal(first_state.layout.position_ids, second_state.layout.position_ids)
    assert first_state.layout.video_indices[0] < first_state.layout.audio_indices[0]
    assert second_state.layout.audio_indices[0] < second_state.layout.video_indices[0]


def test_request_rng_draws_each_visual_condition_then_video_then_audio() -> None:
    args = _fastvideo_args()
    components = _components()
    references = [_image((220, 10, 20)), _image((10, 20, 220))]
    batch = _make_batch(references, inject_latents=False)

    _prepare_reference_latents(batch, components, args)
    state = get_minimax_h3_state(batch)

    assert state.layout is not None
    assert state.video_latents is not None
    assert state.audio_latents is not None
    assert state.layout.num_condition_video_rows == 2

    expected_generator = torch.Generator("cpu").manual_seed(_REQUEST_SEED)
    for _ in references:
        torch.randn(
            (1, _VIDEO_CHANNELS, 1, 2, 2),
            generator=expected_generator,
            dtype=torch.float32,
        )
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

    assert_close(
        state.video_latents[state.layout.num_condition_video_rows:],
        patchify_video_latents(expected_video, _PATCH_SIZE),
        rtol=0,
        atol=0,
    )
    assert_close(state.audio_latents, expected_audio, rtol=0, atol=0)
    assert_close(
        torch.randn(8, generator=batch.generator),
        torch.randn(8, generator=expected_generator),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "references,error_type,message",
    [
        ([], ValueError, "at least one reference"),
        ([object()], TypeError, "Every Ref2VA entry"),
        ([_image((0, 0, 0))] * 10, ValueError, "at most 9 image references"),
        (
            [_image((0, 0, 0))] + [_video(soundtrack=False)] * 4,
            ValueError,
            "at most 3 video references",
        ),
        (
            [_image((0, 0, 0))] + [_audio()] * 4,
            ValueError,
            "at most 3 audio references",
        ),
        (
            [_image((0, 0, 0))] * 9 + [_video(soundtrack=False)] * 3 + [_audio()],
            ValueError,
            "at most 12 references",
        ),
        ([_audio(), _audio()], ValueError, "paired with at least one image or video"),
    ],
    ids=[
        "empty",
        "untyped",
        "images",
        "videos",
        "audios",
        "total",
        "audio_only",
    ],
)
def test_reference_count_and_pairing_validation(
    references: list[Any],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        MiniMaxH3ReferencePreparationStage._validate_references(references)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"source": None}, "requires a media source"),
        ({"source": Image.new("RGB", (8, 8)), "media_type": "image", "soundtrack": torch.zeros(1, 4)},
         "Only a video reference"),
        ({"source": np.zeros((2, 8, 8, 3), dtype=np.uint8), "media_type": "video", "fps": 0.0},
         "fps.*positive"),
        ({"source": torch.zeros(1, 4), "media_type": "audio", "fps": 24.0}, "fps.*only valid for video"),
        ({"source": Image.new("RGB", (8, 8)), "media_type": "image", "sample_rate": 32_000},
         "sample_rate.*only valid"),
        ({"source": torch.zeros(1, 4), "media_type": "audio", "sample_rate": 0}, "sample_rate.*positive"),
    ],
    ids=["missing_source", "soundtrack_pair", "video_fps", "audio_fps", "image_rate", "audio_rate"],
)
def test_reference_metadata_rejects_invalid_pairing_and_rates(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MiniMaxH3Reference(**kwargs)


def test_silent_video_rate_and_waveform_channel_validation() -> None:
    stage = MiniMaxH3ReferencePreparationStage(_RefVideoVAE(), _RefAudioVAE())
    silent_video = MiniMaxH3Reference(
        source=np.zeros((2, _HEIGHT, _WIDTH, 3), dtype=np.uint8),
        media_type="video",
        sample_rate=_AUDIO_SAMPLE_RATE,
    )
    with pytest.raises(ValueError, match="silent video.*sample_rate"):
        stage._prepare_reference(silent_video, _NUM_FRAMES, _AUDIO_SAMPLE_RATE)

    invalid_audio = MiniMaxH3Reference(
        source=torch.zeros(3, 8),
        media_type="audio",
        sample_rate=_AUDIO_SAMPLE_RATE,
    )
    with pytest.raises(ValueError, match="mono or stereo"):
        stage._prepare_reference(invalid_audio, _NUM_FRAMES, _AUDIO_SAMPLE_RATE)


def test_reference_fps_and_sample_rate_are_resolved_before_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    resample_calls: list[tuple[int, int]] = []

    def resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        resample_calls.append((source_rate, target_rate))
        size = round(waveform.shape[-1] * target_rate / source_rate)
        return F.interpolate(waveform[None], size=size, mode="linear", align_corners=False)[0]

    class _Resample:

        def __init__(self, source_rate: int, target_rate: int) -> None:
            self.source_rate = source_rate
            self.target_rate = target_rate

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            return resample(waveform, self.source_rate, self.target_rate)

    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        SimpleNamespace(transforms=SimpleNamespace(Resample=_Resample)),
    )
    stage = MiniMaxH3ReferencePreparationStage(_RefVideoVAE(), _RefAudioVAE())
    args = _fastvideo_args()

    frames = np.zeros((3, _HEIGHT, _WIDTH, 3), dtype=np.uint8)
    waveform = torch.linspace(-1, 1, 8).reshape(1, 8)
    resampled_batch = _make_batch(
        [
            MiniMaxH3Reference(source=frames, media_type="video", fps=12.0),
            MiniMaxH3Reference(source=waveform, media_type="audio", sample_rate=_AUDIO_SAMPLE_RATE // 2),
        ],
        inject_latents=True,
    )
    stage(resampled_batch, args)
    resampled = get_minimax_h3_state(resampled_batch).prepared_references

    assert resampled[0].frames is not None and resampled[0].frames.shape[0] == 6
    assert resampled[1].waveform is not None and resampled[1].waveform.shape == (2, 16)
    assert resample_calls == [(_AUDIO_SAMPLE_RATE // 2, _AUDIO_SAMPLE_RATE)]

    default_batch = _make_batch(
        [
            MiniMaxH3Reference(source=frames, media_type="video"),
            MiniMaxH3Reference(source=waveform, media_type="audio"),
        ],
        inject_latents=True,
    )
    stage(default_batch, args)
    defaulted = get_minimax_h3_state(default_batch).prepared_references

    assert defaulted[0].frames is not None and defaulted[0].frames.shape[0] == 3
    assert defaulted[1].waveform is not None and defaulted[1].waveform.shape == (2, 8)
    assert resample_calls == [(_AUDIO_SAMPLE_RATE // 2, _AUDIO_SAMPLE_RATE)]
