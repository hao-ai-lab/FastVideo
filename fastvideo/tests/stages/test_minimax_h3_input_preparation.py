# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from PIL import Image
import pytest

from fastvideo.pipelines.basic.minimax_h3.reference import (
    MINIMAX_H3_REFERENCE_RESIZE_MODES,
    MiniMaxH3Reference,
    resolve_reference_image_size,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
    MiniMaxH3InputPreparationStage,
    resolve_target_num_frames,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


@pytest.mark.parametrize("requested, expected", [(120, 124), (345, 345), (346, 362), (360, 362), (362, 362)])
def test_target_frames_allow_padding_at_duration_limit(requested: int, expected: int) -> None:
    assert resolve_target_num_frames(requested) == expected
    assert resolve_target_num_frames(expected) == expected


@pytest.mark.parametrize("requested", [107, 363, 379])
def test_target_frames_reject_buckets_outside_duration_limits(requested: int) -> None:
    with pytest.raises(ValueError, match="MiniMax-H3 generates"):
        resolve_target_num_frames(requested)


@pytest.mark.parametrize("requested", [360.0, "360", None])
def test_target_frames_require_integer(requested: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        resolve_target_num_frames(requested)


# (width, height) -> (height, width) per mode on a 1344x768 canvas, as SGLang's
# minimax_h3_resolve_reference_image_shape resolves them.
_RESIZE_VECTORS = [
    ((500, 800), {"diffusers": (3264, 2048), "max": (800, 512), "match": (800, 512)}),
    ((2000, 2000), {"diffusers": (2048, 2048), "max": (1984, 1984), "match": (1024, 1024)}),
    ((4000, 3000), {"diffusers": (2048, 2720), "max": (2048, 2720), "match": (864, 1184)}),
    ((1000, 4000), {"diffusers": (8192, 2048), "max": (4000, 992), "match": (2016, 512)}),
    ((3000, 12000), {"diffusers": (8192, 2048), "max": (8192, 2048), "match": (2016, 512)}),
    ((64, 64), {"diffusers": (2048, 2048), "max": (64, 64), "match": (64, 64)}),
]


@pytest.mark.parametrize("mode", MINIMAX_H3_REFERENCE_RESIZE_MODES)
@pytest.mark.parametrize("size, expected", _RESIZE_VECTORS)
def test_reference_image_size_matches_sglang_resize_modes(size, expected, mode) -> None:
    assert resolve_reference_image_size(*size, mode, (768, 1344)) == expected[mode]


def test_reference_image_size_defaults_to_the_released_short_edge() -> None:
    assert resolve_reference_image_size(500, 800) == resolve_reference_image_size(500, 800, "diffusers")


def test_reference_image_size_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="reference_resize_mode must be one of"):
        resolve_reference_image_size(500, 800, "min")


def test_reference_image_size_match_needs_canvas() -> None:
    with pytest.raises(ValueError, match="needs the target canvas"):
        resolve_reference_image_size(500, 800, "match")


@pytest.mark.parametrize("extra, expected", [({}, (2048, 3264)), ({"reference_resize_mode": "max"}, (512, 800)),
                                             ({"reference_resize_mode": "match"}, (512, 800))])
def test_ref2va_input_preparation_reads_resize_mode_from_batch_extra(extra, expected) -> None:
    stage = MiniMaxH3InputPreparationStage(SimpleNamespace(spatial_compression_ratio=16, latent_channels=48),
                                           SimpleNamespace(sampling_rate=32000),
                                           ref2va=True)
    batch = ForwardBatch(data_type="video",
                         references=[MiniMaxH3Reference(source=Image.new("RGB", (500, 800)), media_type="image")],
                         height=768,
                         width=1344,
                         num_frames=124,
                         extra=dict(extra))
    stage._prepare_ref2va(batch)
    assert batch.references[0].image.size == expected
