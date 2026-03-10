"""Unit tests for the OpenAI-compatible API server helpers (no GPU needed)."""

import os
from unittest.mock import patch

import pytest

from fastvideo.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
    ImageResponseData,
    VideoGenerationsRequest,
    VideoListResponse,
    VideoResponse,
    generate_request_id,
)
from fastvideo.entrypoints.openai.utils import (
    choose_image_ext,
    merge_image_input_list,
    parse_size,
)

# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


class TestParseSize:

    def test_valid(self):
        assert parse_size("1024x768") == (1024, 768)

    def test_valid_uppercase(self):
        assert parse_size("512X512") == (512, 512)

    def test_valid_with_spaces(self):
        assert parse_size("  720 x 480 ") == (720, 480)

    def test_invalid_single_number(self):
        assert parse_size("1024") == (None, None)

    def test_invalid_non_numeric(self):
        assert parse_size("widexhigh") == (None, None)

    def test_invalid_empty(self):
        assert parse_size("") == (None, None)

    def test_invalid_triple(self):
        assert parse_size("1x2x3") == (None, None)

    def test_zero_dimensions(self):
        assert parse_size("0x0") == (0, 0)


# ---------------------------------------------------------------------------
# choose_image_ext
# ---------------------------------------------------------------------------


class TestChooseImageExt:

    def test_explicit_png(self):
        assert choose_image_ext("png", None) == "png"

    def test_explicit_webp(self):
        assert choose_image_ext("webp", None) == "webp"

    def test_explicit_jpeg_normalises(self):
        assert choose_image_ext("jpeg", None) == "jpg"

    def test_explicit_jpg(self):
        assert choose_image_ext("jpg", None) == "jpg"

    def test_transparent_background_defaults_png(self):
        assert choose_image_ext(None, "transparent") == "png"

    def test_opaque_background_defaults_jpg(self):
        assert choose_image_ext(None, "opaque") == "jpg"

    def test_no_args_defaults_jpg(self):
        assert choose_image_ext(None, None) == "jpg"

    def test_format_overrides_background(self):
        assert choose_image_ext("webp", "transparent") == "webp"


# ---------------------------------------------------------------------------
# merge_image_input_list
# ---------------------------------------------------------------------------


class TestMergeImageInputList:

    def test_none_inputs(self):
        assert merge_image_input_list(None, None) == []

    def test_single_item(self):
        assert merge_image_input_list("a") == ["a"]

    def test_list_input(self):
        assert merge_image_input_list(["a", "b"]) == ["a", "b"]

    def test_mixed_inputs(self):
        result = merge_image_input_list(None, ["x", "y"], "z")
        assert result == ["x", "y", "z"]

    def test_empty_list_ignored(self):
        assert merge_image_input_list([], "a") == ["a"]


# ---------------------------------------------------------------------------
# image_api._build_generation_kwargs
# ---------------------------------------------------------------------------


class TestImageBuildGenerationKwargs:

    @pytest.fixture(autouse=True)
    def _patch_output_dir(self, tmp_path):
        with patch(
                "fastvideo.entrypoints.openai.image_api.get_output_dir",
                return_value=str(tmp_path),
        ):
            yield tmp_path

    def _build(self, **overrides):
        from fastvideo.entrypoints.openai.image_api import (
            _build_generation_kwargs, )

        defaults = dict(request_id="req-1", prompt="a cat")
        defaults.update(overrides)
        return _build_generation_kwargs(**defaults)

    def test_output_path_under_images_subdir(self, tmp_path):
        kw = self._build()
        assert kw["output_path"].startswith(
            os.path.join(str(tmp_path), "images"))

    def test_num_frames_always_one(self):
        kw = self._build()
        assert kw["num_frames"] == 1

    def test_size_parsed(self):
        kw = self._build(size="640x480")
        assert kw["width"] == 640
        assert kw["height"] == 480

    def test_seed_forwarded(self):
        kw = self._build(seed=42)
        assert kw["seed"] == 42

    def test_n_clamped(self):
        kw = self._build(n=20)
        assert kw["num_videos_per_prompt"] == 10

    def test_extension_jpg_default(self):
        kw = self._build()
        assert kw["output_path"].endswith(".jpg")

    def test_extension_png_for_transparent(self):
        kw = self._build(background="transparent")
        assert kw["output_path"].endswith(".png")


# ---------------------------------------------------------------------------
# video_api._build_generation_kwargs
# ---------------------------------------------------------------------------


class TestVideoBuildGenerationKwargs:

    @pytest.fixture(autouse=True)
    def _patch_output_dir(self, tmp_path):
        with patch(
                "fastvideo.entrypoints.openai.video_api.get_output_dir",
                return_value=str(tmp_path),
        ):
            yield tmp_path

    def _build(self, **overrides):
        from fastvideo.entrypoints.openai.video_api import (
            _build_generation_kwargs, )

        defaults = dict(prompt="a running dog", seconds=4)
        defaults.update(overrides)
        req = VideoGenerationsRequest(**defaults)
        return _build_generation_kwargs("req-v1", req)

    def test_output_path_under_videos_subdir(self, tmp_path):
        kw = self._build()
        assert kw["output_path"].startswith(
            os.path.join(str(tmp_path), "videos"))
        assert kw["output_path"].endswith(".mp4")

    def test_fps_defaults_24(self):
        kw = self._build()
        assert kw["fps"] == 24

    def test_num_frames_from_seconds(self):
        kw = self._build(seconds=2, fps=30)
        assert kw["num_frames"] == 60

    def test_explicit_num_frames_overrides_seconds(self):
        kw = self._build(seconds=10, num_frames=5)
        assert kw["num_frames"] == 5

    def test_seed_forwarded(self):
        kw = self._build(seed=123)
        assert kw["seed"] == 123

    def test_custom_output_path_overrides_default(self, tmp_path):
        custom = str(tmp_path / "custom")
        kw = self._build(output_path=custom)
        assert kw["output_path"].startswith(custom)


# ---------------------------------------------------------------------------
# Protocol Pydantic models
# ---------------------------------------------------------------------------


class TestProtocolModels:

    def test_image_request_required_fields(self):
        req = ImageGenerationsRequest(prompt="test")
        assert req.prompt == "test"
        assert req.n == 1
        assert req.size == "1024x1024"

    def test_image_request_missing_prompt_raises(self):
        with pytest.raises(Exception):
            ImageGenerationsRequest()

    def test_video_request_defaults(self):
        req = VideoGenerationsRequest(prompt="hello")
        assert req.seconds == 4
        assert req.seed == 1024

    def test_video_response_defaults(self):
        resp = VideoResponse(id="v1")
        assert resp.status == "queued"
        assert resp.object == "video"

    def test_image_response_data_optional_fields(self):
        d = ImageResponseData()
        assert d.b64_json is None
        assert d.url is None

    def test_generate_request_id_unique(self):
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_video_list_response(self):
        resp = VideoListResponse(
            data=[VideoResponse(
                id="a"), VideoResponse(id="b")])
        assert len(resp.data) == 2
        assert resp.object == "list"
