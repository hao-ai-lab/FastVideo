"""Unit tests for the OpenAI-compatible API server helpers (no GPU needed)."""

import os
from unittest.mock import patch

import pytest

from fastvideo.api.parser import parse_config
from fastvideo.api.schema import GenerationRequest
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
# video_api._build_generation_kwargs with ServeConfig.default_request
# ---------------------------------------------------------------------------


def _make_default_request(raw: dict) -> GenerationRequest:
    """Parse a raw config dict into a tracked GenerationRequest."""
    return parse_config(GenerationRequest, raw)


class TestVideoDefaultRequestMerge:

    @pytest.fixture(autouse=True)
    def _patch_output_dir(self, tmp_path):
        with patch(
                "fastvideo.entrypoints.openai.video_api.get_output_dir",
                return_value=str(tmp_path),
        ):
            yield tmp_path

    def _build(self, default_raw=None, **body_overrides):
        from fastvideo.entrypoints.openai.video_api import (
            _build_generation_kwargs, )

        body_defaults = dict(prompt="a running dog", seconds=4)
        body_defaults.update(body_overrides)
        req = VideoGenerationsRequest(**body_defaults)
        default_request = _make_default_request(
            default_raw) if default_raw else None
        return _build_generation_kwargs(
            "req-v1", req, default_request=default_request)

    def test_default_seed_flows_through_when_body_omits(self):
        kw = self._build(default_raw={"sampling": {"seed": 42}})
        assert kw["seed"] == 42

    def test_body_seed_overrides_default(self):
        kw = self._build(
            default_raw={"sampling": {
                "seed": 42
            }},
            seed=7,
        )
        assert kw["seed"] == 7

    def test_default_fps_used_for_num_frames_from_seconds(self):
        # Default fps=30, body only provides seconds=2 -> num_frames=60.
        kw = self._build(default_raw={"sampling": {"fps": 30}}, seconds=2)
        assert kw["fps"] == 30
        assert kw["num_frames"] == 60

    def test_default_guidance_scale_preserved_when_body_omits(self):
        kw = self._build(default_raw={"sampling": {"guidance_scale": 5.5}})
        assert kw["guidance_scale"] == 5.5

    def test_body_guidance_scale_overrides_default(self):
        kw = self._build(
            default_raw={"sampling": {
                "guidance_scale": 5.5
            }},
            guidance_scale=9.0,
        )
        assert kw["guidance_scale"] == 9.0

    def test_body_size_overrides_default_sampling_dims(self):
        kw = self._build(
            default_raw={
                "sampling": {
                    "width": 640,
                    "height": 360
                }
            },
            size="1024x576",
        )
        assert kw["width"] == 1024
        assert kw["height"] == 576

    def test_default_width_height_preserved_when_body_omits_size(self):
        kw = self._build(default_raw={
            "sampling": {
                "width": 640,
                "height": 360
            }
        })
        assert kw["width"] == 640
        assert kw["height"] == 360

    def test_default_output_path_used_as_output_dir(self, tmp_path):
        custom = str(tmp_path / "from_default")
        kw = self._build(default_raw={"output": {"output_path": custom}})
        assert kw["output_path"].startswith(custom)

    def test_body_output_path_overrides_default(self, tmp_path):
        body_dir = str(tmp_path / "body")
        default_dir = str(tmp_path / "default")
        kw = self._build(
            default_raw={"output": {
                "output_path": default_dir
            }},
            output_path=body_dir,
        )
        assert kw["output_path"].startswith(body_dir)
        assert default_dir not in kw["output_path"]

    def test_default_negative_prompt_flows_through(self):
        kw = self._build(
            default_raw={"negative_prompt": "low quality, blur"})
        assert kw["negative_prompt"] == "low quality, blur"

    def test_body_negative_prompt_overrides_default(self):
        kw = self._build(
            default_raw={"negative_prompt": "low quality"},
            negative_prompt="watermark",
        )
        assert kw["negative_prompt"] == "watermark"

    def test_no_default_request_behaves_like_before(self):
        kw = self._build(seed=123, fps=24)
        assert kw["seed"] == 123
        assert kw["fps"] == 24

    def test_default_request_not_mutated_by_build(self):
        # Merge should operate on a fresh copy (caller supplies a clone);
        # the helper itself must not mutate the passed-in default.
        default_request = _make_default_request(
            {"sampling": {
                "seed": 42,
                "fps": 30
            }})
        from fastvideo.entrypoints.openai.video_api import (
            _build_generation_kwargs, )
        req = VideoGenerationsRequest(prompt="p", seconds=1)
        _ = _build_generation_kwargs(
            "req-1", req, default_request=default_request)
        assert default_request.sampling.seed == 42
        assert default_request.sampling.fps == 30


# ---------------------------------------------------------------------------
# Preset stage-override validation
# ---------------------------------------------------------------------------


class TestValidateDefaultRequestAgainstPreset:
    """Startup-time validation lives in api_server._validate_default_request_against_preset.

    Called once by ``run_server`` before the FastAPI app is created — the
    default_request is static server config, so per-request re-validation
    would be pure overhead.
    """

    def test_empty_stage_overrides_is_noop(self):
        from fastvideo.entrypoints.openai.api_server import (
            _validate_default_request_against_preset, )
        default_request = _make_default_request({
            "sampling": {
                "seed": 42
            }
        })
        _validate_default_request_against_preset(default_request, "any/model")

    def test_unknown_model_path_is_noop(self):
        from fastvideo.entrypoints.openai.api_server import (
            _validate_default_request_against_preset, )
        default_request = _make_default_request({
            "stage_overrides": {
                "denoise": {
                    "num_inference_steps": 10
                }
            }
        })
        with patch(
                "fastvideo.entrypoints.openai.api_server.get_preset_selection",
                return_value=(None, None),
        ):
            _validate_default_request_against_preset(
                default_request, "unknown/model")

    def test_unknown_stage_name_raises(self):
        from fastvideo.api.errors import ConfigValidationError
        from fastvideo.entrypoints.openai.api_server import (
            _validate_default_request_against_preset, )
        default_request = _make_default_request({
            "stage_overrides": {
                "not_a_real_stage": {
                    "num_inference_steps": 10
                }
            }
        })
        with patch(
                "fastvideo.entrypoints.openai.api_server.get_preset_selection",
                return_value=("wan_t2v_1_3b", "wan"),
        ):
            with pytest.raises(ConfigValidationError):
                _validate_default_request_against_preset(
                    default_request, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")


# ---------------------------------------------------------------------------
# Server state accessors
# ---------------------------------------------------------------------------


class TestDefaultRequestState:

    def test_set_and_get_default_request(self):
        from fastvideo.entrypoints.openai import state as state_mod
        saved = state_mod._default_request
        try:
            dr = _make_default_request({"sampling": {"seed": 7}})
            state_mod.set_state.__wrapped__ if False else None  # keep lint happy
            state_mod._default_request = dr
            assert state_mod.get_default_request() is dr
        finally:
            state_mod._default_request = saved

    def test_clear_state_resets_default_request(self):
        from fastvideo.entrypoints.openai import state as state_mod
        saved = state_mod._default_request
        try:
            state_mod._default_request = _make_default_request(
                {"sampling": {
                    "seed": 7
                }})
            state_mod.clear_state()
            assert state_mod.get_default_request() is None
        finally:
            state_mod._default_request = saved


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
