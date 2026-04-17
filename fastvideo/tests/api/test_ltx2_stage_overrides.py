# SPDX-License-Identifier: Apache-2.0
"""Tests for typed LTX-2 stage override dataclasses."""
from __future__ import annotations

import pytest

from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.presets import get_preset, validate_stage_overrides
from fastvideo.pipelines.basic.ltx2.stage_overrides import (
    LTX2RefinePresetOverride,
    LTX2RefineStageOverride,
    refine_preset_override_fields,
    refine_preset_override_to_dict,
    refine_stage_override_fields,
    refine_stage_override_to_dict,
)


class TestRefineStageOverrideDataclass:

    def test_all_fields_default_to_none(self) -> None:
        override = LTX2RefineStageOverride()
        assert override.num_inference_steps is None
        assert override.guidance_scale is None
        assert override.image_crf is None
        assert override.video_position_offset_sec is None

    def test_explicit_construction(self) -> None:
        override = LTX2RefineStageOverride(
            num_inference_steps=2,
            guidance_scale=1.0,
            image_crf=18,
            video_position_offset_sec=2.5,
        )
        assert override.num_inference_steps == 2
        assert override.guidance_scale == 1.0
        assert override.image_crf == 18
        assert override.video_position_offset_sec == 2.5

    def test_to_dict_drops_none(self) -> None:
        override = LTX2RefineStageOverride(num_inference_steps=3)
        assert refine_stage_override_to_dict(override) == {
            "num_inference_steps": 3,
        }

    def test_to_dict_with_all_fields(self) -> None:
        override = LTX2RefineStageOverride(
            num_inference_steps=2,
            guidance_scale=1.0,
            image_crf=18,
            video_position_offset_sec=0.0,
        )
        assert refine_stage_override_to_dict(override) == {
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "image_crf": 18,
            "video_position_offset_sec": 0.0,
        }

    def test_fields_accessor_matches_dataclass(self) -> None:
        assert refine_stage_override_fields() == frozenset({
            "num_inference_steps",
            "guidance_scale",
            "image_crf",
            "video_position_offset_sec",
        })


class TestRefinePresetOverrideDataclass:

    def test_all_fields_default_to_none(self) -> None:
        override = LTX2RefinePresetOverride()
        assert override.enabled is None
        assert override.add_noise is None

    def test_to_dict_drops_none(self) -> None:
        override = LTX2RefinePresetOverride(enabled=True)
        assert refine_preset_override_to_dict(override) == {
            "enabled": True,
        }

    def test_to_dict_with_all_fields(self) -> None:
        override = LTX2RefinePresetOverride(enabled=True, add_noise=False)
        assert refine_preset_override_to_dict(override) == {
            "enabled": True,
            "add_noise": False,
        }

    def test_fields_accessor_matches_dataclass(self) -> None:
        assert refine_preset_override_fields() == frozenset({
            "enabled",
            "add_noise",
        })


class TestStageOverridesMirrorPresetSchema:
    """The ltx2_two_stage preset's refine stage schema must list
    exactly the :class:`LTX2RefineStageOverride` field names."""

    def test_allowed_overrides_mirror_dataclass(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("ltx2_two_stage", "ltx2")
        refine_schema = next(
            s for s in preset.stage_schemas if s.name == "refine")
        assert refine_schema.allowed_overrides == refine_stage_override_fields()

    def test_roundtrip_through_validate_stage_overrides(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("ltx2_two_stage", "ltx2")
        override = LTX2RefineStageOverride(
            num_inference_steps=3,
            guidance_scale=1.0,
        )
        validate_stage_overrides(
            preset, {"refine": refine_stage_override_to_dict(override)})

    def test_unknown_field_rejected(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("ltx2_two_stage", "ltx2")
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                preset, {"refine": {"unknown_key": 1}})
