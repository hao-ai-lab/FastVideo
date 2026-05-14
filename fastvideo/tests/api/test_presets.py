# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.presets import (
    InferencePreset,
    PresetStageSpec,
    get_all_preset_names,
    get_preset,
    get_presets_for_family,
    register_preset,
    validate_preset_selection,
    validate_stage_names,
    validate_stage_overrides,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture()
def _isolated_registry():
    """Run each test with an empty preset registry, restoring after."""
    from fastvideo.api.presets import _PRESET_REGISTRY
    saved = dict(_PRESET_REGISTRY)
    _PRESET_REGISTRY.clear()
    yield
    _PRESET_REGISTRY.clear()
    _PRESET_REGISTRY.update(saved)


_SIMPLE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

_SR_STAGE = PresetStageSpec(
    name="sr",
    kind="super_resolution",
    allowed_overrides=frozenset({"height_sr", "width_sr"}),
)

_NO_OVERRIDES_STAGE = PresetStageSpec(
    name="encode",
    kind="text_encoding",
)


def _make_preset(
    name: str = "test_preset",
    version: int = 1,
    model_family: str = "test",
    stage_schemas: tuple[PresetStageSpec, ...] = (_SIMPLE_STAGE, ),
    **kwargs,
) -> InferencePreset:
    return InferencePreset(
        name=name,
        version=version,
        model_family=model_family,
        stage_schemas=stage_schemas,
        **kwargs,
    )


# -------------------------------------------------------------------
# Registration and lookup
# -------------------------------------------------------------------


class TestRegistration:

    @pytest.mark.usefixtures("_isolated_registry")
    def test_register_and_get(self) -> None:
        p = _make_preset()
        register_preset(p)
        assert get_preset("test_preset", "test") is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_with_explicit_version(self) -> None:
        p = _make_preset(version=2)
        register_preset(p)
        assert get_preset("test_preset", "test", version=2) is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_latest_version(self) -> None:
        p1 = _make_preset(version=1)
        p2 = _make_preset(version=2)
        register_preset(p1)
        register_preset(p2)
        assert get_preset("test_preset", "test") is p2

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_missing_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown preset"):
            get_preset("nope", "test")

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_wrong_version_raises(self) -> None:
        register_preset(_make_preset(version=1))
        with pytest.raises(ConfigValidationError, match="version"):
            get_preset("test_preset", "test", version=99)

    @pytest.mark.usefixtures("_isolated_registry")
    def test_duplicate_raises(self) -> None:
        register_preset(_make_preset())
        with pytest.raises(ValueError, match="Duplicate"):
            register_preset(_make_preset())

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_presets_for_family(self) -> None:
        register_preset(_make_preset(name="a"))
        register_preset(_make_preset(name="b"))
        register_preset(_make_preset(
            name="c", model_family="other"))
        result = get_presets_for_family("test")
        assert {p.name for p in result} == {"a", "b"}

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_all_preset_names(self) -> None:
        register_preset(_make_preset(name="beta"))
        register_preset(_make_preset(name="alpha"))
        assert get_all_preset_names() == ["alpha", "beta"]


# -------------------------------------------------------------------
# Stage-name validation
# -------------------------------------------------------------------


class TestStageNameValidation:

    def test_valid_stage_name_passes(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, _SR_STAGE))
        validate_stage_names(
            preset, {"denoise": {}, "sr": {}})

    def test_unknown_stage_name_raises(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError, match="stage_overrides.bogus"
        ):
            validate_stage_names(preset, {"bogus": {}})

    def test_empty_overrides_passes(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        validate_stage_names(preset, {})

    def test_error_lists_valid_stages(self) -> None:
        preset = _make_preset(
            stage_schemas=(_SIMPLE_STAGE, _SR_STAGE))
        with pytest.raises(
            ConfigValidationError, match="'denoise'"
        ):
            validate_stage_names(preset, {"nope": {}})


# -------------------------------------------------------------------
# Stage-override validation
# -------------------------------------------------------------------


class TestStageOverrideValidation:

    def test_allowed_override_passes(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        validate_stage_overrides(
            preset,
            {"denoise": {"num_inference_steps": 25}},
        )

    def test_disallowed_override_raises(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError,
            match="stage_overrides.denoise.height",
        ):
            validate_stage_overrides(
                preset,
                {"denoise": {"height": 720}},
            )

    def test_override_on_stage_with_no_allowed_raises(self) -> None:
        preset = _make_preset(stage_schemas=(_NO_OVERRIDES_STAGE, ))
        with pytest.raises(
            ConfigValidationError,
            match="does not accept overrides",
        ):
            validate_stage_overrides(
                preset,
                {"encode": {"some_key": 1}},
            )

    def test_empty_override_on_no_allowed_passes(self) -> None:
        preset = _make_preset(stage_schemas=(_NO_OVERRIDES_STAGE, ))
        validate_stage_overrides(preset, {"encode": {}})

    def test_non_mapping_override_raises(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        with pytest.raises(ConfigValidationError, match="mapping"):
            validate_stage_overrides(
                preset, {"denoise": "not a dict"})

    def test_unknown_stage_still_caught(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        with pytest.raises(ConfigValidationError, match="unknown"):
            validate_stage_overrides(
                preset, {"missing_stage": {"a": 1}})

    def test_error_lists_allowed_overrides(self) -> None:
        preset = _make_preset(stage_schemas=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError, match="guidance_scale"
        ):
            validate_stage_overrides(
                preset,
                {"denoise": {"bad_key": 1}},
            )


# -------------------------------------------------------------------
# validate_preset_selection end-to-end
# -------------------------------------------------------------------


class TestValidatePresetSelection:

    @pytest.mark.usefixtures("_isolated_registry")
    def test_none_preset_returns_none(self) -> None:
        assert validate_preset_selection(
            None, "test") is None

    @pytest.mark.usefixtures("_isolated_registry")
    def test_valid_preset_resolves(self) -> None:
        p = _make_preset()
        register_preset(p)
        result = validate_preset_selection(
            "test_preset", "test")
        assert result is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_valid_preset_with_overrides(self) -> None:
        p = _make_preset()
        register_preset(p)
        result = validate_preset_selection(
            "test_preset",
            "test",
            stage_overrides={"denoise": {"guidance_scale": 2.0}},
        )
        assert result is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown"):
            validate_preset_selection("nope", "test")

    @pytest.mark.usefixtures("_isolated_registry")
    def test_bad_stage_override_raises(self) -> None:
        register_preset(_make_preset())
        with pytest.raises(ConfigValidationError):
            validate_preset_selection(
                "test_preset",
                "test",
                stage_overrides={"denoise": {"bad": 1}},
            )


# -------------------------------------------------------------------
# Wan preset integration (uses real registry)
# -------------------------------------------------------------------


class TestWanPresets:
    """Verify the Wan presets registered from registry.py."""

    def test_wan_presets_are_registered(self) -> None:
        # Force registration by importing registry.
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("wan")
        names = {p.name for p in presets}
        assert "wan_t2v_1_3b" in names
        assert "wan_t2v_14b" in names
        assert "wan_i2v_14b_480p" in names
        assert "wan_2_2_t2v_a14b" in names

    def test_wan_t2v_1_3b_lookup(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("wan_t2v_1_3b", "wan")
        assert preset.model_family == "wan"
        assert preset.workload_type == "t2v"
        assert len(preset.stage_schemas) == 1
        assert preset.stage_schemas[0].name == "denoise"
        assert preset.defaults["height"] == 480
        assert preset.defaults["width"] == 832

    def test_wan_2_2_allows_dual_guidance(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("wan_2_2_t2v_a14b", "wan")
        stage = preset.stage_schemas[0]
        assert "guidance_scale_2" in stage.allowed_overrides
        assert "boundary_ratio" in stage.allowed_overrides

    def test_wan_stage_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        preset = get_preset("wan_t2v_14b", "wan")
        # Valid override.
        validate_stage_overrides(
            preset,
            {"denoise": {"num_inference_steps": 25}},
        )
        # Invalid override key.
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                preset,
                {"denoise": {"height": 1080}},
            )

    def test_wan_model_family_in_registry(self) -> None:
        from fastvideo.registry import get_model_family
        family = get_model_family(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        assert family == "wan"


# -------------------------------------------------------------------
# LTX2 preset integration
# -------------------------------------------------------------------


class TestLtx2Presets:

    def test_ltx2_presets_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("ltx2")
        names = {p.name for p in presets}
        assert names == {"ltx2_base", "ltx2_distilled", "ltx2_two_stage"}

    def test_ltx2_base_lookup(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("ltx2_base", "ltx2")
        assert p.workload_type == "t2v"
        assert p.defaults["height"] == 512
        assert p.defaults["width"] == 768

    def test_ltx2_distilled_fewer_steps(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("ltx2_distilled", "ltx2")
        assert p.defaults["num_inference_steps"] == 8
        assert p.defaults["guidance_scale"] == 1.0

    def test_ltx2_two_stage_is_two_stage(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("ltx2_two_stage", "ltx2")
        assert len(p.stage_schemas) == 2
        assert p.stage_schemas[0].name == "denoise"
        assert p.stage_schemas[1].name == "refine"
        assert p.stage_schemas[1].kind == "refinement"

    def test_ltx2_two_stage_stage_defaults(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("ltx2_two_stage", "ltx2")
        refine = p.stage_defaults["refine"]
        # stage-2 refine only supports 2 or 3 denoising steps; preset
        # defaults to 2 (matches gpu_pool.py load_kwargs).
        assert refine["num_inference_steps"] == 2
        assert refine["guidance_scale"] == 1.0

    def test_ltx2_two_stage_refine_overrides_valid(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("ltx2_two_stage", "ltx2")
        validate_stage_overrides(
            p, {"refine": {"num_inference_steps": 3}})
        validate_stage_overrides(
            p, {"refine": {"guidance_scale": 1.0}})
        validate_stage_overrides(
            p, {"refine": {"image_crf": 18}})
        validate_stage_overrides(
            p, {"refine": {"video_position_offset_sec": 2.5}})

    def test_ltx2_two_stage_rejects_unknown_refine_override(self) -> None:
        import fastvideo.registry  # noqa: F401
        from fastvideo.api.errors import ConfigValidationError
        p = get_preset("ltx2_two_stage", "ltx2")
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                p, {"refine": {"bogus_field": 1}})


# -------------------------------------------------------------------
# Hunyuan preset integration
# -------------------------------------------------------------------


class TestHunyuanPresets:

    def test_hunyuan_presets_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("hunyuan")
        names = {p.name for p in presets}
        assert names == {"hunyuan_t2v", "fast_hunyuan_t2v"}

    def test_fast_hunyuan_fewer_steps(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("fast_hunyuan_t2v", "hunyuan")
        assert p.defaults["num_inference_steps"] == 6


# -------------------------------------------------------------------
# Hunyuan15 preset integration (includes two-stage SR)
# -------------------------------------------------------------------


class TestHunyuan15Presets:

    def test_hunyuan15_presets_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("hunyuan15")
        assert len(presets) == 5

    def test_hunyuan15_sr_is_two_stage(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("hunyuan15_sr_1080p", "hunyuan15")
        assert len(p.stage_schemas) == 2
        assert p.stage_schemas[0].name == "denoise"
        assert p.stage_schemas[1].name == "sr"
        assert p.stage_schemas[1].kind == "super_resolution"

    def test_hunyuan15_sr_stage_defaults(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("hunyuan15_sr_1080p", "hunyuan15")
        sr = p.stage_defaults["sr"]
        assert sr["height_sr"] == 1072
        assert sr["width_sr"] == 1920

    def test_hunyuan15_sr_stage_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("hunyuan15_sr_1080p", "hunyuan15")
        # Valid: override sr num_inference_steps.
        validate_stage_overrides(
            p, {"sr": {"num_inference_steps": 12}})
        # Invalid: height not in sr allowed_overrides.
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                p, {"sr": {"height": 1080}})


# -------------------------------------------------------------------
# Cosmos / Cosmos25 preset integration
# -------------------------------------------------------------------


class TestCosmosPresets:

    def test_cosmos_preset_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("cosmos")
        assert len(presets) == 1
        assert presets[0].name == "cosmos_predict2_2b"

    def test_cosmos25_separate_family(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("cosmos25")
        assert len(presets) == 1
        assert presets[0].name == "cosmos25_predict2_2b"

    def test_cosmos_and_cosmos25_different_fps(self) -> None:
        import fastvideo.registry  # noqa: F401
        c = get_preset("cosmos_predict2_2b", "cosmos")
        c25 = get_preset("cosmos25_predict2_2b", "cosmos25")
        assert c.defaults["fps"] == 16
        assert c25.defaults["fps"] == 24


# -------------------------------------------------------------------
# TurboDiffusion preset integration
# -------------------------------------------------------------------


class TestTurboDiffusionPresets:

    def test_turbo_presets_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("turbodiffusion")
        names = {p.name for p in presets}
        assert names == {
            "turbo_t2v_1_3b",
            "turbo_t2v_14b",
            "turbo_i2v_a14b",
        }

    def test_turbo_4_step(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("turbo_t2v_14b", "turbodiffusion")
        assert p.defaults["num_inference_steps"] == 4
        assert p.defaults["guidance_scale"] == 1.0


# -------------------------------------------------------------------
# SD35 preset integration
# -------------------------------------------------------------------


class TestSD35Presets:

    def test_sd35_preset_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("sd35_medium", "sd35")
        assert p.workload_type == "t2i"
        assert p.defaults["height"] == 512
        assert p.defaults["num_frames"] == 1


# -------------------------------------------------------------------
# LingBotWorld preset integration (dual guidance)
# -------------------------------------------------------------------


class TestLingBotWorldPresets:

    def test_lingbotworld_dual_guidance(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("lingbotworld_i2v", "lingbotworld")
        stage = p.stage_schemas[0]
        assert "guidance_scale_2" in stage.allowed_overrides
        assert "boundary_ratio" in stage.allowed_overrides

    def test_lingbotworld_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("lingbotworld_i2v", "lingbotworld")
        validate_stage_overrides(
            p, {"denoise": {"boundary_ratio": 0.95}})
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                p, {"denoise": {"height": 720}})


# -------------------------------------------------------------------
# Remaining single-preset families
# -------------------------------------------------------------------


class TestSinglePresetFamilies:

    def test_hyworld_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("hyworld_t2v", "hyworld")
        assert p.workload_type == "t2v"

    def test_gamecraft_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("gamecraft_i2v", "gamecraft")
        assert p.workload_type == "i2v"
        assert p.defaults["num_frames"] == 33

    def test_gen3c_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("gen3c_cosmos_7b", "gen3c")
        assert p.defaults["num_inference_steps"] == 35

    def test_matrixgame_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_preset("matrixgame_i2v", "matrixgame")
        assert p.defaults["num_inference_steps"] == 3
        assert p.defaults["fps"] == 25

    def test_longcat_presets_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        presets = get_presets_for_family("longcat")
        names = {p.name for p in presets}
        assert names == {
            "longcat_t2v", "longcat_i2v", "longcat_vc"
        }


# -------------------------------------------------------------------
# Cross-family: total preset count
# -------------------------------------------------------------------


class TestPresetCountIntegrity:

    def test_total_preset_count(self) -> None:
        """At least the baseline 37 presets from 13 families are registered."""
        import fastvideo.registry  # noqa: F401
        names = get_all_preset_names()
        assert len(names) >= 37


class TestPresetDefaultTypes:
    """Preset ``defaults`` values must match the types on
    :class:`SamplingParam`. Assigning ``None`` to a typed-``str`` field
    (e.g. ``negative_prompt``) breaks downstream stages that assert the
    runtime type — see the CFG branch in
    ``pipelines/stages/text_encoding.py:81``."""

    def test_ltx2_cfg_defaults_are_off(self) -> None:
        """SamplingParam's LTX-2 CFG class defaults must be 1.0 (CFG
        off). ``ForwardBatch.__post_init__`` force-enables
        ``do_classifier_free_guidance`` when either
        ``ltx2_cfg_scale_video`` or ``ltx2_cfg_scale_audio`` is != 1.0,
        so any non-1.0 default silently forces CFG on for every model
        family that doesn't explicitly override these fields. Guard
        against the regression that surfaced as the TurboDiffusion I2V
        SSIM crash (``text_encoding.py:81`` assertion on
        ``negative_prompt``)."""
        from fastvideo.api.sampling_param import SamplingParam
        sp = SamplingParam()
        assert sp.ltx2_cfg_scale_video == 1.0
        assert sp.ltx2_cfg_scale_audio == 1.0

    def test_no_preset_sets_negative_prompt_to_none(self) -> None:
        import fastvideo.registry  # noqa: F401
        from fastvideo.api.presets import _PRESET_REGISTRY
        offenders = [
            f"{preset.model_family}/{preset.name}"
            for preset in _PRESET_REGISTRY.values()
            if preset.defaults.get("negative_prompt", "") is None
        ]
        assert not offenders, (
            "These presets set negative_prompt=None, which violates "
            "SamplingParam.negative_prompt's typed str contract and "
            "crashes the CFG path in text_encoding. Use \"\" instead:\n"
            + "\n".join(f"  - {p}" for p in offenders))
