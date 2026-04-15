# SPDX-License-Identifier: Apache-2.0
import pytest

from fastvideo.api.errors import ConfigValidationError
from fastvideo.api.profiles import (
    PipelineProfile,
    ProfileStageSpec,
    _clear_registry,
    get_all_profile_names,
    get_profile,
    get_profiles_for_family,
    register_profile,
    validate_profile_selection,
    validate_stage_names,
    validate_stage_overrides,
)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture()
def _isolated_registry():
    """Run each test with an empty profile registry, restoring after."""
    from fastvideo.api.profiles import _PROFILE_REGISTRY
    saved = dict(_PROFILE_REGISTRY)
    _PROFILE_REGISTRY.clear()
    yield
    _PROFILE_REGISTRY.clear()
    _PROFILE_REGISTRY.update(saved)


_SIMPLE_STAGE = ProfileStageSpec(
    name="denoise",
    kind="denoising",
    allowed_overrides=frozenset({"num_inference_steps", "guidance_scale"}),
)

_SR_STAGE = ProfileStageSpec(
    name="sr",
    kind="super_resolution",
    allowed_overrides=frozenset({"height_sr", "width_sr"}),
)

_NO_OVERRIDES_STAGE = ProfileStageSpec(
    name="encode",
    kind="text_encoding",
)


def _make_profile(
    name: str = "test_profile",
    version: str = "1",
    model_family: str = "test",
    stages: tuple[ProfileStageSpec, ...] = (_SIMPLE_STAGE, ),
    **kwargs,
) -> PipelineProfile:
    return PipelineProfile(
        name=name,
        version=version,
        model_family=model_family,
        stages=stages,
        **kwargs,
    )


# -------------------------------------------------------------------
# Registration and lookup
# -------------------------------------------------------------------


class TestRegistration:

    @pytest.mark.usefixtures("_isolated_registry")
    def test_register_and_get(self) -> None:
        p = _make_profile()
        register_profile(p)
        assert get_profile("test_profile", "test") is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_with_explicit_version(self) -> None:
        p = _make_profile(version="2")
        register_profile(p)
        assert get_profile("test_profile", "test", version="2") is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_latest_version(self) -> None:
        p1 = _make_profile(version="1")
        p2 = _make_profile(version="2")
        register_profile(p1)
        register_profile(p2)
        assert get_profile("test_profile", "test") is p2

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_missing_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown profile"):
            get_profile("nope", "test")

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_wrong_version_raises(self) -> None:
        register_profile(_make_profile(version="1"))
        with pytest.raises(ConfigValidationError, match="version"):
            get_profile("test_profile", "test", version="99")

    @pytest.mark.usefixtures("_isolated_registry")
    def test_duplicate_raises(self) -> None:
        register_profile(_make_profile())
        with pytest.raises(ValueError, match="Duplicate"):
            register_profile(_make_profile())

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_profiles_for_family(self) -> None:
        register_profile(_make_profile(name="a"))
        register_profile(_make_profile(name="b"))
        register_profile(_make_profile(
            name="c", model_family="other"))
        result = get_profiles_for_family("test")
        assert {p.name for p in result} == {"a", "b"}

    @pytest.mark.usefixtures("_isolated_registry")
    def test_get_all_profile_names(self) -> None:
        register_profile(_make_profile(name="beta"))
        register_profile(_make_profile(name="alpha"))
        assert get_all_profile_names() == ["alpha", "beta"]


# -------------------------------------------------------------------
# Stage-name validation
# -------------------------------------------------------------------


class TestStageNameValidation:

    def test_valid_stage_name_passes(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, _SR_STAGE))
        validate_stage_names(
            profile, {"denoise": {}, "sr": {}})

    def test_unknown_stage_name_raises(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError, match="stage_overrides.bogus"
        ):
            validate_stage_names(profile, {"bogus": {}})

    def test_empty_overrides_passes(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        validate_stage_names(profile, {})

    def test_error_lists_valid_stages(self) -> None:
        profile = _make_profile(
            stages=(_SIMPLE_STAGE, _SR_STAGE))
        with pytest.raises(
            ConfigValidationError, match="'denoise'"
        ):
            validate_stage_names(profile, {"nope": {}})


# -------------------------------------------------------------------
# Stage-override validation
# -------------------------------------------------------------------


class TestStageOverrideValidation:

    def test_allowed_override_passes(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        validate_stage_overrides(
            profile,
            {"denoise": {"num_inference_steps": 25}},
        )

    def test_disallowed_override_raises(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError,
            match="stage_overrides.denoise.height",
        ):
            validate_stage_overrides(
                profile,
                {"denoise": {"height": 720}},
            )

    def test_override_on_stage_with_no_allowed_raises(self) -> None:
        profile = _make_profile(stages=(_NO_OVERRIDES_STAGE, ))
        with pytest.raises(
            ConfigValidationError,
            match="does not accept overrides",
        ):
            validate_stage_overrides(
                profile,
                {"encode": {"some_key": 1}},
            )

    def test_empty_override_on_no_allowed_passes(self) -> None:
        profile = _make_profile(stages=(_NO_OVERRIDES_STAGE, ))
        validate_stage_overrides(profile, {"encode": {}})

    def test_non_mapping_override_raises(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        with pytest.raises(ConfigValidationError, match="mapping"):
            validate_stage_overrides(
                profile, {"denoise": "not a dict"})

    def test_unknown_stage_still_caught(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        with pytest.raises(ConfigValidationError, match="unknown"):
            validate_stage_overrides(
                profile, {"missing_stage": {"a": 1}})

    def test_error_lists_allowed_overrides(self) -> None:
        profile = _make_profile(stages=(_SIMPLE_STAGE, ))
        with pytest.raises(
            ConfigValidationError, match="guidance_scale"
        ):
            validate_stage_overrides(
                profile,
                {"denoise": {"bad_key": 1}},
            )


# -------------------------------------------------------------------
# validate_profile_selection end-to-end
# -------------------------------------------------------------------


class TestValidateProfileSelection:

    @pytest.mark.usefixtures("_isolated_registry")
    def test_none_profile_returns_none(self) -> None:
        assert validate_profile_selection(
            None, "test") is None

    @pytest.mark.usefixtures("_isolated_registry")
    def test_valid_profile_resolves(self) -> None:
        p = _make_profile()
        register_profile(p)
        result = validate_profile_selection(
            "test_profile", "test")
        assert result is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_valid_profile_with_overrides(self) -> None:
        p = _make_profile()
        register_profile(p)
        result = validate_profile_selection(
            "test_profile",
            "test",
            stage_overrides={"denoise": {"guidance_scale": 2.0}},
        )
        assert result is p

    @pytest.mark.usefixtures("_isolated_registry")
    def test_invalid_profile_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="unknown"):
            validate_profile_selection("nope", "test")

    @pytest.mark.usefixtures("_isolated_registry")
    def test_bad_stage_override_raises(self) -> None:
        register_profile(_make_profile())
        with pytest.raises(ConfigValidationError):
            validate_profile_selection(
                "test_profile",
                "test",
                stage_overrides={"denoise": {"bad": 1}},
            )


# -------------------------------------------------------------------
# Wan profile integration (uses real registry)
# -------------------------------------------------------------------


class TestWanProfiles:
    """Verify the Wan profiles registered from registry.py."""

    def test_wan_profiles_are_registered(self) -> None:
        # Force registration by importing registry.
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("wan")
        names = {p.name for p in profiles}
        assert "wan_t2v_1_3b" in names
        assert "wan_t2v_14b" in names
        assert "wan_i2v_14b_480p" in names
        assert "wan_2_2_t2v_a14b" in names

    def test_wan_t2v_1_3b_lookup(self) -> None:
        import fastvideo.registry  # noqa: F401
        profile = get_profile("wan_t2v_1_3b", "wan")
        assert profile.model_family == "wan"
        assert profile.workload_type == "t2v"
        assert len(profile.stages) == 1
        assert profile.stages[0].name == "denoise"
        assert profile.defaults["height"] == 480
        assert profile.defaults["width"] == 832

    def test_wan_2_2_allows_dual_guidance(self) -> None:
        import fastvideo.registry  # noqa: F401
        profile = get_profile("wan_2_2_t2v_a14b", "wan")
        stage = profile.stages[0]
        assert "guidance_scale_2" in stage.allowed_overrides
        assert "boundary_ratio" in stage.allowed_overrides

    def test_wan_stage_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        profile = get_profile("wan_t2v_14b", "wan")
        # Valid override.
        validate_stage_overrides(
            profile,
            {"denoise": {"num_inference_steps": 25}},
        )
        # Invalid override key.
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                profile,
                {"denoise": {"height": 1080}},
            )

    def test_wan_model_family_in_registry(self) -> None:
        from fastvideo.registry import get_model_family
        family = get_model_family(
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
        assert family == "wan"


# -------------------------------------------------------------------
# LTX2 profile integration
# -------------------------------------------------------------------


class TestLtx2Profiles:

    def test_ltx2_profiles_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("ltx2")
        names = {p.name for p in profiles}
        assert names == {"ltx2_base", "ltx2_distilled"}

    def test_ltx2_base_lookup(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("ltx2_base", "ltx2")
        assert p.workload_type == "t2v"
        assert p.defaults["height"] == 512
        assert p.defaults["width"] == 768

    def test_ltx2_distilled_fewer_steps(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("ltx2_distilled", "ltx2")
        assert p.defaults["num_inference_steps"] == 8
        assert p.defaults["guidance_scale"] == 1.0


# -------------------------------------------------------------------
# Hunyuan profile integration
# -------------------------------------------------------------------


class TestHunyuanProfiles:

    def test_hunyuan_profiles_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("hunyuan")
        names = {p.name for p in profiles}
        assert names == {"hunyuan_t2v", "fast_hunyuan_t2v"}

    def test_fast_hunyuan_fewer_steps(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("fast_hunyuan_t2v", "hunyuan")
        assert p.defaults["num_inference_steps"] == 6


# -------------------------------------------------------------------
# Hunyuan15 profile integration (includes two-stage SR)
# -------------------------------------------------------------------


class TestHunyuan15Profiles:

    def test_hunyuan15_profiles_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("hunyuan15")
        assert len(profiles) == 5

    def test_hunyuan15_sr_is_two_stage(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("hunyuan15_sr_1080p", "hunyuan15")
        assert len(p.stages) == 2
        assert p.stages[0].name == "denoise"
        assert p.stages[1].name == "sr"
        assert p.stages[1].kind == "super_resolution"

    def test_hunyuan15_sr_stage_defaults(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("hunyuan15_sr_1080p", "hunyuan15")
        sr = p.stage_defaults["sr"]
        assert sr["height_sr"] == 1072
        assert sr["width_sr"] == 1920

    def test_hunyuan15_sr_stage_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("hunyuan15_sr_1080p", "hunyuan15")
        # Valid: override sr num_inference_steps.
        validate_stage_overrides(
            p, {"sr": {"num_inference_steps": 12}})
        # Invalid: height not in sr allowed_overrides.
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                p, {"sr": {"height": 1080}})


# -------------------------------------------------------------------
# Cosmos / Cosmos25 profile integration
# -------------------------------------------------------------------


class TestCosmosProfiles:

    def test_cosmos_profile_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("cosmos")
        assert len(profiles) == 1
        assert profiles[0].name == "cosmos_predict2_2b"

    def test_cosmos25_separate_family(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("cosmos25")
        assert len(profiles) == 1
        assert profiles[0].name == "cosmos25_predict2_2b"

    def test_cosmos_and_cosmos25_different_fps(self) -> None:
        import fastvideo.registry  # noqa: F401
        c = get_profile("cosmos_predict2_2b", "cosmos")
        c25 = get_profile("cosmos25_predict2_2b", "cosmos25")
        assert c.defaults["fps"] == 16
        assert c25.defaults["fps"] == 24


# -------------------------------------------------------------------
# TurboDiffusion profile integration
# -------------------------------------------------------------------


class TestTurboDiffusionProfiles:

    def test_turbo_profiles_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("turbodiffusion")
        names = {p.name for p in profiles}
        assert names == {
            "turbo_t2v_1_3b",
            "turbo_t2v_14b",
            "turbo_i2v_a14b",
        }

    def test_turbo_4_step(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("turbo_t2v_14b", "turbodiffusion")
        assert p.defaults["num_inference_steps"] == 4
        assert p.defaults["guidance_scale"] == 1.0


# -------------------------------------------------------------------
# SD35 profile integration
# -------------------------------------------------------------------


class TestSD35Profiles:

    def test_sd35_profile_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("sd35_medium", "sd35")
        assert p.workload_type == "t2i"
        assert p.defaults["height"] == 512
        assert p.defaults["num_frames"] == 1


# -------------------------------------------------------------------
# LingBotWorld profile integration (dual guidance)
# -------------------------------------------------------------------


class TestLingBotWorldProfiles:

    def test_lingbotworld_dual_guidance(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("lingbotworld_i2v", "lingbotworld")
        stage = p.stages[0]
        assert "guidance_scale_2" in stage.allowed_overrides
        assert "boundary_ratio" in stage.allowed_overrides

    def test_lingbotworld_override_validation(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("lingbotworld_i2v", "lingbotworld")
        validate_stage_overrides(
            p, {"denoise": {"boundary_ratio": 0.95}})
        with pytest.raises(ConfigValidationError):
            validate_stage_overrides(
                p, {"denoise": {"height": 720}})


# -------------------------------------------------------------------
# Remaining single-profile families
# -------------------------------------------------------------------


class TestSingleProfileFamilies:

    def test_hyworld_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("hyworld_t2v", "hyworld")
        assert p.workload_type == "t2v"

    def test_gamecraft_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("gamecraft_i2v", "gamecraft")
        assert p.workload_type == "i2v"
        assert p.defaults["num_frames"] == 33

    def test_gen3c_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("gen3c_cosmos_7b", "gen3c")
        assert p.defaults["num_inference_steps"] == 35

    def test_matrixgame_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        p = get_profile("matrixgame_i2v", "matrixgame")
        assert p.defaults["num_inference_steps"] == 3
        assert p.defaults["fps"] == 25

    def test_longcat_profiles_registered(self) -> None:
        import fastvideo.registry  # noqa: F401
        profiles = get_profiles_for_family("longcat")
        names = {p.name for p in profiles}
        assert names == {
            "longcat_t2v", "longcat_i2v", "longcat_vc"
        }


# -------------------------------------------------------------------
# Cross-family: total profile count
# -------------------------------------------------------------------


class TestProfileCountIntegrity:

    def test_total_profile_count(self) -> None:
        """All 37 profiles from 13 families are registered."""
        import fastvideo.registry  # noqa: F401
        names = get_all_profile_names()
        assert len(names) == 37
