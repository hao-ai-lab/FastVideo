import pytest

from dreamverse.creation_capabilities import (
    capabilities_for_model,
    lobby_capabilities_as_dict,
    validate_lobby_creation_config,
)


def test_lobby_capabilities_include_all_registry_models():
    caps = lobby_capabilities_as_dict()
    assert set(caps["model_ids"]) == {"fast-ltx2", "fast-ltx23", "fast-h3"}
    assert "fl2va" not in caps["generation_modes"]
    assert "4k" not in caps["resolutions"]


def test_fast_h3_capabilities_use_fixed_geometry():
    h3_caps = capabilities_for_model("fast-h3")
    assert h3_caps.generation_modes == frozenset({"t2va", "ref2va"})
    assert h3_caps.aspect_ratios == frozenset({"16:9"})
    assert h3_caps.resolutions == frozenset({"720p"})


def test_validate_lobby_creation_config_accepts_supported_t2va():
    validate_lobby_creation_config(
        model_id="fast-ltx23",
        generation_mode="t2va",
        aspect_ratio="16:9",
        resolution="1080p",
        duration_sec=5,
    )


def test_validate_lobby_creation_config_accepts_fast_h3():
    validate_lobby_creation_config(
        model_id="fast-h3",
        generation_mode="ref2va",
        aspect_ratio="16:9",
        resolution="720p",
        duration_sec=10,
    )


def test_validate_lobby_creation_config_rejects_fl2va():
    with pytest.raises(ValueError, match="FL2VA"):
        validate_lobby_creation_config(
            model_id="fast-ltx23",
            generation_mode="fl2va",
            aspect_ratio="16:9",
            resolution="720p",
            duration_sec=5,
        )


def test_validate_lobby_creation_config_rejects_4k():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        validate_lobby_creation_config(
            model_id="fast-ltx2",
            generation_mode="t2va",
            aspect_ratio="16:9",
            resolution="4k",
            duration_sec=10,
        )


def test_validate_lobby_creation_config_rejects_invalid_h3_aspect():
    with pytest.raises(ValueError, match="Unsupported aspect_ratio"):
        validate_lobby_creation_config(
            model_id="fast-h3",
            generation_mode="t2va",
            aspect_ratio="9:16",
            resolution="720p",
            duration_sec=5,
        )
