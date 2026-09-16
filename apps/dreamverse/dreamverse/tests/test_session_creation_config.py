import pytest

from dreamverse.session_creation_config import (
    duration_sec_to_segment_cap,
    parse_session_creation_config,
    resolve_frame_size,
    validate_generation_mode_assets,
)


def test_parse_session_creation_config_defaults():
    config = parse_session_creation_config({})
    assert config.model_id == "fast-ltx23"
    assert config.generation_mode == "t2va"
    assert config.aspect_ratio == "16:9"
    assert config.resolution == "720p"
    assert config.duration_sec == 5
    assert config.generation_segment_cap == 1


def test_parse_session_creation_config_maps_duration_to_segment_cap():
    config = parse_session_creation_config(
        {
            "model_id": "fast-ltx2",
            "generation_mode": "ref2va",
            "aspect_ratio": "9:16",
            "resolution": "480p",
            "duration_sec": 15,
        },
    )
    assert config.model_id == "fast-ltx2"
    assert config.generation_mode == "ref2va"
    assert config.generation_segment_cap == 3
    assert config.frame_width >= 480
    assert config.frame_height >= 480


def test_resolve_frame_size_uses_model_default_for_1080p_landscape():
    width, height = resolve_frame_size("16:9", "1080p")
    assert (width, height) == (1920, 1088)


def test_duration_sec_to_segment_cap_respects_global_cap():
    assert duration_sec_to_segment_cap(15, global_cap=2) == 2


def test_parse_session_creation_config_accepts_fast_h3():
    config = parse_session_creation_config(
        {
            "model_id": "fast-h3",
            "generation_mode": "t2va",
            "aspect_ratio": "16:9",
            "resolution": "720p",
            "duration_sec": 10,
        },
    )
    assert config.model_id == "fast-h3"
    assert config.generation_mode == "t2va"
    assert config.generation_segment_cap == 2


def test_parse_session_creation_config_rejects_fl2va():
    with pytest.raises(ValueError, match="FL2VA"):
        parse_session_creation_config(
            {
                "generation_mode": "fl2va",
                "aspect_ratio": "16:9",
                "resolution": "720p",
                "duration_sec": 5,
            },
        )


def test_parse_session_creation_config_rejects_4k():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        parse_session_creation_config(
            {
                "generation_mode": "t2va",
                "aspect_ratio": "16:9",
                "resolution": "4k",
                "duration_sec": 5,
            },
        )


def test_validate_generation_mode_assets():
    validate_generation_mode_assets("t2va", has_initial_image=False, has_last_frame_image=False)
    with pytest.raises(ValueError, match="Ref2VA"):
        validate_generation_mode_assets("ref2va", has_initial_image=False, has_last_frame_image=False)
